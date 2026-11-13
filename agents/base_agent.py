"""
Base Agent
===========
Abstract base class for all specialist agents.
Implements the shared LCEL pipeline pattern:
  RunnableLambda(retrieve_and_format) | ChatPromptTemplate | ChatGroq | StrOutputParser

Features:
- Conditional retrieval via needs_retrieval flag
- Source metadata propagation
- Insufficient retrieval guard (< MIN_RETRIEVAL_CHUNKS)
- Reflection/verification chain
- Chat history support via MessagesPlaceholder
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from config.settings import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_MAX_RETRIES,
    GROQ_API_KEY,
    RETRIEVAL_TOP_K,
    MIN_RETRIEVAL_CHUNKS,
)
from core.vector_store import similarity_search
from prompts.reflection_prompts import (
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
)
from utils.helpers import setup_logger, format_source_pages

logger = setup_logger(__name__)


class BaseAgent:
    """Base class for all specialist agents.

    Each agent is an LCEL chain: prompt | llm | parser
    with optional retrieval and reflection steps.
    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        top_k: int = RETRIEVAL_TOP_K,
        agent_name: str = "BaseAgent",
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.top_k = top_k
        self.agent_name = agent_name
        self._retrieval_metadata: dict = {}

        self.llm = ChatGroq(
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
            api_key=GROQ_API_KEY,
        )

        self.chain = self._build_chain()
        logger.info(f"{agent_name} initialized with model={GENERATION_MODEL}")

    # ------------------------------------------------------------------
    # LCEL pipeline steps
    # ------------------------------------------------------------------

    def _retrieve_and_format(self, inputs: dict) -> dict:
        """RunnableLambda step: conditionally retrieves context from ChromaDB.

        Only retrieves when *needs_retrieval=True* **and** *paper_hash* is
        provided.  Stores retrieval metadata (including the rendered context
        string) as a side-effect so that :pymeth:`run` can forward it to the
        reflection verifier.
        """
        needs_retrieval: bool = inputs.get("needs_retrieval", True)
        paper_hash: str | None = inputs.get("paper_hash")
        query: str = inputs.get("query", "")

        context = "No context available."
        sources: list[int] = []
        chunk_count = 0
        insufficient = False

        if needs_retrieval and paper_hash:
            try:
                docs = similarity_search(paper_hash, query, k=self.top_k)
                chunk_count = len(docs)

                if chunk_count < MIN_RETRIEVAL_CHUNKS:
                    insufficient = True
                    if docs:
                        context = (
                            "LIMITED CONTEXT (insufficient retrieval):\n"
                            + "\n\n".join(
                                f"[Page {d.metadata.get('page', '?')}]: {d.page_content}"
                                for d in docs
                            )
                        )
                    else:
                        context = (
                            "No relevant content found in the paper for this query."
                        )
                else:
                    context = "\n\n".join(
                        f"[Page {d.metadata.get('page', '?')}]: {d.page_content}"
                        for d in docs
                    )
                    sources = sorted(
                        set(d.metadata.get("page", 0) for d in docs)
                    )
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                context = "Error retrieving content from the document."

        # Persist metadata — including the rendered context — so that run()
        # can later pass it to verify_response().
        self._retrieval_metadata = {
            "sources": sources,
            "chunk_count": chunk_count,
            "needs_retrieval": needs_retrieval,
            "insufficient_retrieval": insufficient,
            "context": context,
        }

        # Build the dict that flows into the ChatPromptTemplate.
        # We forward any extra keys the caller passed (e.g. slide_count)
        # while avoiding collisions with the canonical keys.
        _reserved = {
            "query",
            "context",
            "chat_history",
            "paper_hash",
            "needs_retrieval",
            "source_pages",
            "chunk_count",
        }

        return {
            "query": query,
            "context": context,
            "chat_history": inputs.get("chat_history", []),
            "source_pages": format_source_pages(sources),
            "chunk_count": str(chunk_count),
            **{k: v for k, v in inputs.items() if k not in _reserved},
        }

    def _build_chain(self):
        """Build the LCEL chain: retrieve ➜ prompt ➜ llm ➜ parse."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", self.user_prompt),
            ]
        )

        return (
            RunnableLambda(self._retrieve_and_format)
            | prompt
            | self.llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Reflection / verification
    # ------------------------------------------------------------------

    def verify_response(self, context: str, response: str) -> dict:
        """Run reflection chain to verify response grounding.

        Uses a separate LLM call to check whether the response is
        grounded in the retrieved context.

        Args:
            context: The rendered context string used for generation.
            response: The LLM-generated answer to verify.

        Returns:
            Dict with ``verified`` (bool) and ``status`` (str).
        """
        try:
            reflection_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", REFLECTION_SYSTEM_PROMPT),
                    ("human", REFLECTION_USER_PROMPT),
                ]
            )
            reflection_chain = (
                reflection_prompt | self.llm | StrOutputParser()
            )
            verification = reflection_chain.invoke(
                {
                    "context": context,
                    "response": response,
                }
            )
            is_verified = "yes" in verification.lower()
            return {
                "verified": is_verified,
                "status": (
                    "✅ Response grounded in retrieved content"
                    if is_verified
                    else "⚠️ Some claims may not be fully supported by the source"
                ),
            }
        except Exception as e:
            logger.warning(f"Reflection verification failed: {e}")
            return {"verified": True, "status": "✅ Verification skipped"}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        paper_hash: str | None = None,
        chat_history: list | None = None,
        needs_retrieval: bool = True,
        **kwargs,
    ) -> dict:
        """Execute the agent chain and return response with metadata.

        Args:
            query: User's question.
            paper_hash: SHA-256 hash prefix of the current paper.
            chat_history: List of previous messages for context.
            needs_retrieval: Whether to retrieve from ChromaDB.
            **kwargs: Additional inputs (e.g., ``slide_count``).

        Returns:
            Dict with keys: ``response``, ``sources``, ``chunk_count``,
            ``needs_retrieval``, ``insufficient_retrieval``, ``verification``.
        """
        inputs: dict = {
            "query": query,
            "paper_hash": paper_hash,
            "chat_history": chat_history or [],
            "needs_retrieval": needs_retrieval,
            **kwargs,
        }

        try:
            response = self.chain.invoke(inputs)
        except Exception as e:
            logger.error(f"{self.agent_name} chain error: {e}")
            return {
                "response": (
                    f"I encountered an error while processing your request: {e}"
                ),
                "sources": [],
                "chunk_count": 0,
                "needs_retrieval": needs_retrieval,
                "insufficient_retrieval": False,
                "verification": {
                    "verified": False,
                    "status": "❌ Error occurred",
                },
            }

        metadata = self._retrieval_metadata.copy()

        # Run reflection verification for retrieval-based responses that
        # returned enough chunks.
        verification: dict = {
            "verified": True,
            "status": "✅ No verification needed",
        }
        if (
            needs_retrieval
            and metadata.get("chunk_count", 0) >= MIN_RETRIEVAL_CHUNKS
        ):
            verification = self.verify_response(
                metadata.get("context", ""),
                response,
            )

        return {
            "response": response,
            "sources": metadata.get("sources", []),
            "chunk_count": metadata.get("chunk_count", 0),
            "needs_retrieval": needs_retrieval,
            "insufficient_retrieval": metadata.get(
                "insufficient_retrieval", False
            ),
            "verification": verification,
        }
