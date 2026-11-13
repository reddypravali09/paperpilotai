"""
Slide Generator Agent
======================
Creates presentation-ready slides with Pydantic JSON validation.
Retrieves top-10 chunks.  Auto-retries on invalid JSON up to
MAX_SLIDE_RETRIES times.  Outputs validated SlideData for PPTX
generation.
"""

import json
import re

from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config.settings import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_MAX_RETRIES,
    GROQ_API_KEY,
    SLIDE_TOP_K,
    MIN_RETRIEVAL_CHUNKS,
    DEFAULT_SLIDE_COUNT,
    MAX_SLIDE_RETRIES,
)
from core.vector_store import similarity_search
from schemas.slides import SlideData
from prompts.slide_prompts import (
    SLIDE_SYSTEM_PROMPT,
    SLIDE_USER_PROMPT,
    SLIDE_CORRECTION_PROMPT,
)
from utils.helpers import setup_logger, format_source_pages

logger = setup_logger(__name__)


class SlideGeneratorAgent:
    """Specialized agent for slide generation with JSON validation + retry."""

    def __init__(self):
        self.agent_name = "SlideGenerator"
        self.top_k = SLIDE_TOP_K
        self._retrieval_metadata: dict = {}

        self.llm = ChatGroq(
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
            api_key=GROQ_API_KEY,
        )
        logger.info(f"{self.agent_name} initialized")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str, paper_hash: str | None) -> dict:
        """Retrieve relevant chunks for slide generation."""
        context = "No context available."
        sources: list[int] = []
        chunk_count = 0
        insufficient = False

        if paper_hash:
            try:
                docs = similarity_search(paper_hash, query, k=self.top_k)
                chunk_count = len(docs)

                if chunk_count < MIN_RETRIEVAL_CHUNKS:
                    insufficient = True
                    if docs:
                        context = (
                            "LIMITED CONTEXT:\n"
                            + "\n\n".join(
                                f"[Page {d.metadata.get('page', '?')}]: "
                                f"{d.page_content}"
                                for d in docs
                            )
                        )
                else:
                    context = "\n\n".join(
                        f"[Page {d.metadata.get('page', '?')}]: "
                        f"{d.page_content}"
                        for d in docs
                    )
                    sources = sorted(
                        set(d.metadata.get("page", 0) for d in docs)
                    )
            except Exception as e:
                logger.error(f"Retrieval error: {e}")

        self._retrieval_metadata = {
            "sources": sources,
            "chunk_count": chunk_count,
            "insufficient_retrieval": insufficient,
            "context": context,
        }
        return {
            "context": context,
            "sources": sources,
            "chunk_count": chunk_count,
            "source_pages": format_source_pages(sources),
        }

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Strip markdown code fences to extract valid JSON."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    # ------------------------------------------------------------------
    # Chain builders
    # ------------------------------------------------------------------

    def _build_slide_chain(self):
        """Build LCEL chain for initial slide generation."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SLIDE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", SLIDE_USER_PROMPT),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _build_correction_chain(self):
        """Build LCEL chain for retry with correction prompt."""
        correction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SLIDE_SYSTEM_PROMPT),
                ("human", SLIDE_CORRECTION_PROMPT),
            ]
        )
        return correction_prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        paper_hash: str | None = None,
        chat_history: list | None = None,
        needs_retrieval: bool = True,
        slide_count: int = DEFAULT_SLIDE_COUNT,
        **kwargs,
    ) -> dict:
        """Generate slides with JSON validation and retry logic.

        Args:
            query: User's request.
            paper_hash: Paper identifier for retrieval.
            chat_history: Previous messages.
            needs_retrieval: Should always be True for slides.
            slide_count: Number of slides to generate (8-12).

        Returns:
            Dict with ``response`` (markdown preview),
            ``slide_data`` (:class:`SlideData` or *None*),
            and retrieval metadata.
        """
        # Step 1: Retrieve context
        retrieval = self._retrieve_context(query, paper_hash)

        # Step 2: Build chains
        chain = self._build_slide_chain()
        correction_chain = self._build_correction_chain()

        chain_inputs: dict = {
            "query": query,
            "context": retrieval["context"],
            "chat_history": chat_history or [],
            "source_pages": retrieval["source_pages"],
            "chunk_count": str(retrieval["chunk_count"]),
            "slide_count": str(slide_count),
        }

        # Step 3: Generate and validate with retry
        slide_data: SlideData | None = None
        last_error: Exception | None = None

        for attempt in range(MAX_SLIDE_RETRIES + 1):
            try:
                if attempt == 0:
                    raw_json = chain.invoke(chain_inputs)
                else:
                    raw_json = correction_chain.invoke(
                        {**chain_inputs, "error": str(last_error)}
                    )

                cleaned = self._clean_json_response(raw_json)
                slide_data = SlideData.model_validate_json(cleaned)
                logger.info(
                    f"Slide JSON validated on attempt {attempt + 1}"
                )
                break

            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    f"Slide JSON validation failed "
                    f"(attempt {attempt + 1}): {e}"
                )
                if attempt >= MAX_SLIDE_RETRIES:
                    logger.error("All slide generation retries exhausted")

        # Step 4: Build response
        metadata = self._retrieval_metadata.copy()

        if slide_data:
            preview_lines = [
                f"## 📊 Presentation: {slide_data.paper_title}\n"
            ]
            if slide_data.authors:
                preview_lines.append(
                    f"**Authors:** {slide_data.authors}\n"
                )
            preview_lines.append(f"**Slides:** {len(slide_data.slides)}\n")

            for i, slide in enumerate(slide_data.slides, 1):
                preview_lines.append(f"\n### Slide {i}: {slide.title}")
                for bullet in slide.bullets:
                    preview_lines.append(f"- {bullet}")

            response = "\n".join(preview_lines)
        else:
            response = (
                "I was unable to generate valid slide content. "
                f"The last error was: {last_error}\n\n"
                "Please try again or rephrase your request."
            )

        return {
            "response": response,
            "slide_data": slide_data,
            "sources": metadata.get("sources", []),
            "chunk_count": metadata.get("chunk_count", 0),
            "needs_retrieval": True,
            "insufficient_retrieval": metadata.get(
                "insufficient_retrieval", False
            ),
            "verification": {
                "verified": slide_data is not None,
                "status": (
                    "✅ Slides generated and validated"
                    if slide_data
                    else "❌ Slide generation failed"
                ),
            },
        }


def create_slide_generator() -> SlideGeneratorAgent:
    """Create and return the Slide Generator agent."""
    return SlideGeneratorAgent()
