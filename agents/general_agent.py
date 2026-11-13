"""
General Agent
==============
Handles greetings, off-topic queries, and general conversation.
IMPORTANT: This agent does NOT access ChromaDB.
It proves that retrieval is genuinely conditional.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config.settings import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_MAX_RETRIES,
    GROQ_API_KEY,
)
from prompts.general_prompts import GENERAL_SYSTEM_PROMPT, GENERAL_USER_PROMPT
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class GeneralAgent:
    """Agent for non-retrieval queries.

    Does NOT touch ChromaDB.  Handles greetings, capability questions,
    and off-topic redirects.
    """

    def __init__(self):
        self.agent_name = "GeneralAgent"

        self.llm = ChatGroq(
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
            api_key=GROQ_API_KEY,
        )

        self.chain = self._build_chain()
        logger.info(f"{self.agent_name} initialized (NO retrieval)")

    def _build_chain(self):
        """Build LCEL chain without any retrieval step."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERAL_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", GENERAL_USER_PROMPT),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def run(
        self,
        query: str,
        chat_history: list | None = None,
        **kwargs,
    ) -> dict:
        """Execute the general agent chain.

        Args:
            query: User's message.
            chat_history: Previous conversation messages.
            **kwargs: Absorbed silently (e.g. paper_hash, needs_retrieval).

        Returns:
            Dict with response and metadata (no sources, no retrieval).
        """
        inputs: dict = {
            "query": query,
            "chat_history": chat_history or [],
        }

        try:
            response = self.chain.invoke(inputs)
        except Exception as e:
            logger.error(f"GeneralAgent error: {e}")
            response = "I'm sorry, I encountered an error. Please try again."

        return {
            "response": response,
            "sources": [],
            "chunk_count": 0,
            "needs_retrieval": False,
            "insufficient_retrieval": False,
            "verification": {
                "verified": True,
                "status": "💬 General response (no retrieval needed)",
            },
        }


def create_general_agent() -> GeneralAgent:
    """Create and return the General Agent."""
    return GeneralAgent()
