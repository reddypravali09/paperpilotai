"""
Summarizer Agent
=================
Produces structured research paper summaries using LCEL chain.
Retrieves top-10 chunks for comprehensive coverage.
"""

from agents.base_agent import BaseAgent
from prompts.summarizer_prompts import (
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_PROMPT,
)
from config.settings import SUMMARY_TOP_K


def create_summarizer() -> BaseAgent:
    """Create and return the Summarizer agent."""
    return BaseAgent(
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
        user_prompt=SUMMARIZER_USER_PROMPT,
        top_k=SUMMARY_TOP_K,
        agent_name="Summarizer",
    )
