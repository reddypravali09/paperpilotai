"""
Viva Agent
===========
Generates examiner-style viva questions with model answers.
Retrieves top-7 chunks relevant to methodology and results.
"""

from agents.base_agent import BaseAgent
from prompts.viva_prompts import VIVA_SYSTEM_PROMPT, VIVA_USER_PROMPT
from config.settings import RETRIEVAL_TOP_K


def create_viva_agent() -> BaseAgent:
    """Create and return the Viva Agent."""
    return BaseAgent(
        system_prompt=VIVA_SYSTEM_PROMPT,
        user_prompt=VIVA_USER_PROMPT,
        top_k=RETRIEVAL_TOP_K,
        agent_name="VivaAgent",
    )
