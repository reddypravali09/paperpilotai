"""
Concept Extractor Agent
========================
Extracts key contributions, algorithms, methodology, and results.
Retrieves top-7 chunks focused on methodology/results.
"""

from agents.base_agent import BaseAgent
from prompts.concept_prompts import (
    CONCEPT_SYSTEM_PROMPT,
    CONCEPT_USER_PROMPT,
)
from config.settings import RETRIEVAL_TOP_K


def create_concept_extractor() -> BaseAgent:
    """Create and return the Concept Extractor agent."""
    return BaseAgent(
        system_prompt=CONCEPT_SYSTEM_PROMPT,
        user_prompt=CONCEPT_USER_PROMPT,
        top_k=RETRIEVAL_TOP_K,
        agent_name="ConceptExtractor",
    )
