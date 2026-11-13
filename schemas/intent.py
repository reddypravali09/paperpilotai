"""
Intent Classification Schema
=============================
Pydantic models for structured intent detection.
Used by the Intent Router to classify user queries
and determine whether ChromaDB retrieval is needed.
"""

from enum import Enum
from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    """Supported user intent categories."""
    SUMMARIZE = "summarize"
    CONCEPTS = "concepts"
    SLIDES = "slides"
    VIVA = "viva"
    GENERAL = "general"


# Intents that require ChromaDB retrieval
RETRIEVAL_INTENTS = {
    IntentEnum.SUMMARIZE,
    IntentEnum.CONCEPTS,
    IntentEnum.SLIDES,
    IntentEnum.VIVA,
}


class IntentClassification(BaseModel):
    """Structured output from the intent classifier.

    Fields:
        intent: The classified intent category
        reasoning: Brief explanation for the classification
        needs_retrieval: Whether this intent requires vector store retrieval
    """
    intent: IntentEnum = Field(
        description="The classified intent of the user's query"
    )
    reasoning: str = Field(
        description="Brief reason for the classification"
    )
    needs_retrieval: bool = Field(
        description="Whether this intent requires retrieving paper content from the vector store"
    )
