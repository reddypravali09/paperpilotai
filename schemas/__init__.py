# schemas/__init__.py
"""PaperPilot AI — Pydantic schema models."""

from schemas.intent import IntentEnum, IntentClassification, RETRIEVAL_INTENTS
from schemas.slides import SlideContent, SlideData

__all__ = [
    "IntentEnum",
    "IntentClassification",
    "RETRIEVAL_INTENTS",
    "SlideContent",
    "SlideData",
]
