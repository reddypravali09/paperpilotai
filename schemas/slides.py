"""
Slide Generation Schema
========================
Pydantic models for validating structured slide data
produced by the Slide Generator Agent.
Used to ensure valid JSON before PPTX generation.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class SlideContent(BaseModel):
    """A single slide in the presentation."""
    title: str = Field(description="Slide title", min_length=1)
    bullets: list[str] = Field(description="Bullet points for this slide", min_length=1)
    speaker_notes: Optional[str] = Field(default=None, description="Optional speaker notes")


class SlideData(BaseModel):
    """Complete presentation data, validated before PPTX generation.

    Validates that:
    - paper_title is non-empty
    - At least 3 slides are present
    - Each slide has a title and at least one bullet point
    """
    paper_title: str = Field(description="Title of the research paper", min_length=1)
    authors: Optional[str] = Field(default=None, description="Paper authors")
    slides: list[SlideContent] = Field(description="List of slide contents")

    @field_validator("slides")
    @classmethod
    def validate_slide_count(cls, v):
        if len(v) < 3:
            raise ValueError(f"Expected at least 3 slides, got {len(v)}")
        return v
