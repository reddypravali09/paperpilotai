
import io

from pptx import Presentation

from presentation.pptx_builder import (
    build_presentation,
    _add_title_slide,
    _add_content_slide,
    _add_footer,
)
from schemas.slides import SlideData, SlideContent


def create_sample_slide_data():
    """Create valid sample data with at least 3 content slides."""
    return SlideData(
        paper_title="Attention Is All You Need",
        authors="Ashish Vaswani et al.",
        slides=[
            SlideContent(
                title="Introduction",
                bullets=[
                    "Introduces the Transformer architecture",
                    "Uses attention mechanisms",
                ],
                speaker_notes="Explain the motivation behind Transformers.",
            ),
            SlideContent(
                title="Methodology",
                bullets=[
                    "Uses self-attention",
                    "Removes recurrent layers",
                ],
                speaker_notes="Explain the model architecture.",
            ),
            SlideContent(
                title="Conclusion",
                bullets=[
                    "Transformers improve sequence modelling",
                    "Attention is the core mechanism",
                ],
                speaker_notes="Summarize the research findings.",
            ),
        ],
    )


def test_build_presentation_returns_bytesio():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)

    assert isinstance(result, io.BytesIO)


def test_build_presentation_creates_correct_number_of_slides():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    # One title slide + three content slides
    assert len(prs.slides) == 4


def test_title_slide_contains_paper_title():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    title_slide_text = " ".join(
        shape.text
        for shape in prs.slides[0].shapes
        if shape.has_text_frame
    )

    assert "Attention Is All You Need" in title_slide_text


def test_title_slide_contains_authors():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    title_slide_text = " ".join(
        shape.text
        for shape in prs.slides[0].shapes
        if shape.has_text_frame
    )

    assert "Ashish Vaswani et al." in title_slide_text


def test_content_slide_contains_title_and_bullets():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    content_slide_text = " ".join(
        shape.text
        for shape in prs.slides[1].shapes
        if shape.has_text_frame
    )

    assert "Introduction" in content_slide_text
    assert "Introduces the Transformer architecture" in content_slide_text
    assert "Uses attention mechanisms" in content_slide_text


def test_content_slide_contains_speaker_notes():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    notes = prs.slides[1].notes_slide.notes_text_frame.text

    assert "Explain the motivation behind Transformers." in notes


def test_content_slide_without_notes():
    slide_data = SlideData(
        paper_title="Sample Research Paper",
        slides=[
            SlideContent(
                title="Introduction",
                bullets=["First point", "Second point"],
                speaker_notes=None,
            ),
            SlideContent(
                title="Methodology",
                bullets=["Third point"],
            ),
            SlideContent(
                title="Conclusion",
                bullets=["Final point"],
            ),
        ],
    )

    result = build_presentation(slide_data)
    prs = Presentation(result)

    # One title slide + three content slides
    assert len(prs.slides) == 4


def test_title_slide_without_authors():
    slide_data = SlideData(
        paper_title="Sample Research Paper",
        slides=[
            SlideContent(
                title="Introduction",
                bullets=["First point"],
            ),
            SlideContent(
                title="Methodology",
                bullets=["Second point"],
            ),
            SlideContent(
                title="Conclusion",
                bullets=["Third point"],
            ),
        ],
    )

    result = build_presentation(slide_data)
    prs = Presentation(result)

    title_slide_text = " ".join(
        shape.text
        for shape in prs.slides[0].shapes
        if shape.has_text_frame
    )

    assert "Sample Research Paper" in title_slide_text


def test_footer_is_added_to_slide():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)
    prs = Presentation(result)

    footer_text = " ".join(
        shape.text
        for shape in prs.slides[1].shapes
        if shape.has_text_frame
    )

    assert "PaperPilot AI" in footer_text


def test_presentation_buffer_is_readable():
    slide_data = create_sample_slide_data()

    result = build_presentation(slide_data)

    # Verify that the generated PowerPoint can be opened
    prs = Presentation(result)

    assert len(prs.slides) == 4
