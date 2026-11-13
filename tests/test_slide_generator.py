
import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agents import slide_generator
from agents.slide_generator import (
    SlideGeneratorAgent,
    create_slide_generator,
)
from config.settings import SLIDE_TOP_K, MAX_SLIDE_RETRIES
from prompts.slide_prompts import (
    SLIDE_SYSTEM_PROMPT,
    SLIDE_USER_PROMPT,
    SLIDE_CORRECTION_PROMPT,
)
from schemas.slides import SlideData


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(
        slide_generator,
        "ChatGroq",
        lambda **kwargs: MagicMock(),
    )
    return SlideGeneratorAgent()


def valid_slide_json():
    return json.dumps({
        "paper_title": "AI Research",
        "authors": "John Doe",
        "slides": [
            {
                "title": f"Slide {i}",
                "bullets": ["Point one", "Point two"],
                "speaker_notes": "Explain the slide."
            }
            for i in range(1, 4)
        ]
    })


# 1. Test agent creation
def test_create_slide_generator(monkeypatch):
    monkeypatch.setattr(
        slide_generator,
        "ChatGroq",
        lambda **kwargs: MagicMock(),
    )

    generator = create_slide_generator()

    assert isinstance(generator, SlideGeneratorAgent)


# 2. Test agent configuration
def test_slide_generator_configuration(agent):
    assert agent.agent_name == "SlideGenerator"
    assert agent.top_k == SLIDE_TOP_K


# 3. Test cleaning JSON with markdown fences
def test_clean_json_response():
    raw = '```json\n{"title": "Research"}\n```'

    cleaned = SlideGeneratorAgent._clean_json_response(raw)

    assert cleaned == '{"title": "Research"}'


# 4. Test cleaning JSON without markdown fences
def test_clean_json_without_code_fence():
    raw = '{"title": "Research"}'

    cleaned = SlideGeneratorAgent._clean_json_response(raw)

    assert cleaned == raw


# 5. Test valid slide data
def test_slide_data_validates_correct_json():
    data = SlideData.model_validate_json(valid_slide_json())

    assert data.paper_title == "AI Research"
    assert len(data.slides) == 3


# 6. Test rejection of fewer than three slides
def test_slide_data_rejects_fewer_than_three_slides():
    invalid_data = {
        "paper_title": "AI Research",
        "slides": [
            {"title": "Slide 1", "bullets": ["Point one"]},
            {"title": "Slide 2", "bullets": ["Point two"]},
        ]
    }

    with pytest.raises(ValidationError):
        SlideData.model_validate(invalid_data)


# 7. Test rejection of an empty slide title
def test_slide_data_rejects_empty_title():
    invalid_data = {
        "paper_title": "AI Research",
        "slides": [
            {"title": "", "bullets": ["Point one"]},
            {"title": "Slide 2", "bullets": ["Point two"]},
            {"title": "Slide 3", "bullets": ["Point three"]},
        ]
    }

    with pytest.raises(ValidationError):
        SlideData.model_validate(invalid_data)


# 8. Test required prompt placeholders
def test_slide_prompts_contain_required_placeholders():
    assert "{slide_count}" in SLIDE_SYSTEM_PROMPT
    assert "{slide_count}" in SLIDE_USER_PROMPT
    assert "{context}" in SLIDE_USER_PROMPT
    assert "{query}" in SLIDE_USER_PROMPT
    assert "{source_pages}" in SLIDE_USER_PROMPT
    assert "{chunk_count}" in SLIDE_USER_PROMPT
    assert "{error}" in SLIDE_CORRECTION_PROMPT


# 9. Test successful slide generation
def test_slide_generation_success(agent, monkeypatch):

    def fake_retrieve_context(query, paper_hash):
        metadata = {
            "sources": [1, 2],
            "chunk_count": 2,
            "insufficient_retrieval": False,
            "context": "Research context",
        }

        agent._retrieval_metadata = metadata.copy()

        return {
            "context": "Research context",
            "sources": [1, 2],
            "chunk_count": 2,
            "source_pages": "1, 2",
        }

    monkeypatch.setattr(
        agent,
        "_retrieve_context",
        fake_retrieve_context,
    )

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = valid_slide_json()

    monkeypatch.setattr(
        agent,
        "_build_slide_chain",
        lambda: mock_chain,
    )

    monkeypatch.setattr(
        agent,
        "_build_correction_chain",
        lambda: MagicMock(),
    )

    result = agent.run(
        query="Create slides",
        paper_hash="test-paper",
    )

    assert result["slide_data"] is not None
    assert result["slide_data"].paper_title == "AI Research"
    assert result["chunk_count"] == 2
    assert result["sources"] == [1, 2]
    assert result["verification"]["verified"] is True
    assert "Presentation: AI Research" in result["response"]


# 10. Test retry after invalid JSON
def test_slide_generation_retries_invalid_json(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_retrieve_context",
        lambda query, paper_hash: {
            "context": "Research context",
            "sources": [],
            "chunk_count": 0,
            "source_pages": "Not available",
        },
    )

    initial_chain = MagicMock()
    initial_chain.invoke.return_value = "invalid JSON"

    correction_chain = MagicMock()
    correction_chain.invoke.return_value = valid_slide_json()

    monkeypatch.setattr(
        agent,
        "_build_slide_chain",
        lambda: initial_chain,
    )

    monkeypatch.setattr(
        agent,
        "_build_correction_chain",
        lambda: correction_chain,
    )

    result = agent.run(query="Create slides")

    assert result["slide_data"] is not None
    assert correction_chain.invoke.call_count == 1


# 11. Test failure after all retries
def test_slide_generation_fails_after_all_retries(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_retrieve_context",
        lambda query, paper_hash: {
            "context": "Research context",
            "sources": [],
            "chunk_count": 0,
            "source_pages": "Not available",
        },
    )

    initial_chain = MagicMock()
    initial_chain.invoke.return_value = "invalid JSON"

    correction_chain = MagicMock()
    correction_chain.invoke.return_value = "still invalid"

    monkeypatch.setattr(
        agent,
        "_build_slide_chain",
        lambda: initial_chain,
    )

    monkeypatch.setattr(
        agent,
        "_build_correction_chain",
        lambda: correction_chain,
    )

    result = agent.run(query="Create slides")

    assert result["slide_data"] is None
    assert result["verification"]["verified"] is False
    assert "unable to generate valid slide content" in result["response"]
    assert correction_chain.invoke.call_count == MAX_SLIDE_RETRIES
