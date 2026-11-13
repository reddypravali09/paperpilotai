
from agents import concept_extractor
from config.settings import RETRIEVAL_TOP_K
from prompts.concept_prompts import (
    CONCEPT_SYSTEM_PROMPT,
    CONCEPT_USER_PROMPT,
)


def test_create_concept_extractor_uses_correct_configuration(monkeypatch):
    captured = {}

    class FakeBaseAgent:
        def __init__(
            self,
            system_prompt,
            user_prompt,
            top_k,
            agent_name,
        ):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["top_k"] = top_k
            captured["agent_name"] = agent_name

    monkeypatch.setattr(
        concept_extractor,
        "BaseAgent",
        FakeBaseAgent,
    )

    agent = concept_extractor.create_concept_extractor()

    assert isinstance(agent, FakeBaseAgent)
    assert captured["system_prompt"] == CONCEPT_SYSTEM_PROMPT
    assert captured["user_prompt"] == CONCEPT_USER_PROMPT
    assert captured["top_k"] == RETRIEVAL_TOP_K
    assert captured["agent_name"] == "ConceptExtractor"


def test_concept_prompts_include_grounding_rules():
    assert "EXCLUSIVELY" in CONCEPT_SYSTEM_PROMPT
    assert "Do NOT invent or hallucinate" in CONCEPT_SYSTEM_PROMPT
    assert "Not identified in the provided sections." in CONCEPT_SYSTEM_PROMPT


def test_concept_prompt_contains_all_required_sections():
    required_sections = [
        "Novel Contributions",
        "Algorithms & Techniques",
        "Methodology",
        "Key Results",
        "Limitations",
    ]

    for section in required_sections:
        assert section in CONCEPT_SYSTEM_PROMPT


def test_concept_user_prompt_contains_required_placeholders():
    required_placeholders = [
        "{chunk_count}",
        "{source_pages}",
        "{context}",
        "{query}",
    ]

    for placeholder in required_placeholders:
        assert placeholder in CONCEPT_USER_PROMPT
