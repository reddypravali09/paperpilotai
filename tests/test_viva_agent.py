
from agents import viva_agent
from config.settings import RETRIEVAL_TOP_K
from prompts.viva_prompts import (
    VIVA_SYSTEM_PROMPT,
    VIVA_USER_PROMPT,
)


def test_create_viva_agent_uses_correct_configuration(monkeypatch):
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

    monkeypatch.setattr(viva_agent, "BaseAgent", FakeBaseAgent)

    agent = viva_agent.create_viva_agent()

    assert isinstance(agent, FakeBaseAgent)
    assert captured["system_prompt"] == VIVA_SYSTEM_PROMPT
    assert captured["user_prompt"] == VIVA_USER_PROMPT
    assert captured["top_k"] == RETRIEVAL_TOP_K
    assert captured["agent_name"] == "VivaAgent"


def test_viva_prompt_contains_grounding_rules():
    assert "EXCLUSIVELY" in VIVA_SYSTEM_PROMPT
    assert "Do NOT invent paper-specific information." in VIVA_SYSTEM_PROMPT
    assert "provided context" in VIVA_SYSTEM_PROMPT


def test_viva_prompt_contains_difficulty_levels():
    assert "Basic" in VIVA_SYSTEM_PROMPT
    assert "Intermediate" in VIVA_SYSTEM_PROMPT
    assert "Advanced" in VIVA_SYSTEM_PROMPT


def test_viva_prompt_contains_question_types():
    question_types = [
        "Conceptual",
        "Methodological",
        "Analytical",
        "Critical",
    ]

    for question_type in question_types:
        assert question_type in VIVA_SYSTEM_PROMPT


def test_viva_prompt_specifies_question_count():
    assert "8-10 QUESTIONS" in VIVA_SYSTEM_PROMPT


def test_viva_user_prompt_contains_required_placeholders():
    required_placeholders = [
        "{chunk_count}",
        "{source_pages}",
        "{context}",
        "{query}",
    ]

    for placeholder in required_placeholders:
        assert placeholder in VIVA_USER_PROMPT
