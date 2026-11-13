
from agents import summarizer
from config.settings import SUMMARY_TOP_K
from prompts.summarizer_prompts import (
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_PROMPT,
)


def test_create_summarizer_uses_correct_configuration(monkeypatch):
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

    monkeypatch.setattr(summarizer, "BaseAgent", FakeBaseAgent)

    agent = summarizer.create_summarizer()

    assert isinstance(agent, FakeBaseAgent)
    assert captured["system_prompt"] == SUMMARIZER_SYSTEM_PROMPT
    assert captured["user_prompt"] == SUMMARIZER_USER_PROMPT
    assert captured["top_k"] == SUMMARY_TOP_K
    assert captured["agent_name"] == "Summarizer"


def test_summarizer_prompts_include_grounding_rules():
    assert "EXCLUSIVELY" in SUMMARIZER_SYSTEM_PROMPT
    assert "Do NOT invent" in SUMMARIZER_SYSTEM_PROMPT
    assert "Not available in the provided sections." in SUMMARIZER_SYSTEM_PROMPT


def test_summarizer_user_prompt_contains_required_placeholders():
    assert "{chunk_count}" in SUMMARIZER_USER_PROMPT
    assert "{source_pages}" in SUMMARIZER_USER_PROMPT
    assert "{context}" in SUMMARIZER_USER_PROMPT
    assert "{query}" in SUMMARIZER_USER_PROMPT
