
from unittest.mock import MagicMock

from agents import general_agent
from prompts.general_prompts import (
    GENERAL_SYSTEM_PROMPT,
    GENERAL_USER_PROMPT,
)


def test_create_general_agent(monkeypatch):
    fake_llm = MagicMock()

    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(return_value=fake_llm),
    )

    agent = general_agent.create_general_agent()

    assert isinstance(agent, general_agent.GeneralAgent)
    assert agent.agent_name == "GeneralAgent"
    assert agent.llm is fake_llm


def test_general_agent_builds_chain_without_retrieval(monkeypatch):
    fake_llm = MagicMock()

    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(return_value=fake_llm),
    )

    agent = general_agent.create_general_agent()

    # The chain should exist without any vector-store retrieval.
    assert agent.chain is not None

    # General Agent should not have a vector store attribute.
    assert not hasattr(agent, "vector_store")


def test_general_agent_returns_expected_response(monkeypatch):
    fake_llm = MagicMock()

    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(return_value=fake_llm),
    )

    agent = general_agent.create_general_agent()

    agent.chain = MagicMock()
    agent.chain.invoke.return_value = "Hello! I am PaperPilot AI."

    result = agent.run("Hello")

    assert result["response"] == "Hello! I am PaperPilot AI."
    assert result["sources"] == []
    assert result["chunk_count"] == 0
    assert result["needs_retrieval"] is False
    assert result["insufficient_retrieval"] is False
    assert result["verification"]["verified"] is True


def test_general_agent_passes_chat_history(monkeypatch):
    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(),
    )

    agent = general_agent.create_general_agent()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = "Previous conversation remembered."

    history = ["Previous message"]

    agent.run("Continue our conversation", chat_history=history)

    agent.chain.invoke.assert_called_once_with(
        {
            "query": "Continue our conversation",
            "chat_history": history,
        }
    )


def test_general_agent_handles_llm_error(monkeypatch):
    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(),
    )

    agent = general_agent.create_general_agent()
    agent.chain = MagicMock()
    agent.chain.invoke.side_effect = Exception("API error")

    result = agent.run("Hello")

    assert result["response"] == (
        "I'm sorry, I encountered an error. Please try again."
    )
    assert result["sources"] == []
    assert result["needs_retrieval"] is False


def test_general_agent_ignores_retrieval_arguments(monkeypatch):
    monkeypatch.setattr(
        general_agent,
        "ChatGroq",
        MagicMock(),
    )

    agent = general_agent.create_general_agent()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = "I can help with general questions."

    result = agent.run(
        "What can you do?",
        paper_hash="sample_hash",
        needs_retrieval=True,
    )

    assert result["response"] == "I can help with general questions."
    assert result["needs_retrieval"] is False
    assert result["sources"] == []
    assert result["chunk_count"] == 0


def test_general_prompts_contain_capabilities():
    capabilities = [
        "Summarizing research papers",
        "Extracting key concepts",
        "Generating presentation slides",
        "Creating viva/examination questions",
    ]

    for capability in capabilities:
        assert capability in GENERAL_SYSTEM_PROMPT


def test_general_prompts_contain_required_placeholders():
    assert "{query}" in GENERAL_USER_PROMPT


def test_general_prompts_prevent_false_paper_claims():
    assert "Do NOT make up information about any specific paper." in (
        GENERAL_SYSTEM_PROMPT
    )
    assert "Do NOT pretend to have analyzed a paper" in (
        GENERAL_SYSTEM_PROMPT
    )
