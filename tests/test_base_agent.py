
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from agents import base_agent
from agents.base_agent import BaseAgent
from config.settings import MIN_RETRIEVAL_CHUNKS


@pytest.fixture
def agent(monkeypatch):
    """Create a BaseAgent without calling the real Groq API."""
    fake_llm = RunnableLambda(lambda prompt: "Mock agent response")

    monkeypatch.setattr(
        base_agent,
        "ChatGroq",
        MagicMock(return_value=fake_llm),
    )

    return BaseAgent(
        system_prompt="You are a test assistant.",
        user_prompt="Context: {context}\nQuery: {query}",
        top_k=5,
        agent_name="TestAgent",
    )


def test_base_agent_initialization(agent):
    assert agent.agent_name == "TestAgent"
    assert agent.top_k == 5
    assert agent.system_prompt == "You are a test assistant."
    assert agent.user_prompt == "Context: {context}\nQuery: {query}"
    assert agent.chain is not None


def test_build_chain_creates_lcel_chain(agent):
    assert agent.chain is not None

    result = agent.chain.invoke({
        "query": "Hello",
        "paper_hash": None,
        "chat_history": [],
        "needs_retrieval": False,
    })

    assert result == "Mock agent response"


def test_retrieval_is_skipped_when_disabled(agent, monkeypatch):
    search_mock = MagicMock()

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        search_mock,
    )

    result = agent._retrieve_and_format({
        "query": "Hello",
        "paper_hash": "paper123",
        "needs_retrieval": False,
    })

    search_mock.assert_not_called()

    assert result["context"] == "No context available."
    assert agent._retrieval_metadata["needs_retrieval"] is False
    assert agent._retrieval_metadata["chunk_count"] == 0


def test_retrieval_is_skipped_without_paper_hash(agent, monkeypatch):
    search_mock = MagicMock()

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        search_mock,
    )

    agent._retrieve_and_format({
        "query": "Explain the methodology",
        "paper_hash": None,
        "needs_retrieval": True,
    })

    search_mock.assert_not_called()
    assert agent._retrieval_metadata["chunk_count"] == 0


def test_successful_retrieval_formats_context_and_sources(
    agent,
    monkeypatch,
):
    docs = [
        Document(
            page_content="First research finding.",
            metadata={"page": 1},
        ),
        Document(
            page_content="Second research finding.",
            metadata={"page": 3},
        ),
    ]

    search_mock = MagicMock(return_value=docs)

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        search_mock,
    )

    result = agent._retrieve_and_format({
        "query": "What are the findings?",
        "paper_hash": "paper123",
        "needs_retrieval": True,
    })

    search_mock.assert_called_once_with(
        "paper123",
        "What are the findings?",
        k=5,
    )

    assert result["context"] == (
        "[Page 1]: First research finding.\n\n"
        "[Page 3]: Second research finding."
    )
    assert result["source_pages"] == "1, 3"
    assert result["chunk_count"] == "2"

    assert agent._retrieval_metadata["sources"] == [1, 3]
    assert agent._retrieval_metadata["chunk_count"] == 2
    assert agent._retrieval_metadata["insufficient_retrieval"] is False


def test_insufficient_retrieval_with_one_document(
    agent,
    monkeypatch,
):
    docs = [
        Document(
            page_content="Only one matching result.",
            metadata={"page": 2},
        )
    ]

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        MagicMock(return_value=docs),
    )

    result = agent._retrieve_and_format({
        "query": "Explain the result",
        "paper_hash": "paper123",
        "needs_retrieval": True,
    })

    assert "LIMITED CONTEXT" in result["context"]
    assert "Only one matching result." in result["context"]
    assert agent._retrieval_metadata["insufficient_retrieval"] is True
    assert agent._retrieval_metadata["sources"] == []


def test_insufficient_retrieval_with_no_documents(
    agent,
    monkeypatch,
):
    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        MagicMock(return_value=[]),
    )

    result = agent._retrieve_and_format({
        "query": "Unknown topic",
        "paper_hash": "paper123",
        "needs_retrieval": True,
    })

    assert result["context"] == (
        "No relevant content found in the paper for this query."
    )
    assert agent._retrieval_metadata["chunk_count"] == 0
    assert agent._retrieval_metadata["insufficient_retrieval"] is True


def test_retrieval_error_is_handled(agent, monkeypatch):
    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        MagicMock(side_effect=Exception("Database error")),
    )

    result = agent._retrieve_and_format({
        "query": "Find information",
        "paper_hash": "paper123",
        "needs_retrieval": True,
    })

    assert result["context"] == (
        "Error retrieving content from the document."
    )
    assert agent._retrieval_metadata["chunk_count"] == 0


def test_extra_arguments_are_forwarded(agent):
    result = agent._retrieve_and_format({
        "query": "Generate slides",
        "paper_hash": None,
        "needs_retrieval": False,
        "slide_count": 10,
    })

    assert result["slide_count"] == 10
    assert result["query"] == "Generate slides"


def test_chat_history_is_forwarded(agent):
    history = ["Previous question", "Previous answer"]

    result = agent._retrieve_and_format({
        "query": "Continue",
        "needs_retrieval": False,
        "chat_history": history,
    })

    assert result["chat_history"] == history


def test_run_returns_response_and_metadata(agent, monkeypatch):
    docs = [
        Document(
            page_content="Research content one.",
            metadata={"page": 1},
        ),
        Document(
            page_content="Research content two.",
            metadata={"page": 2},
        ),
    ]

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        MagicMock(return_value=docs),
    )

    agent.verify_response = MagicMock(
        return_value={
            "verified": True,
            "status": "Verified",
        }
    )

    result = agent.run(
        query="Summarize the paper",
        paper_hash="paper123",
    )

    assert result["response"] == "Mock agent response"
    assert result["sources"] == [1, 2]
    assert result["chunk_count"] == 2
    assert result["needs_retrieval"] is True
    assert result["insufficient_retrieval"] is False
    assert result["verification"]["verified"] is True

    agent.verify_response.assert_called_once()


def test_run_skips_verification_when_retrieval_disabled(agent):
    agent.verify_response = MagicMock()

    result = agent.run(
        query="Hello",
        needs_retrieval=False,
    )

    agent.verify_response.assert_not_called()

    assert result["response"] == "Mock agent response"
    assert result["needs_retrieval"] is False
    assert result["verification"]["verified"] is True


def test_run_skips_verification_for_insufficient_chunks(
    agent,
    monkeypatch,
):
    docs = [
        Document(
            page_content="Only one result.",
            metadata={"page": 1},
        )
    ]

    monkeypatch.setattr(
        base_agent,
        "similarity_search",
        MagicMock(return_value=docs),
    )

    agent.verify_response = MagicMock()

    result = agent.run(
        query="Explain this",
        paper_hash="paper123",
    )

    agent.verify_response.assert_not_called()

    assert result["insufficient_retrieval"] is True
    assert result["verification"]["verified"] is True


def test_run_handles_chain_error(agent):
    agent.chain = MagicMock()
    agent.chain.invoke.side_effect = Exception("LLM failure")

    result = agent.run(
        query="Summarize",
        paper_hash="paper123",
    )

    assert "LLM failure" in result["response"]
    assert result["sources"] == []
    assert result["chunk_count"] == 0
    assert result["verification"]["verified"] is False
    assert result["verification"]["status"] == "❌ Error occurred"


def test_verify_response_handles_reflection_error(agent, monkeypatch):
    monkeypatch.setattr(
        base_agent.ChatPromptTemplate,
        "from_messages",
        MagicMock(side_effect=Exception("Reflection error")),
    )

    result = agent.verify_response(
        context="Research context",
        response="Generated answer",
    )

    assert result["verified"] is True
    assert result["status"] == "✅ Verification skipped"
