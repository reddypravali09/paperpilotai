
import pytest
from unittest.mock import patch

from agents import intent_router
from agents.intent_router import (
    classify_intent,
    create_router_branch,
    route_query,
)
from schemas.intent import IntentEnum, IntentClassification


class FakeAgent:
    """A fake agent used to test routing without calling an LLM."""

    def __init__(self, name):
        self.name = name
        self.received_input = None

    def run(self, **kwargs):
        self.received_input = kwargs

        return {
            "response": f"{self.name} agent response",
            "sources": [],
            "metadata": {}
        }


@pytest.fixture
def fake_agents():
    return {
        "summarize": FakeAgent("Summarizer"),
        "concepts": FakeAgent("Concept"),
        "slides": FakeAgent("Slide"),
        "viva": FakeAgent("Viva"),
        "general": FakeAgent("General"),
    }


def test_classify_intent_success():
    """Check that a successful classification is returned."""

    expected = IntentClassification(
        intent=IntentEnum.SUMMARIZE,
        reasoning="User wants a paper summary",
        needs_retrieval=True
    )

    with patch(
        "agents.intent_router.get_classifier_llm"
    ) as mock_llm:

        mock_chain = mock_llm.return_value.with_structured_output.return_value
        mock_prompt = patch(
            "agents.intent_router.ChatPromptTemplate.from_messages"
        )

        with mock_prompt as prompt_mock:
            prompt_mock.return_value.__or__.return_value.invoke.return_value = expected

            result = classify_intent("Summarize this paper")

    assert result.intent == IntentEnum.SUMMARIZE
    assert result.needs_retrieval is True


def test_classify_intent_fallback():
    """Check that classification errors fall back to GENERAL."""

    with patch(
        "agents.intent_router.get_classifier_llm",
        side_effect=Exception("API unavailable")
    ):
        result = classify_intent("Hello")

    assert result.intent == IntentEnum.GENERAL
    assert result.needs_retrieval is False


@pytest.mark.parametrize(
    "intent,agent_key",
    [
        (IntentEnum.SUMMARIZE, "summarize"),
        (IntentEnum.CONCEPTS, "concepts"),
        (IntentEnum.SLIDES, "slides"),
        (IntentEnum.VIVA, "viva"),
    ]
)
def test_router_branch_routes_correctly(fake_agents, intent, agent_key):
    """Check that each intent reaches its corresponding agent."""

    branch = create_router_branch(fake_agents)

    result = branch.invoke({
        "intent": intent,
        "query": "Test query",
        "paper_hash": "paper123",
        "chat_history": [],
        "needs_retrieval": True
    })

    assert result["response"].startswith(
        fake_agents[agent_key].name
    )


def test_router_branch_general_fallback(fake_agents):
    """Check that unmatched intents go to the general agent."""

    branch = create_router_branch(fake_agents)

    result = branch.invoke({
        "intent": IntentEnum.GENERAL,
        "query": "Hello",
        "paper_hash": None,
        "chat_history": [],
        "needs_retrieval": False
    })

    assert result["response"] == "General agent response"


def test_route_query_passes_classification(fake_agents):
    """Check the complete routing pipeline and classification metadata."""

    classification = IntentClassification(
        intent=IntentEnum.SUMMARIZE,
        reasoning="Summary requested",
        needs_retrieval=True
    )

    with patch(
        "agents.intent_router.classify_intent",
        return_value=classification
    ):
        result = route_query(
            query="Summarize this paper",
            agents=fake_agents,
            paper_hash="paper123"
        )

    assert result["response"] == "Summarizer agent response"

    assert result["classification"]["intent"] == "summarize"
    assert result["classification"]["reasoning"] == "Summary requested"
    assert result["classification"]["needs_retrieval"] is True


def test_route_query_passes_additional_arguments(fake_agents):
    """Check that extra arguments reach the selected agent."""

    classification = IntentClassification(
        intent=IntentEnum.SLIDES,
        reasoning="User wants slides",
        needs_retrieval=True
    )

    with patch(
        "agents.intent_router.classify_intent",
        return_value=classification
    ):
        route_query(
            query="Create slides",
            agents=fake_agents,
            paper_hash="paper123",
            slide_count=10
        )

    received = fake_agents["slides"].received_input

    assert received["slide_count"] == 10
    assert received["paper_hash"] == "paper123"
