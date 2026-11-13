"""
Intent Router
==============
Classifies user intent using Pydantic structured output,
then dispatches to the correct agent via RunnableBranch.

Uses the lightweight classifier model (llama-3.1-8b-instant)
for cost-efficient intent detection.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_groq import ChatGroq

from config.settings import (
    CLASSIFIER_MODEL,
    CLASSIFIER_TEMPERATURE,
    LLM_MAX_RETRIES,
    GROQ_API_KEY,
)
from schemas.intent import IntentEnum, IntentClassification
from prompts.router_prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Classifier helpers
# ------------------------------------------------------------------


def get_classifier_llm() -> ChatGroq:
    """Get the lightweight classifier LLM."""
    return ChatGroq(
        model=CLASSIFIER_MODEL,
        temperature=CLASSIFIER_TEMPERATURE,
        max_retries=LLM_MAX_RETRIES,
        api_key=GROQ_API_KEY,
    )


def classify_intent(query: str) -> IntentClassification:
    """Classify a user query into an intent category.

    Uses Pydantic structured output with the classifier model.
    Falls back to *GENERAL* intent on any error.

    Args:
        query: The user's raw query string.

    Returns:
        :class:`IntentClassification` with intent, reasoning, and
        needs_retrieval.
    """
    try:
        llm = get_classifier_llm()
        classifier = llm.with_structured_output(IntentClassification)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", ROUTER_USER_PROMPT),
            ]
        )

        chain = prompt | classifier
        result: IntentClassification = chain.invoke({"query": query})

        logger.info(
            f"Intent classified: {result.intent.value} "
            f"(needs_retrieval={result.needs_retrieval}) "
            f"Reason: {result.reasoning}"
        )
        return result

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return IntentClassification(
            intent=IntentEnum.GENERAL,
            reasoning=(
                f"Classification failed, defaulting to general: {e}"
            ),
            needs_retrieval=False,
        )


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------


def create_router_branch(agents: dict) -> RunnableBranch:
    """Create a RunnableBranch that dispatches to the correct agent.

    The branch receives a dict with an ``intent`` key and routes
    to the corresponding agent's ``run`` lambda.

    Args:
        agents: Dict mapping intent names to agent instances.
            Expected keys: ``summarize``, ``concepts``, ``slides``,
            ``viva``, ``general``.

    Returns:
        :class:`RunnableBranch` configured for all five intent
        categories.
    """
    return RunnableBranch(
        (
            lambda x: x.get("intent") == IntentEnum.SUMMARIZE,
            RunnableLambda(lambda x: agents["summarize"].run(**x)),
        ),
        (
            lambda x: x.get("intent") == IntentEnum.CONCEPTS,
            RunnableLambda(lambda x: agents["concepts"].run(**x)),
        ),
        (
            lambda x: x.get("intent") == IntentEnum.SLIDES,
            RunnableLambda(lambda x: agents["slides"].run(**x)),
        ),
        (
            lambda x: x.get("intent") == IntentEnum.VIVA,
            RunnableLambda(lambda x: agents["viva"].run(**x)),
        ),
        # Default: General agent (no retrieval)
        RunnableLambda(lambda x: agents["general"].run(**x)),
    )


def route_query(
    query: str,
    agents: dict,
    paper_hash: str | None = None,
    chat_history: list | None = None,
    **kwargs,
) -> dict:
    """Full routing pipeline: classify intent then dispatch to agent.

    This is the **main entry point** for the agent layer.

    Args:
        query: User's question.
        agents: Dict of agent instances.
        paper_hash: Current paper's hash for retrieval.
        chat_history: Previous conversation messages.
        **kwargs: Additional params (e.g., ``slide_count``).

    Returns:
        Dict with response, sources, metadata, and classification info.
    """
    # Step 1: Classify intent
    classification = classify_intent(query)

    # Step 2: Build routing input
    route_input: dict = {
        "intent": classification.intent,
        "query": query,
        "paper_hash": paper_hash,
        "chat_history": chat_history or [],
        "needs_retrieval": classification.needs_retrieval,
        **kwargs,
    }

    # Step 3: Create and invoke branch
    branch = create_router_branch(agents)
    result: dict = branch.invoke(route_input)

    # Step 4: Attach classification metadata
    result["classification"] = {
        "intent": classification.intent.value,
        "reasoning": classification.reasoning,
        "needs_retrieval": classification.needs_retrieval,
    }

    return result
