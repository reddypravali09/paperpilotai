
import pytest

from core import embeddings
from core.embeddings import get_embeddings


@pytest.fixture(autouse=True)
def reset_embeddings_singleton():
    """Reset the singleton before each test."""
    embeddings._embeddings_instance = None

    yield

    embeddings._embeddings_instance = None


def test_get_embeddings_initializes_model():
    """Check that the embedding model initializes."""

    model = get_embeddings()

    assert model is not None


def test_get_embeddings_singleton():
    """Check that repeated calls return the same instance."""

    model1 = get_embeddings()
    model2 = get_embeddings()

    assert model1 is model2


def test_generate_embeddings():
    """Check that text can be converted into embeddings."""

    model = get_embeddings()

    result = model.embed_query("PaperPilot AI academic assistant")

    assert isinstance(result, list)
    assert len(result) == 384


def test_embedding_values_are_normalized():
    """Check that the generated embedding is normalized."""

    model = get_embeddings()

    result = model.embed_query("Research paper summarization")

    magnitude = sum(value ** 2 for value in result) ** 0.5

    assert magnitude == pytest.approx(1.0, abs=0.01)

