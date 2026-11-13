
import pytest
from langchain_core.documents import Document

from core import vector_store


@pytest.fixture
def temporary_chroma(monkeypatch, tmp_path):
    """Use temporary storage and a lightweight fake embedding model."""

    monkeypatch.setattr(
        vector_store,
        "CHROMA_PERSIST_DIR",
        str(tmp_path / "chroma_test")
    )

    monkeypatch.setattr(
        vector_store,
        "get_embeddings",
        lambda: FakeEmbeddings()
    )

    yield


class FakeEmbeddings:
    """Simple deterministic embeddings for testing ChromaDB."""

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        # Three-dimensional vector based on character counts.
        return [
            float(sum(c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in text)),
            float(sum(c.lower() in "aeiou" for c in text)),
            float(len(text))
        ]


def test_get_collection_name():
    name = vector_store.get_collection_name("abc123")

    assert name == "paper_abc123"


def test_collection_creation(temporary_chroma):
    collection = vector_store.get_or_create_collection("test123")

    assert collection is not None
    assert vector_store.collection_exists("test123")


def test_add_documents(temporary_chroma):
    documents = [
        Document(
            page_content="Machine learning is a branch of artificial intelligence.",
            metadata={"page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"page": 2}
        )
    ]

    count = vector_store.add_documents("paper123", documents)

    assert count == 2


def test_similarity_search(temporary_chroma):
    documents = [
        Document(
            page_content="Machine learning uses algorithms to learn from data.",
            metadata={"page": 1}
        ),
        Document(
            page_content="Cooking involves preparing food.",
            metadata={"page": 2}
        )
    ]

    vector_store.add_documents("search123", documents)

    results = vector_store.similarity_search(
        "search123",
        "machine learning",
        k=1
    )

    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_delete_collection(temporary_chroma):
    vector_store.get_or_create_collection("delete123")

    assert vector_store.collection_exists("delete123")

    deleted = vector_store.delete_collection_safe("delete123")

    assert deleted is True
    assert vector_store.collection_exists("delete123") is False


def test_delete_nonexistent_collection(temporary_chroma):
    deleted = vector_store.delete_collection_safe("missing123")

    assert deleted is False


def test_paper_collections_are_separate(temporary_chroma):
    docs_a = [
        Document(
            page_content="Artificial intelligence research paper.",
            metadata={"page": 1}
        )
    ]

    docs_b = [
        Document(
            page_content="Database management research paper.",
            metadata={"page": 1}
        )
    ]

    vector_store.add_documents("paperA", docs_a)
    vector_store.add_documents("paperB", docs_b)

    results_a = vector_store.similarity_search(
        "paperA",
        "artificial intelligence",
        k=1
    )

    results_b = vector_store.similarity_search(
        "paperB",
        "database management",
        k=1
    )

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert "Artificial intelligence" in results_a[0].page_content
    assert "Database management" in results_b[0].page_content
