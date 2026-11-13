
from langchain_core.documents import Document

from core.text_chunker import chunk_documents


def test_chunk_documents():
    """Check that a long document is split into smaller chunks."""

    text = "PaperPilot AI is an academic assistant. " * 50

    documents = [
        Document(
            page_content=text,
            metadata={"page": 1, "source": "uploaded_pdf"}
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 100


def test_page_metadata_preserved():
    """Check that original page numbers are preserved."""

    documents = [
        Document(
            page_content="This is page one. " * 30,
            metadata={"page": 1}
        ),
        Document(
            page_content="This is page two. " * 30,
            metadata={"page": 2}
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20
    )

    pages = {chunk.metadata["page"] for chunk in chunks}

    assert pages == {1, 2}


def test_paper_hash_added():
    """Check that the paper hash is added to every chunk."""

    documents = [
        Document(
            page_content="Testing paper hash metadata. " * 20,
            metadata={"page": 1}
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
        paper_hash="abc123"
    )

    for chunk in chunks:
        assert chunk.metadata["paper_hash"] == "abc123"


def test_chunk_index_added():
    """Check that every chunk receives an index."""

    documents = [
        Document(
            page_content="Testing chunk indexes. " * 30,
            metadata={"page": 1}
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20
    )

    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]

    assert indexes == list(range(len(chunks)))


def test_empty_documents():
    """Check that an empty document list returns an empty list."""

    chunks = chunk_documents([])

    assert chunks == []


def test_chunk_overlap():
    """Check that consecutive chunks share overlapping text."""

    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 20

    documents = [
        Document(
            page_content=text,
            metadata={"page": 1}
        )
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=50,
        chunk_overlap=10
    )

    assert len(chunks) > 1

    # Check that consecutive chunks have some shared text.
    for i in range(len(chunks) - 1):
        assert (
            chunks[i].page_content[-10:]
            in chunks[i + 1].page_content
        )
