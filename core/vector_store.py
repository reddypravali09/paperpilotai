"""
Vector Store Manager
=====================
Manages ChromaDB persistent collections for per-paper document storage.
Each paper gets its own collection (paper_{sha256_hash}) to prevent
cross-paper contamination.

IMPORTANT: Collections are deleted via the ChromaDB API — never by
deleting filesystem directories. This avoids SQLite lock errors on Windows.
"""

import os
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_PERSIST_DIR, COLLECTION_PREFIX, RETRIEVAL_TOP_K
from core.embeddings import get_embeddings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def _get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the persistent ChromaDB client.

    Creates the persist directory if it doesn't exist.

    Returns:
        chromadb.PersistentClient instance.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection_name(paper_hash: str) -> str:
    """Build the ChromaDB collection name for a paper."""
    return f"{COLLECTION_PREFIX}{paper_hash}"


def collection_exists(paper_hash: str) -> bool:
    """Check if a paper's collection already exists in ChromaDB.

    Args:
        paper_hash: SHA-256 hash prefix of the paper.

    Returns:
        True if the collection exists, False otherwise.
    """
    client = _get_chroma_client()
    collection_name = get_collection_name(paper_hash)
    try:
        existing = [c.name for c in client.list_collections()]
        return collection_name in existing
    except Exception as e:
        logger.error(f"Error checking collection existence: {e}")
        return False


def get_or_create_collection(paper_hash: str) -> Chroma:
    """Get an existing collection or create a new one for this paper.

    Args:
        paper_hash: SHA-256 hash prefix identifying the paper.

    Returns:
        LangChain Chroma wrapper for the collection.
    """
    client = _get_chroma_client()
    collection_name = get_collection_name(paper_hash)
    embeddings = get_embeddings()

    logger.info(f"Accessing collection: {collection_name}")
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def add_documents(paper_hash: str, documents: list[Document]) -> int:
    """Add document chunks to a paper's ChromaDB collection.

    Args:
        paper_hash: SHA-256 hash prefix of the paper.
        documents: List of chunked Documents with metadata.

    Returns:
        Number of documents added.
    """
    collection = get_or_create_collection(paper_hash)

    # Generate unique IDs for each document
    ids = [f"{paper_hash}_chunk_{i}" for i in range(len(documents))]

    collection.add_documents(documents=documents, ids=ids)
    logger.info(f"Added {len(documents)} chunks to collection paper_{paper_hash}")
    return len(documents)


def similarity_search(
    paper_hash: str,
    query: str,
    k: int = RETRIEVAL_TOP_K
) -> list[Document]:
    """Search for similar document chunks in a paper's collection.

    Args:
        paper_hash: SHA-256 hash prefix of the paper.
        query: Search query string.
        k: Number of results to return (default: RETRIEVAL_TOP_K).

    Returns:
        List of most similar Documents with metadata.
    """
    collection = get_or_create_collection(paper_hash)
    results = collection.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
    return results


def delete_collection_safe(paper_hash: str) -> bool:
    """Safely delete a paper's collection using the ChromaDB API.

    IMPORTANT: Uses client.delete_collection() — never deletes
    filesystem directories directly. This prevents SQLite lock
    errors (WinError 32) on Windows.

    Args:
        paper_hash: SHA-256 hash prefix of the paper.

    Returns:
        True if collection was deleted, False if it didn't exist.
    """
    client = _get_chroma_client()
    collection_name = get_collection_name(paper_hash)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return True
    except ValueError:
        logger.debug(f"Collection {collection_name} doesn't exist, nothing to delete")
        return False
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        return False
