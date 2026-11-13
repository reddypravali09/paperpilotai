"""
Text Chunker
=============
Splits extracted PDF pages into smaller chunks using
RecursiveCharacterTextSplitter. Preserves page metadata
on every chunk for source attribution.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    paper_hash: str = ""
) -> list[Document]:
    """Split documents into smaller chunks preserving page metadata.

    Args:
        documents: List of page-level Documents from pdf_processor.
        chunk_size: Maximum characters per chunk (default: 500).
        chunk_overlap: Character overlap between chunks (default: 50).
        paper_hash: SHA-256 hash prefix for the paper (added to metadata).

    Returns:
        List of chunked Documents, each retaining:
        - Original page number
        - paper_hash identifier
        - chunk_index within the document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    chunk_index = 0

    for doc in documents:
        page_chunks = splitter.split_documents([doc])
        for chunk in page_chunks:
            chunk.metadata["paper_hash"] = paper_hash
            chunk.metadata["chunk_index"] = chunk_index
            # page metadata is already preserved from the original doc
            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(
        f"Split {len(documents)} pages into {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
