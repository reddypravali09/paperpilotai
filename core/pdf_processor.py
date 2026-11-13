"""
PDF Processor
==============
Extracts text from PDF files page-by-page.
Operates on BytesIO buffers — never writes temp files.
Preserves page numbers as metadata for source attribution.
"""

import io
from typing import BinaryIO
from langchain_core.documents import Document
from pypdf import PdfReader
from utils.helpers import setup_logger

logger = setup_logger(__name__)


def extract_text_from_pdf(pdf_source: bytes | BinaryIO) -> list[Document]:
    """Extract text from a PDF file, returning one Document per page.

    Args:
        pdf_source: Either raw PDF bytes or a file-like object (BytesIO).

    Returns:
        List of LangChain Document objects, each with:
        - page_content: extracted text from one page
        - metadata: {"page": 1-indexed page number, "source": "uploaded_pdf"}

    Raises:
        ValueError: If no text could be extracted from the PDF.
    """
    if isinstance(pdf_source, bytes):
        pdf_source = io.BytesIO(pdf_source)

    reader = PdfReader(pdf_source)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata={"page": page_num, "source": "uploaded_pdf"}
                )
            )
            logger.debug(f"Page {page_num}: extracted {len(text)} chars")
        else:
            logger.debug(f"Page {page_num}: empty or no text")

    if not documents:
        raise ValueError("No text could be extracted from the PDF. The file may be scanned or image-based.")

    logger.info(f"Extracted text from {len(documents)} pages (of {len(reader.pages)} total)")
    return documents
