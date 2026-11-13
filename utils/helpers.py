"""
PaperPilot AI — Utility Functions
===================================
Shared helper functions for logging, hashing, and formatting.
"""

import hashlib
import logging
from config.settings import LOG_LEVEL


def setup_logger(name: str) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger with stream handler and standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def get_paper_hash(pdf_bytes: bytes) -> str:
    """Generate a unique SHA-256 hash for a PDF file.

    Used as a collection identifier in ChromaDB to ensure
    per-paper isolation and prevent cross-paper contamination.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        First 16 characters of the SHA-256 hex digest.
    """
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def format_source_pages(pages: list[int]) -> str:
    """Format a list of page numbers for display.

    Args:
        pages: Sorted list of page numbers.

    Returns:
        Formatted string like '1, 3, 5' or 'N/A'.
    """
    if not pages:
        return "N/A"
    return ", ".join(str(p) for p in sorted(set(pages)))


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis for logging."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
