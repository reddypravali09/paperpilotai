
import hashlib
import logging

from utils.helpers import (
    setup_logger,
    get_paper_hash,
    format_source_pages,
    truncate_text,
)


# 1. Logger tests

def test_setup_logger_returns_logger():
    logger = setup_logger("test_logger")

    assert isinstance(logger, logging.Logger)


def test_setup_logger_sets_log_level():
    logger = setup_logger("test_level_logger")

    assert logger.level == logging.INFO


def test_setup_logger_adds_handler():
    logger = setup_logger("test_handler_logger")

    assert len(logger.handlers) > 0


# 2. Paper hash tests

def test_get_paper_hash_returns_string():
    result = get_paper_hash(b"sample pdf content")

    assert isinstance(result, str)


def test_get_paper_hash_returns_16_characters():
    result = get_paper_hash(b"sample pdf content")

    assert len(result) == 16


def test_get_paper_hash_uses_sha256():
    pdf_bytes = b"sample pdf content"

    expected = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    assert get_paper_hash(pdf_bytes) == expected


def test_get_paper_hash_same_content_same_hash():
    pdf1 = b"same pdf content"
    pdf2 = b"same pdf content"

    assert get_paper_hash(pdf1) == get_paper_hash(pdf2)


def test_get_paper_hash_different_content_different_hash():
    pdf1 = b"first pdf"
    pdf2 = b"second pdf"

    assert get_paper_hash(pdf1) != get_paper_hash(pdf2)


# 3. Source page formatting tests

def test_format_source_pages():
    result = format_source_pages([1, 3, 5])

    assert result == "1, 3, 5"


def test_format_source_pages_sorts_pages():
    result = format_source_pages([5, 1, 3])

    assert result == "1, 3, 5"


def test_format_source_pages_removes_duplicates():
    result = format_source_pages([1, 3, 3, 5, 1])

    assert result == "1, 3, 5"


def test_format_source_pages_empty_list():
    result = format_source_pages([])

    assert result == "N/A"


# 4. Text truncation tests

def test_truncate_text_short_text():
    text = "Hello"

    assert truncate_text(text) == "Hello"


def test_truncate_text_long_text():
    text = "A" * 250

    result = truncate_text(text)

    assert len(result) == 203
    assert result.endswith("...")


def test_truncate_text_custom_length():
    text = "A" * 100

    result = truncate_text(text, max_length=20)

    assert result == ("A" * 20) + "..."


def test_truncate_text_exact_length():
    text = "A" * 200

    assert truncate_text(text) == text


def test_truncate_text_empty_string():
    assert truncate_text("") == ""
