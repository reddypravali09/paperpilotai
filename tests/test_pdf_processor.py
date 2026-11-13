
import io
import pytest

from reportlab.pdfgen import canvas
from pypdf.errors import PdfReadError

from core.pdf_processor import extract_text_from_pdf


def create_test_pdf(texts):
    """Generate a proper PDF containing the given text."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(300, 300))

    for text in texts:
        if text:
            pdf.drawString(50, 200, text)
        pdf.showPage()

    pdf.save()
    buffer.seek(0)

    return buffer


def test_extract_text_from_pdf():
    pdf = create_test_pdf(["Hello PaperPilot"])

    documents = extract_text_from_pdf(pdf)

    assert len(documents) == 1
    assert "Hello PaperPilot" in documents[0].page_content


def test_page_number_metadata():
    pdf = create_test_pdf(["First page", "Second page"])

    documents = extract_text_from_pdf(pdf)

    assert len(documents) == 2
    assert documents[0].metadata["page"] == 1
    assert documents[1].metadata["page"] == 2


def test_empty_pdf():
    pdf = create_test_pdf([""])

    with pytest.raises(ValueError):
        extract_text_from_pdf(pdf)


def test_invalid_pdf():
    invalid_pdf = io.BytesIO(b"This is not a PDF")

    with pytest.raises(PdfReadError):
        extract_text_from_pdf(invalid_pdf)


def test_bytesio_input():
    pdf = create_test_pdf(["BytesIO test"])

    documents = extract_text_from_pdf(pdf)

    assert isinstance(documents, list)
    assert len(documents) == 1
    assert "BytesIO test" in documents[0].page_content
