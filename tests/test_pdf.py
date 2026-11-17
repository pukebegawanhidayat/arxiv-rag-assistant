"""Tests for PDF text extraction and chunking."""

from __future__ import annotations

import pytest

from arxiv_rag.data.pdf import PdfExtractionError, chunk_pdf, extract_pdf_text
from tests.pdf_samples import MINIMAL_PDF, make_text_pdf


def test_extract_text_from_minimal_pdf() -> None:
    text = extract_pdf_text(MINIMAL_PDF)
    assert "transformer" in text.lower()
    assert "self-attention" in text.lower() or "self-attention" in text.lower().replace(" ", "")


def test_extract_text_from_path(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(MINIMAL_PDF)
    text = extract_pdf_text(path)
    assert "Attention" in text


def test_invalid_pdf_raises() -> None:
    with pytest.raises(PdfExtractionError):
        extract_pdf_text(b"this is not a pdf")


def test_empty_pdf_bytes_raise() -> None:
    with pytest.raises(PdfExtractionError):
        extract_pdf_text(b"")


def test_chunk_pdf_assigns_session_ids() -> None:
    chunks = chunk_pdf(
        MINIMAL_PDF,
        filename="attention.pdf",
        session_id="abc123",
        max_chars=1600,
        overlap_chars=200,
    )
    assert len(chunks) >= 1
    assert chunks[0]["chunk_id"] == "pdf:abc123:0"
    assert chunks[0]["arxiv_id"] == "pdf:attention.pdf"
    assert chunks[0]["source_type"] == "pdf"
    assert "transformer" in chunks[0]["text"].lower()
    assert chunks[0]["n_chunks"] == len(chunks)


def test_chunk_pdf_respects_overlap_on_long_text() -> None:
    long_body = " ".join(
        f"Paragraph {i} discusses retrieval augmented generation and citations."
        for i in range(40)
    )
    pdf = make_text_pdf(long_body)
    chunks = chunk_pdf(pdf, filename="long.pdf", session_id="sess", max_chars=180, overlap_chars=40)
    assert len(chunks) >= 2
    assert all(len(c["text"]) <= 180 for c in chunks)
    assert all(c["session_id"] == "sess" for c in chunks)
