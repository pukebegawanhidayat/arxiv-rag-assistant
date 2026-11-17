"""Parse user-uploaded PDFs and turn them into overlapping text chunks."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from arxiv_rag.data.chunking import chunk_text

logger = logging.getLogger(__name__)


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot be opened or contains no extractable text."""


def extract_pdf_text(source: str | Path | bytes | io.BufferedIOBase) -> str:
    """Extract plain text from a PDF path or byte string.

    Pages are joined with blank lines. Whitespace is normalised lightly so
    chunking stays stable, but line breaks inside a page are preserved as spaces.
    """
    try:
        reader = _open_reader(source)
    except Exception as exc:  # noqa: BLE001 - pypdf raises a variety of errors
        raise PdfExtractionError(f"Unable to read PDF: {exc}") from exc

    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from PDF page %s: %s", i, exc)
            text = ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append(cleaned)
    if not pages:
        raise PdfExtractionError("The PDF contains no extractable text.")
    return "\n\n".join(pages)


def _open_reader(source: str | Path | bytes | io.BufferedIOBase) -> PdfReader:
    if isinstance(source, (bytes, bytearray)):
        return PdfReader(io.BytesIO(source))
    if isinstance(source, io.BufferedIOBase):
        return PdfReader(source)
    return PdfReader(str(source))


def chunk_pdf(
    source: str | Path | bytes | io.BufferedIOBase,
    *,
    filename: str,
    session_id: str,
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> list[dict[str, Any]]:
    """Parse a PDF and return chunk dicts compatible with the retrievers."""
    text = extract_pdf_text(source)
    pieces = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    if not pieces:
        raise PdfExtractionError("The PDF produced no text chunks.")
    chunks: list[dict[str, Any]] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            {
                "chunk_id": f"pdf:{session_id}:{i}",
                "arxiv_id": f"pdf:{filename}",
                "chunk_index": i,
                "n_chunks": len(pieces),
                "text": piece,
                "title": filename,
                "authors": "",
                "categories": ["pdf"],
                "year": None,
                "doi": "",
                "source_type": "pdf",
                "filename": filename,
                "session_id": session_id,
            }
        )
    return chunks
