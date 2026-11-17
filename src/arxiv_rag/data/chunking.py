"""Configurable chunking of title + abstract."""

from __future__ import annotations

import re
from typing import Any

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def source_text(paper: dict[str, Any], fields: list[str] | None = None) -> str:
    fields = fields or ["title", "abstract"]
    parts: list[str] = []
    for field in fields:
        value = str(paper.get(field) or "").strip()
        if not value:
            continue
        label = field.capitalize()
        parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = SENTENCE_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(text: str, max_chars: int = 1600, overlap_chars: int = 200) -> list[str]:
    """Pack sentences into chunks of at most ``max_chars`` with character overlap.

    If a single sentence is longer than ``max_chars``, it is hard-wrapped.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    overlap_chars = max(0, min(overlap_chars, max_chars - 1))
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = split_sentences(text)
    # Fall back to hard wrapping when sentence splitting fails to shrink.
    units: list[str] = []
    for sent in sentences or [text]:
        if len(sent) <= max_chars:
            units.append(sent)
        else:
            step = max_chars - overlap_chars or max_chars
            for i in range(0, len(sent), step):
                piece = sent[i : i + max_chars].strip()
                if piece:
                    units.append(piece)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current} {unit}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            if overlap_chars > 0:
                tail = current[-overlap_chars:].lstrip()
                current = f"{tail} {unit}".strip() if tail else unit
                # If overlap + new unit overflow, just start a new chunk.
                if len(current) > max_chars:
                    current = unit
            else:
                current = unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_papers(
    papers: list[dict[str, Any]],
    max_chars: int = 1600,
    overlap_chars: int = 200,
    source_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Turn papers into chunk dicts with stable ids and copied metadata."""
    chunks: list[dict[str, Any]] = []
    for paper in papers:
        text = source_text(paper, source_fields)
        pieces = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{paper['arxiv_id']}::{i}",
                    "arxiv_id": paper["arxiv_id"],
                    "chunk_index": i,
                    "n_chunks": len(pieces),
                    "text": piece,
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", ""),
                    "categories": list(paper.get("categories") or []),
                    "year": paper.get("year"),
                    "doi": paper.get("doi", ""),
                }
            )
    return chunks
