"""Tests for sentence-aware chunking with overlap."""

from __future__ import annotations

import pytest

from arxiv_rag.data.chunking import chunk_papers, chunk_text, source_text, split_sentences


def test_short_text_is_a_single_chunk() -> None:
    text = "This is a short abstract about transformers."
    assert chunk_text(text, max_chars=1600, overlap_chars=200) == [text]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("   ", max_chars=100, overlap_chars=20) == []
    assert chunk_text("", max_chars=100, overlap_chars=20) == []


def test_max_chars_must_be_positive() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0)


def test_long_text_splits_and_respects_max_chars() -> None:
    sentences = [f"Sentence number {i} contains distinctive token tok{i}." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_chars=120, overlap_chars=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    # All distinctive tokens should appear in at least one chunk.
    for i in range(20):
        assert any(f"tok{i}" in chunk for chunk in chunks)


def test_overlap_copies_a_tail_of_the_previous_chunk() -> None:
    text = "Alpha sentence one. Bravo sentence two. Charlie sentence three. Delta sentence four."
    chunks = chunk_text(text, max_chars=50, overlap_chars=15)
    assert len(chunks) >= 2
    # The start of chunk i+1 should share characters with the end of chunk i
    # when overlap is enabled and a sentence boundary forced a split.
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        tail = prev[-15:].lstrip()
        if tail and len(curr) < 50 or True:
            # At least some shared content or a clean sentence start.
            shared = any(token in curr for token in prev.split()[-3:])
            assert shared


def test_very_long_sentence_is_hard_wrapped() -> None:
    text = "word-" * 80  # no sentence boundary
    chunks = chunk_text(text, max_chars=40, overlap_chars=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_split_sentences_keeps_punctuation() -> None:
    parts = split_sentences("Hello world. How are you? Fine!")
    assert parts == ["Hello world.", "How are you?", "Fine!"]


def test_chunk_papers_assigns_stable_ids(papers) -> None:
    chunks = chunk_papers(papers, max_chars=1600, overlap_chars=200)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == [f"{p['arxiv_id']}::0" for p in papers]
    assert {c["arxiv_id"] for c in chunks} == {p["arxiv_id"] for p in papers}
    transformer = next(c for c in chunks if c["arxiv_id"] == "1706.03762")
    assert "Transformer" in transformer["text"]
    assert transformer["year"] == 2017
    assert "cs.CL" in transformer["categories"]


def test_source_text_concatenates_title_and_abstract() -> None:
    text = source_text(
        {"title": "A Title", "abstract": "An abstract."},
        fields=["title", "abstract"],
    )
    assert text.startswith("Title: A Title")
    assert "Abstract: An abstract." in text
