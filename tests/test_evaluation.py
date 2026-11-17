"""Tests for citation parsing and retrieval metrics."""

from __future__ import annotations

from arxiv_rag.citations import extract_citation_ids, linkify_citations
from arxiv_rag.evaluation import (
    build_eval_questions,
    first_relevant_rank,
    mean_reciprocal_rank,
    pick_query_span,
    recall_at_k,
)
from arxiv_rag.retrievers.base import RetrievedChunk


def test_extract_modern_and_legacy_ids() -> None:
    text = "See the Transformer [1706.03762] and an older note [cs/0001001]. Also arXiv:1810.04805."
    ids = extract_citation_ids(text)
    assert ids[0] == "1706.03762"
    assert "cs/0001001" in ids
    # Naked arXiv: mentions are used only when no bracketed citations exist; here brackets exist.
    assert "1810.04805" not in ids


def test_extract_naked_arxiv_when_no_brackets() -> None:
    ids = extract_citation_ids("Discussed in arXiv:2005.14165 and arXiv:2203.15556v1.")
    assert ids == ["2005.14165", "2203.15556v1"]


def test_linkify_citations() -> None:
    text = linkify_citations("Based on [1706.03762].")
    assert text == "Based on [1706.03762](https://arxiv.org/abs/1706.03762)."


def test_recall_and_mrr() -> None:
    ranks = [1, 3, None, 5]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.75
    assert mean_reciprocal_rank(ranks) == (1 / 1 + 1 / 3 + 0 + 1 / 5) / 4


def test_first_relevant_rank_matches_arxiv_id() -> None:
    chunks = [
        RetrievedChunk("a::0", "t", 1.0, {"arxiv_id": "1111.00001"}),
        RetrievedChunk("b::0", "t", 0.5, {"arxiv_id": "1706.03762"}),
    ]
    assert first_relevant_rank(chunks, "1706.03762") == 2
    assert first_relevant_rank(chunks, "missing") is None


def test_build_eval_questions_uses_known_ids(papers) -> None:
    questions = build_eval_questions(papers, n_questions=3, seed=0)
    assert len(questions) == 3
    gold = {q["gold_arxiv_id"] for q in questions}
    assert gold <= {p["arxiv_id"] for p in papers}
    for q in questions:
        assert len(q["query"]) >= 40


def test_pick_query_span_skips_generic_leads() -> None:
    abstract = (
        "We present a new model for graph learning. "
        "The encoder uses a spectral convolution over citation networks and "
        "outperforms prior message-passing baselines on Cora."
    )
    span = pick_query_span(abstract)
    assert span is not None
    assert not span.lower().startswith("we present")
