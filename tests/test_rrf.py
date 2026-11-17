"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from arxiv_rag.retrievers.base import RetrievedChunk, reciprocal_rank_fusion


def _hit(chunk_id: str, score: float = 1.0, source: str = "x") -> RetrievedChunk:
    arxiv_id = chunk_id.split("::", 1)[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        score=score,
        metadata={"arxiv_id": arxiv_id, "title": chunk_id},
        source=source,
    )


def test_document_present_in_both_lists_ranks_higher() -> None:
    bm25 = [_hit("A::0"), _hit("B::0"), _hit("C::0")]
    dense = [_hit("C::0"), _hit("A::0"), _hit("D::0")]
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    ids = [c.chunk_id for c in fused]
    # A is rank 1 and 2; C is rank 3 and 1. C's RRF score is 1/61 + 1/63,
    # A's is 1/61 + 1/62. A should win by a hair, then C, then B, then D.
    assert ids[0] == "A::0"
    assert ids[1] == "C::0"
    assert set(ids) == {"A::0", "B::0", "C::0", "D::0"}
    assert all(c.source == "hybrid" for c in fused)


def test_rrf_score_formula() -> None:
    left = [_hit("only-left::0")]
    right = [_hit("only-right::0")]
    fused = reciprocal_rank_fusion([left, right], k=10)
    # Each unique document sits at rank 1 of one list: 1 / (10 + 1).
    assert fused[0].score == pytest_approx(1.0 / 11.0)
    assert fused[1].score == pytest_approx(1.0 / 11.0)


def pytest_approx(value: float):  # local import-style helper
    import pytest

    return pytest.approx(value)


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_single_list_preserves_order() -> None:
    ranking = [_hit("a::0"), _hit("b::0"), _hit("c::0")]
    fused = reciprocal_rank_fusion([ranking], k=60)
    assert [c.chunk_id for c in fused] == ["a::0", "b::0", "c::0"]
