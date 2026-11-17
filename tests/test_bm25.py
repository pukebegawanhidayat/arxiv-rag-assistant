"""Tests for the BM25 retriever and its tokenizer."""

from __future__ import annotations

from arxiv_rag.data.chunking import chunk_papers
from arxiv_rag.retrievers.bm25 import BM25Retriever, tokenize


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Self-Attention, BERT!") == ["self", "attention", "bert"]
    assert tokenize("") == []


def test_bm25_ranks_the_matching_paper_first(papers) -> None:
    chunks = chunk_papers(papers)
    retriever = BM25Retriever.from_chunks(chunks)
    hits = retriever.search("masked language model bidirectional encoder", k=3)
    assert hits
    assert hits[0].arxiv_id == "1810.04805"
    assert hits[0].source == "bm25"
    assert hits[0].score > 0


def test_bm25_year_filter(papers) -> None:
    retriever = BM25Retriever.from_chunks(chunk_papers(papers))
    hits = retriever.search("learning model", k=10, filters={"year_min": 2020, "year_max": 2022})
    years = {h.metadata["year"] for h in hits}
    assert years <= {2020, 2022}
    assert "2005.14165" in {h.arxiv_id for h in hits} or "2203.15556" in {h.arxiv_id for h in hits}


def test_bm25_category_filter(papers) -> None:
    retriever = BM25Retriever.from_chunks(chunk_papers(papers))
    hits = retriever.search("learning", k=10, filters={"category": "cs.AI"})
    assert hits
    assert all("cs.AI" in h.metadata["categories"] for h in hits)
    assert any(h.arxiv_id == "1509.06461" for h in hits)


def test_empty_query_returns_nothing(papers) -> None:
    retriever = BM25Retriever.from_chunks(chunk_papers(papers))
    assert retriever.search("   ", k=5) == []
    assert retriever.search("zzzzzzzzzzz", k=5) == []


def test_bm25_roundtrip_pickle(papers, tmp_path) -> None:
    original = BM25Retriever.from_chunks(chunk_papers(papers))
    original.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path)
    a = [h.arxiv_id for h in original.search("transformer attention", k=3)]
    b = [h.arxiv_id for h in loaded.search("transformer attention", k=3)]
    assert a == b
