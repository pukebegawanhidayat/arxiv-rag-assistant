"""Pipeline tests with a hashing embedder, lexical reranker, and mocked LLM."""

from __future__ import annotations

from arxiv_rag.llm import StaticLLMClient
from arxiv_rag.pipeline import RAGPipeline
from arxiv_rag.prompts import REFUSAL_MESSAGE


def test_retrieve_hybrid_finds_transformer_paper(pipeline: RAGPipeline) -> None:
    hits = pipeline.retrieve("transformer self-attention machine translation", method="hybrid", k=3)
    assert hits
    assert hits[0].arxiv_id == "1706.03762"


def test_retrieve_variants_are_available(pipeline: RAGPipeline) -> None:
    query = "compute-optimal allocation between parameters and tokens Chinchilla"
    for method in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        hits = pipeline.retrieve(query, method=method, k=5)  # type: ignore[arg-type]
        assert hits, method
        ids = [h.arxiv_id for h in hits]
        assert "2203.15556" in ids


def test_year_and_category_filters(pipeline: RAGPipeline) -> None:
    hits = pipeline.retrieve(
        "learning",
        method="bm25",
        k=10,
        filters={"year_min": 2015, "year_max": 2016, "category": "cs.AI"},
    )
    assert hits
    assert all(h.arxiv_id == "1509.06461" for h in hits)


def test_answer_calls_llm_with_context_and_returns_citations(pipeline: RAGPipeline, llm: StaticLLMClient) -> None:
    result = pipeline.answer("What architecture dispenses with recurrence?")
    assert not result.refused
    assert "1706.03762" in result.answer
    assert llm.calls, "the mocked LLM should have been invoked"
    bundled = llm.calls[0][1]["content"]
    assert "1706.03762" in bundled or "Transformer" in bundled
    cited = [s for s in result.sources if s.get("cited")]
    assert any(s["arxiv_id"] == "1706.03762" for s in cited)
    assert any(s.get("url") == "https://arxiv.org/abs/1706.03762" for s in result.sources)


def test_refuse_when_best_score_below_threshold(pipeline: RAGPipeline, llm: StaticLLMClient) -> None:
    pipeline.cfg.retrieval.min_rerank_score = 999.0
    before = len(llm.calls)
    result = pipeline.answer("What architecture dispenses with recurrence?")
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE
    assert len(llm.calls) == before


def test_refuse_when_no_chunks_retrieved(pipeline: RAGPipeline) -> None:
    # Drop the arXiv index so a query against it yields nothing.
    pipeline.hybrid = None
    result = pipeline.answer("anything at all", source="arxiv")
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE


def test_pdf_session_can_be_queried_instead_of_arxiv(pipeline: RAGPipeline) -> None:
    from tests.pdf_samples import MINIMAL_PDF

    session = pipeline.sessions.add_pdf("attention.pdf", MINIMAL_PDF)
    hits = pipeline.retrieve(
        "transformer architecture self-attention",
        method="hybrid",
        source="pdf",
        session_id=session.session_id,
        k=3,
    )
    assert hits
    assert all(h.arxiv_id.startswith("pdf:") for h in hits)


def test_pdf_and_arxiv_can_be_fused(pipeline: RAGPipeline) -> None:
    from tests.pdf_samples import MINIMAL_PDF

    session = pipeline.sessions.add_pdf("attention.pdf", MINIMAL_PDF)
    hits = pipeline.retrieve(
        "transformer self-attention",
        method="hybrid",
        source="both",
        session_id=session.session_id,
        k=8,
    )
    ids = {h.arxiv_id for h in hits}
    assert "1706.03762" in ids
    assert any(i.startswith("pdf:") for i in ids)
