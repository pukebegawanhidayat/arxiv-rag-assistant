"""API tests using FastAPI's TestClient and a mocked pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from arxiv_rag.api import create_app
from arxiv_rag.pipeline import RAGPipeline
from tests.pdf_samples import MINIMAL_PDF


def test_health_ok(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["ollama"] is True  # StaticLLMClient reports available
    assert body["n_chunks"] == pipeline.n_arxiv_chunks


def test_health_degraded_without_pipeline(cfg) -> None:
    app = create_app(pipeline=None, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "degraded"
    assert res.json()["index_loaded"] is False


def test_query_returns_answer_and_sources(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post("/query", json={"question": "What is the Transformer architecture?"})
    assert res.status_code == 200
    body = res.json()
    assert body["refused"] is False
    assert "1706.03762" in body["answer"]
    assert body["sources"]
    urls = [s.get("url") for s in body["sources"]]
    assert "https://arxiv.org/abs/1706.03762" in urls


def test_query_honours_filters(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post(
            "/query",
            json={
                "question": "What method reduces overestimation bias in Q-learning?",
                "year_min": 2015,
                "year_max": 2016,
                "category": "cs.AI",
            },
        )
    assert res.status_code == 200
    ids = {s["arxiv_id"] for s in res.json()["sources"]}
    assert ids <= {"1509.06461"}


def test_query_rejects_inverted_years(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post(
            "/query",
            json={"question": "What is attention?", "year_min": 2024, "year_max": 2010},
        )
    assert res.status_code == 400


def test_query_pdf_source_requires_session(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post("/query", json={"question": "What is in the pdf?", "source": "pdf"})
    assert res.status_code == 400


def test_upload_pdf_and_query(pipeline: RAGPipeline, cfg, tmp_path) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post(
            "/upload-pdf",
            files={"file": ("attention.pdf", MINIMAL_PDF, "application/pdf")},
        )
        assert res.status_code == 200, res.text
        payload = res.json()
        assert payload["n_chunks"] >= 1
        assert payload["filename"] == "attention.pdf"
        session_id = payload["session_id"]
        saved = tmp_path / "uploads" / f"{session_id}.pdf"
        assert saved.exists()

        res = client.post(
            "/query",
            json={
                "question": "What architecture uses self-attention?",
                "source": "pdf",
                "session_id": session_id,
            },
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["sources"]
        assert all(str(s["arxiv_id"]).startswith("pdf:") for s in body["sources"])


def test_upload_rejects_non_pdf(pipeline: RAGPipeline, cfg) -> None:
    app = create_app(pipeline=pipeline, cfg=cfg, load_index=False)
    with TestClient(app) as client:
        res = client.post(
            "/upload-pdf",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert res.status_code == 400
