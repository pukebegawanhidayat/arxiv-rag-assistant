"""FastAPI application: /health, /query, /upload-pdf."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from arxiv_rag.config import AppConfig, load_config
from arxiv_rag.data.pdf import PdfExtractionError
from arxiv_rag.pipeline import RAGPipeline, SourceMode

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    year_min: int | None = None
    year_max: int | None = None
    category: Literal["cs.AI", "cs.CL", "cs.LG"] | None = None
    session_id: str | None = None
    source: SourceMode = "arxiv"
    method: Literal["bm25", "dense", "hybrid", "hybrid+rerank"] = "hybrid+rerank"


class SourceOut(BaseModel):
    chunk_id: str | None = None
    arxiv_id: str
    title: str = ""
    authors: str = ""
    year: int | None = None
    categories: list[str] = Field(default_factory=list)
    url: str | None = None
    score: float | None = None
    cited: bool = False
    text: str = ""


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceOut]
    method: str
    model: str = ""


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    n_chunks: int


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    ollama_url: str
    model: str
    index_loaded: bool
    n_chunks: int = 0
    n_pdf_sessions: int = 0


def _filters(req: QueryRequest) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if req.category:
        payload["category"] = req.category
    if req.year_min is not None:
        payload["year_min"] = req.year_min
    if req.year_max is not None:
        payload["year_max"] = req.year_max
    return payload or None


def _source_fields(item: dict[str, Any]) -> dict[str, Any]:
    allowed = SourceOut.model_fields.keys()
    return {key: item.get(key) for key in allowed}


def create_app(
    pipeline: RAGPipeline | None = None,
    cfg: AppConfig | None = None,
    *,
    load_index: bool = True,
) -> FastAPI:
    """Build the FastAPI app.

    Tests pass a ready-made ``pipeline`` and ``load_index=False`` so no models
    or on-disk indexes are required. The real index is loaded during lifespan
    when the API process starts, not at import time.
    """
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        if pipeline is not None:
            app.state.pipeline = pipeline
        elif load_index:
            from arxiv_rag.pipeline import load_pipeline

            logger.info("Loading RAG pipeline and indexes...")
            app.state.pipeline = load_pipeline(cfg, require_index=True)
            logger.info("Pipeline ready (%s chunks)", app.state.pipeline.n_arxiv_chunks)
        else:
            app.state.pipeline = None
        yield

    app = FastAPI(
        title="arXiv RAG Assistant",
        description="Domain-specific RAG assistant for AI/ML and NLP papers.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_pipeline() -> RAGPipeline:
        pipe: RAGPipeline | None = getattr(app.state, "pipeline", None)
        if pipe is None:
            raise HTTPException(
                status_code=503,
                detail="The RAG pipeline is not loaded. Build the index and restart the API.",
            )
        return pipe

    def get_pipeline_optional() -> RAGPipeline | None:
        return getattr(app.state, "pipeline", None)

    @app.get("/health", response_model=HealthResponse)
    def health(pipe: RAGPipeline | None = Depends(get_pipeline_optional)) -> HealthResponse:
        ollama_ok = False
        if pipe is not None and hasattr(pipe.llm, "is_available"):
            try:
                ollama_ok = bool(pipe.llm.is_available())
            except Exception:  # noqa: BLE001
                ollama_ok = False
        return HealthResponse(
            status="ok" if pipe is not None else "degraded",
            ollama=ollama_ok,
            ollama_url=cfg.llm.base_url,
            model=cfg.llm.model,
            index_loaded=bool(pipe and pipe.hybrid is not None),
            n_chunks=pipe.n_arxiv_chunks if pipe else 0,
            n_pdf_sessions=len(pipe.sessions.sessions) if pipe and pipe.sessions else 0,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, pipe: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
        if req.source in ("pdf", "both") and not req.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required when source is 'pdf' or 'both'.",
            )
        if req.year_min is not None and req.year_max is not None and req.year_min > req.year_max:
            raise HTTPException(status_code=400, detail="year_min cannot be greater than year_max.")
        try:
            result = pipe.answer(
                req.question,
                filters=_filters(req),
                source=req.source,
                session_id=req.session_id,
                method=req.method,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        sources = [SourceOut(**_source_fields(item)) for item in result.sources]
        return QueryResponse(
            answer=result.answer,
            refused=result.refused,
            sources=sources,
            method=result.method,
            model=result.model,
        )

    @app.post("/upload-pdf", response_model=UploadResponse)
    async def upload_pdf(
        file: UploadFile = File(...),
        pipe: RAGPipeline = Depends(get_pipeline),
    ) -> UploadResponse:
        if pipe.sessions is None or pipe.embedder is None:
            raise HTTPException(status_code=503, detail="PDF upload is not available.")
        filename = file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
        data = await file.read()
        max_bytes = cfg.api.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than the {cfg.api.max_upload_mb} MB limit.",
            )
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        try:
            session = pipe.sessions.add_pdf(filename, data)
        except PdfExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        uploads = cfg.uploads_dir
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / f"{session.session_id}.pdf"
        dest.write_bytes(data)
        return UploadResponse(
            session_id=session.session_id,
            filename=filename,
            n_chunks=session.n_chunks,
        )

    return app


def create_default_app() -> FastAPI:
    """Factory used by uvicorn: ``uvicorn arxiv_rag.api:app``."""
    skip = os.environ.get("ARXIV_RAG_SKIP_INDEX") == "1"
    return create_app(load_index=not skip)


app = create_default_app()
