"""End-to-end RAG pipeline: retrieve, rerank, generate, cite, or refuse."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from arxiv_rag.citations import extract_citation_ids, sources_from_chunks
from arxiv_rag.config import AppConfig
from arxiv_rag.embeddings import Embedder
from arxiv_rag.llm import LLMClient
from arxiv_rag.prompts import REFUSAL_MESSAGE, build_messages
from arxiv_rag.reranker import Reranker
from arxiv_rag.retrievers.base import RetrievedChunk
from arxiv_rag.retrievers.hybrid import HybridRetriever
from arxiv_rag.sessions import SessionStore, merge_collections

logger = logging.getLogger(__name__)

RetrievalMethod = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]
SourceMode = Literal["arxiv", "pdf", "both"]


@dataclass
class QueryResult:
    answer: str
    refused: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    method: str = "hybrid+rerank"
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "refused": self.refused,
            "sources": self.sources,
            "method": self.method,
            "model": self.model,
        }


class RAGPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        hybrid: HybridRetriever | None,
        reranker: Reranker,
        llm: LLMClient,
        embedder: Embedder | None = None,
        sessions: SessionStore | None = None,
        n_arxiv_chunks: int = 0,
    ):
        self.cfg = cfg
        self.hybrid = hybrid
        self.reranker = reranker
        self.llm = llm
        self.embedder = embedder
        self.sessions = sessions
        self.n_arxiv_chunks = n_arxiv_chunks

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        method: RetrievalMethod = "hybrid+rerank",
        source: SourceMode = "arxiv",
        session_id: str | None = None,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run one retrieval variant and optionally rerank."""
        if method == "hybrid+rerank":
            final_k = k or self.cfg.retrieval.final_k
            pool_k = max(self.cfg.retrieval.rerank_top_n, final_k)
            candidates = self._collect(
                query,
                filters=filters,
                source=source,
                session_id=session_id,
                k=pool_k,
                method="hybrid",
            )
            return self.reranker.rerank(query, candidates, top_n=final_k)
        top_k = k or self.cfg.retrieval.final_k
        return self._collect(
            query,
            filters=filters,
            source=source,
            session_id=session_id,
            k=top_k,
            method=method,
        )

    def _collect(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        source: SourceMode,
        session_id: str | None,
        k: int,
        method: RetrievalMethod = "hybrid",
    ) -> list[RetrievedChunk]:
        arxiv_hits: list[RetrievedChunk] = []
        pdf_hits: list[RetrievedChunk] = []
        if source in ("arxiv", "both") and self.hybrid is not None:
            arxiv_hits = self._search_arxiv(query, filters=filters, method=method, k=k)
        if source in ("pdf", "both"):
            if not session_id:
                if source == "pdf":
                    raise ValueError("session_id is required when source='pdf'")
            else:
                if self.sessions is None:
                    raise ValueError("No PDF session store is configured.")
                session = self.sessions.get(session_id)
                if session is None:
                    raise KeyError(f"Unknown PDF session: {session_id}")
                pdf_hits = session.search(query, k=k)
        if source == "arxiv":
            return arxiv_hits
        if source == "pdf":
            return pdf_hits
        return merge_collections(
            [arxiv_hits, pdf_hits],
            rrf_k=self.cfg.retrieval.rrf_k,
            k=k,
        )

    def _search_arxiv(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        method: RetrievalMethod,
        k: int,
    ) -> list[RetrievedChunk]:
        assert self.hybrid is not None
        if method == "bm25":
            return self.hybrid.bm25.search(query, k=k, filters=filters)
        if method == "dense":
            return self.hybrid.dense.search(query, k=k, filters=filters)
        # "hybrid" and "hybrid+rerank" both start from fused candidates.
        return self.hybrid.search(query, k=k, filters=filters)

    def answer(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        source: SourceMode = "arxiv",
        session_id: str | None = None,
        method: RetrievalMethod = "hybrid+rerank",
    ) -> QueryResult:
        chunks = self.retrieve(
            query,
            filters=filters,
            method=method,
            source=source,
            session_id=session_id,
        )
        model_name = getattr(self.llm, "model", "") or ""
        if self._should_refuse(chunks):
            return QueryResult(
                answer=REFUSAL_MESSAGE,
                refused=True,
                sources=sources_from_chunks(chunks),
                chunks=chunks,
                method=method,
                model=model_name,
            )
        messages = build_messages(query, chunks)
        text = self.llm.generate(messages)
        cited = extract_citation_ids(text)
        return QueryResult(
            answer=text,
            refused=False,
            sources=sources_from_chunks(chunks, cited_ids=cited),
            chunks=chunks,
            method=method,
            model=model_name,
        )

    def _should_refuse(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return True
        threshold = self.cfg.retrieval.min_rerank_score
        best = chunks[0].score
        if best < threshold:
            logger.info("Refusing: best score %.3f is below threshold %.3f", best, threshold)
            return True
        return False


def build_hybrid(
    chunks: list[dict[str, Any]],
    embedder: Embedder,
    cfg: AppConfig,
    embeddings=None,
) -> HybridRetriever:
    """Build an in-memory hybrid retriever from chunk dicts."""
    from arxiv_rag.retrievers.bm25 import BM25Retriever
    from arxiv_rag.retrievers.dense import DenseRetriever
    from arxiv_rag.vectorstore import FaissVectorStore

    if embeddings is None:
        embeddings = embedder.encode([c["text"] for c in chunks])
    store = FaissVectorStore(dim=embedder.dim, index_type=cfg.index.faiss_index_type)
    metadatas = [dict(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    store.add(embeddings, metadatas, ids)
    bm25 = BM25Retriever.from_chunks(chunks)
    dense = DenseRetriever(store, embedder)
    return HybridRetriever(
        bm25,
        dense,
        rrf_k=cfg.retrieval.rrf_k,
        bm25_k=cfg.retrieval.bm25_k,
        dense_k=cfg.retrieval.dense_k,
    )


def load_index_hybrid(cfg: AppConfig, embedder: Embedder) -> tuple[HybridRetriever, int]:
    """Load a previously built FAISS + BM25 index from disk."""
    from arxiv_rag.retrievers.bm25 import BM25Retriever
    from arxiv_rag.retrievers.dense import DenseRetriever
    from arxiv_rag.vectorstore import FaissVectorStore

    index_dir = cfg.index_dir
    store = FaissVectorStore.load(index_dir)
    bm25 = BM25Retriever.load(index_dir)
    dense = DenseRetriever(store, embedder)
    hybrid = HybridRetriever(
        bm25,
        dense,
        rrf_k=cfg.retrieval.rrf_k,
        bm25_k=cfg.retrieval.bm25_k,
        dense_k=cfg.retrieval.dense_k,
    )
    return hybrid, len(store)


def load_pipeline(
    cfg: AppConfig | None = None,
    *,
    require_index: bool = True,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RAGPipeline:
    """Construct a pipeline, loading real models and the on-disk index."""
    from arxiv_rag.config import load_config
    from arxiv_rag.embeddings import SentenceTransformerEmbedder
    from arxiv_rag.llm import OllamaClient
    from arxiv_rag.reranker import CrossEncoderReranker
    from arxiv_rag.sessions import SessionStore

    cfg = cfg or load_config()
    if embedder is None:
        embedder = SentenceTransformerEmbedder(cfg.embedding, cache_root=cfg.project_root)
    hybrid: HybridRetriever | None = None
    n_chunks = 0
    if require_index:
        hybrid, n_chunks = load_index_hybrid(cfg, embedder)
    if reranker is None:
        reranker = CrossEncoderReranker(
            cfg.retrieval.reranker_model,
            device=cfg.embedding.device,
            cache_dir=str(cfg.resolve(cfg.embedding.cache_dir)),
        )
    if llm is None:
        llm = OllamaClient(cfg.llm)
    sessions = SessionStore(embedder, cfg)
    return RAGPipeline(
        cfg=cfg,
        hybrid=hybrid,
        reranker=reranker,
        llm=llm,
        embedder=embedder,
        sessions=sessions,
        n_arxiv_chunks=n_chunks,
    )
