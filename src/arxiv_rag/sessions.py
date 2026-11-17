"""In-memory session indexes for user-uploaded PDFs."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from arxiv_rag.config import AppConfig
from arxiv_rag.data.pdf import chunk_pdf
from arxiv_rag.embeddings import Embedder
from arxiv_rag.retrievers.base import RetrievedChunk, reciprocal_rank_fusion
from arxiv_rag.retrievers.bm25 import BM25Retriever
from arxiv_rag.retrievers.dense import DenseRetriever
from arxiv_rag.retrievers.hybrid import HybridRetriever
from arxiv_rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


@dataclass
class PdfSession:
    session_id: str
    filename: str
    n_chunks: int
    hybrid: HybridRetriever
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def search(
        self,
        query: str,
        k: int = 50,
        filters: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[RetrievedChunk]:
        # Year / category filters apply to the arXiv collection, not uploads.
        return self.hybrid.search(query, k=k, filters=None)


class SessionStore:
    """Build a throw-away FAISS + BM25 index for each uploaded PDF."""

    def __init__(self, embedder: Embedder, cfg: AppConfig):
        self.embedder = embedder
        self.cfg = cfg
        self.sessions: dict[str, PdfSession] = {}

    def add_pdf(self, filename: str, data: bytes, session_id: str | None = None) -> PdfSession:
        session_id = session_id or uuid.uuid4().hex
        chunks = chunk_pdf(
            data,
            filename=filename,
            session_id=session_id,
            max_chars=self.cfg.chunking.max_chars,
            overlap_chars=self.cfg.chunking.overlap_chars,
        )
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode(texts)
        store = FaissVectorStore(dim=self.embedder.dim, index_type=self.cfg.index.faiss_index_type)
        metadatas = []
        ids = []
        for chunk in chunks:
            meta = dict(chunk)
            ids.append(chunk["chunk_id"])
            metadatas.append(meta)
        store.add(vectors, metadatas, ids)
        bm25 = BM25Retriever.from_chunks(chunks)
        dense = DenseRetriever(store, self.embedder)
        hybrid = HybridRetriever(
            bm25,
            dense,
            rrf_k=self.cfg.retrieval.rrf_k,
            bm25_k=self.cfg.retrieval.bm25_k,
            dense_k=self.cfg.retrieval.dense_k,
        )
        session = PdfSession(
            session_id=session_id,
            filename=filename,
            n_chunks=len(chunks),
            hybrid=hybrid,
            chunks=chunks,
        )
        self.sessions[session_id] = session
        logger.info("Indexed PDF %s into session %s (%s chunks)", filename, session_id, len(chunks))
        return session

    def get(self, session_id: str) -> PdfSession | None:
        return self.sessions.get(session_id)


def merge_collections(
    rankings: list[list[RetrievedChunk]],
    rrf_k: int = 60,
    k: int = 50,
) -> list[RetrievedChunk]:
    """Fuse hits from the arXiv index and a PDF session with RRF."""
    return reciprocal_rank_fusion(rankings, k=rrf_k)[:k]
