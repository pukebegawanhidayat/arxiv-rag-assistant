"""Hybrid retriever: BM25 + dense, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from arxiv_rag.retrievers.base import RetrievedChunk, reciprocal_rank_fusion
from arxiv_rag.retrievers.bm25 import BM25Retriever
from arxiv_rag.retrievers.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        bm25_k: int = 50,
        dense_k: int = 50,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.bm25_k = bm25_k
        self.dense_k = dense_k

    def search(
        self,
        query: str,
        k: int = 50,
        filters: dict[str, Any] | None = None,
        bm25_k: int | None = None,
        dense_k: int | None = None,
    ) -> list[RetrievedChunk]:
        sparse = self.bm25.search(query, k=bm25_k or self.bm25_k, filters=filters)
        dense = self.dense.search(query, k=dense_k or self.dense_k, filters=filters)
        fused = reciprocal_rank_fusion([sparse, dense], k=self.rrf_k)
        return fused[:k]
