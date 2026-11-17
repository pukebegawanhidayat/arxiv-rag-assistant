"""Dense retriever over a VectorStore + Embedder."""

from __future__ import annotations

from typing import Any

from arxiv_rag.embeddings import Embedder
from arxiv_rag.retrievers.base import RetrievedChunk
from arxiv_rag.vectorstore import VectorStore


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, texts: dict[str, str] | None = None):
        self.store = store
        self.embedder = embedder
        # Fallback text lookup if metadata does not already include the chunk text.
        self.texts = texts or {}

    def search(
        self,
        query: str,
        k: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        vector = self.embedder.encode([query], is_query=True)[0]
        hits = self.store.search(vector, k=k, filters=filters)
        results: list[RetrievedChunk] = []
        for hit in hits:
            meta = dict(hit["metadata"])
            text = meta.pop("text", None) or self.texts.get(hit["id"], "")
            results.append(
                RetrievedChunk(
                    chunk_id=hit["id"],
                    text=text,
                    score=float(hit["score"]),
                    metadata=meta,
                    source="dense",
                )
            )
        return results
