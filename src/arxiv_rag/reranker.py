"""Cross-encoder reranker with a no-download fallback for tests."""

from __future__ import annotations

import logging
from typing import Protocol

from arxiv_rag.retrievers.base import RetrievedChunk
from arxiv_rag.retrievers.bm25 import tokenize

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu", cache_dir: str | None = None):
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder %s on %s", model_name, device)
        kwargs: dict = {"device": device}
        if cache_dir:
            kwargs["cache_folder"] = cache_dir
        try:
            self._model = CrossEncoder(model_name, **kwargs)
        except TypeError:
            # Older sentence-transformers versions may not accept cache_folder.
            self._model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                    source="rerank",
                )
            )
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_n]


class LexicalReranker:
    """Overlap-based stand-in used in unit tests (no model download)."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        q = set(tokenize(query))
        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            c = set(tokenize(chunk.text))
            overlap = len(q & c)
            denom = max(len(q), 1)
            score = overlap / denom
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                    source="rerank",
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_n]
