"""Sparse BM25 retriever (rank_bm25) with a simple alphanumeric tokenizer."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from arxiv_rag.retrievers.base import RetrievedChunk
from arxiv_rag.vectorstore import _apply_filters

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(
        self,
        corpus_tokens: list[list[str]],
        chunks: list[dict[str, Any]],
    ):
        if len(corpus_tokens) != len(chunks):
            raise ValueError("corpus_tokens and chunks must be the same length")
        self._bm25 = BM25Okapi(corpus_tokens)
        self.chunks = chunks

    @classmethod
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> BM25Retriever:
        tokens = [tokenize(c["text"]) for c in chunks]
        return cls(tokens, chunks)

    def search(
        self,
        query: str,
        k: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort descending; we only materialise as many as we need after filters.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[RetrievedChunk] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            meta = {key: value for key, value in self.chunks[idx].items() if key != "text"}
            if not _apply_filters(meta, filters):
                continue
            hits.append(
                RetrievedChunk(
                    chunk_id=self.chunks[idx]["chunk_id"],
                    text=self.chunks[idx]["text"],
                    score=float(scores[idx]),
                    metadata=meta,
                    source="bm25",
                )
            )
            if len(hits) >= k:
                break
        return hits

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tokens = [tokenize(c["text"]) for c in self.chunks]
        with (directory / "bm25.pkl").open("wb") as fh:
            pickle.dump({"chunks": self.chunks, "tokens": tokens}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: Path) -> BM25Retriever:
        path = directory / "bm25.pkl"
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["tokens"], payload["chunks"])
