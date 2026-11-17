"""Shared retrieval types and Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @property
    def arxiv_id(self) -> str:
        return str(self.metadata.get("arxiv_id") or self.chunk_id.split("::", 1)[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "arxiv_id": self.arxiv_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "title": self.metadata.get("title", ""),
            "authors": self.metadata.get("authors", ""),
            "categories": self.metadata.get("categories", []),
            "year": self.metadata.get("year"),
            "doi": self.metadata.get("doi", ""),
            "url": (
                None
                if str(self.arxiv_id).startswith("pdf:")
                else f"https://arxiv.org/abs/{self.arxiv_id}"
            ),
        }


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merge ranked lists with RRF: score(d) = sum 1 / (k + rank).

    ``k`` is the standard smoothing constant (Cormack et al., 2009), not top-k.
    """
    fused: dict[str, tuple[float, RetrievedChunk]] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            addition = 1.0 / (k + rank)
            if chunk.chunk_id in fused:
                score, existing = fused[chunk.chunk_id]
                fused[chunk.chunk_id] = (score + addition, existing)
            else:
                fused[chunk.chunk_id] = (addition, chunk)
    merged: list[RetrievedChunk] = []
    for score, chunk in sorted(fused.values(), key=lambda item: item[0], reverse=True):
        merged.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
                source="hybrid",
            )
        )
    return merged
