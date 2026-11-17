"""Vector store interface and a FAISS implementation with metadata filters."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _apply_filters(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    category = filters.get("category")
    if category:
        cats = meta.get("categories") or []
        if category not in cats:
            return False
    year_min = filters.get("year_min")
    year_max = filters.get("year_max")
    year = meta.get("year")
    if year_min is not None or year_max is not None:
        if year is None:
            return False
        if year_min is not None and year < int(year_min):
            return False
        if year_max is not None and year > int(year_max):
            return False
    return True


class VectorStore(ABC):
    dim: int

    @abstractmethod
    def add(self, embeddings: np.ndarray, metadatas: list[dict[str, Any]], ids: list[str]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def save(self, directory: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> VectorStore:
        ...


class FaissVectorStore(VectorStore):
    """Exact inner-product (cosine, if vectors are normalised) FAISS index."""

    def __init__(self, dim: int, index_type: str = "flat_ip"):
        import faiss

        self.dim = dim
        self.index_type = index_type
        self._faiss = faiss
        if index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported faiss_index_type: {index_type}")
        self.metadatas: list[dict[str, Any]] = []
        self.ids: list[str] = []

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, embeddings: np.ndarray, metadatas: list[dict[str, Any]], ids: list[str]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (n, {self.dim}) embeddings, got {embeddings.shape}")
        if not (len(metadatas) == len(ids) == embeddings.shape[0]):
            raise ValueError("embeddings, metadatas, and ids must have the same length")
        vectors = np.ascontiguousarray(embeddings.astype(np.float32))
        self.index.add(vectors)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        vector = np.ascontiguousarray(query_embedding.astype(np.float32).reshape(1, -1))
        # Over-retrieve when filters are set so we can drop non-matching rows.
        overshoot = min(self.index.ntotal, max(k * 8, k, 50) if filters else k)
        # Grow the candidate pool if filtering removed too many hits.
        while True:
            scores, indices = self.index.search(vector, overshoot)
            hits: list[dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx < 0:
                    continue
                meta = self.metadatas[int(idx)]
                if not _apply_filters(meta, filters):
                    continue
                hits.append(
                    {
                        "id": self.ids[int(idx)],
                        "score": float(score),
                        "metadata": meta,
                    }
                )
                if len(hits) >= k:
                    return hits
            if overshoot >= self.index.ntotal:
                return hits
            overshoot = min(self.index.ntotal, overshoot * 2)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / "faiss.index"))
        payload = {
            "dim": self.dim,
            "index_type": self.index_type,
            "ids": self.ids,
            "metadatas": self.metadatas,
        }
        with (directory / "faiss_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        logger.info("Saved FAISS index with %s vectors to %s", len(self.ids), directory)

    @classmethod
    def load(cls, directory: Path) -> FaissVectorStore:
        import faiss

        meta_path = directory / "faiss_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No FAISS metadata at {meta_path}")
        with meta_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(dim=int(payload["dim"]), index_type=payload.get("index_type", "flat_ip"))
        store.index = faiss.read_index(str(directory / "faiss.index"))
        store.ids = list(payload["ids"])
        store.metadatas = list(payload["metadatas"])
        return store
