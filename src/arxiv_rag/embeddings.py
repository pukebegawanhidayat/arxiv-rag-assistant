"""Embedding backends: SentenceTransformer plus a deterministic dummy for tests."""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from arxiv_rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Interface shared by the real model and the test double."""

    dim: int

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """Return an array of shape (len(texts), dim), optionally L2-normalised."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig, cache_root: Path | None = None):
        # Imported lazily so unit tests can run without the model weights.
        from sentence_transformers import SentenceTransformer

        cache = Path(cfg.cache_dir)
        if not cache.is_absolute() and cache_root is not None:
            cache = cache_root / cache
        if not cache.is_absolute():
            os.environ.setdefault("HF_HOME", str(cache))
        cache.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model %s on %s", cfg.model_name, cfg.device)
        self._model = SentenceTransformer(
            cfg.model_name,
            device=cfg.device,
            cache_folder=str(cache),
        )
        self._normalize = cfg.normalize
        self._batch_size = cfg.batch_size
        self._query_prefix = cfg.query_prefix or ""
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = cfg.model_name

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        is_query: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = texts
        if is_query and self._query_prefix:
            payload = [f"{self._query_prefix}{text}" for text in texts]
        vectors = self._model.encode(
            payload,
            batch_size=batch_size or self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(Embedder):
    """Fast, deterministic bag-of-features embedder used in tests (no downloads)."""

    def __init__(self, dim: int = 64, normalize: bool = True, seed: int = 0):
        self.dim = dim
        self._normalize = normalize
        self._seed = seed

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self._seed}:{token}".encode()).digest()
        needed = self.dim
        raw = (digest * ((needed // len(digest)) + 1))[:needed]
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0) - 0.5

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,  # noqa: ARG002
        is_query: bool = False,  # noqa: ARG002
    ) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = [t for t in text.lower().split() if t]
            if not tokens:
                continue
            acc = np.zeros(self.dim, dtype=np.float32)
            for tok in tokens:
                acc += self._token_vector(tok)
            matrix[i] = acc
        if self._normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            matrix = matrix / norms
        return matrix
