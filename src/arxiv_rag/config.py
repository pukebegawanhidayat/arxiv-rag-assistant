"""Load and resolve project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until pyproject.toml or configs/config.yaml is found."""
    env = os.environ.get("ARXIV_RAG_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (
            candidate / "configs" / "config.yaml"
        ).exists():
            return candidate
    return here


@dataclass
class DataConfig:
    kaggle_dataset: str = "Cornell-University/arxiv"
    raw_filename: str = "arxiv-metadata-oai-snapshot.json"
    raw_dir: str = "data/raw"
    processed_path: str = "data/processed/papers.jsonl"
    categories: list[str] = field(default_factory=lambda: ["cs.AI", "cs.CL", "cs.LG"])
    sample_size: int | None = 10000
    random_seed: int = 42
    min_year: int | None = None
    max_year: int | None = None


@dataclass
class ChunkingConfig:
    source_fields: list[str] = field(default_factory=lambda: ["title", "abstract"])
    max_chars: int = 1600
    overlap_chars: int = 200


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"
    cache_dir: str = ".cache/huggingface"
    query_prefix: str = "Represent this sentence for searching relevant passages: "


@dataclass
class IndexConfig:
    index_dir: str = "indexes"
    faiss_index_type: str = "flat_ip"


@dataclass
class RetrievalConfig:
    bm25_k: int = 50
    dense_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 20
    final_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_rerank_score: float = -2.0


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.1
    max_tokens: int = 800
    timeout_seconds: int = 180


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    max_upload_mb: int = 20
    uploads_dir: str = "uploads"


@dataclass
class EvalConfig:
    n_questions: int = 100
    ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    output_dir: str = "reports"
    questions_path: str = "reports/eval_questions.json"
    metrics_path: str = "reports/retrieval_metrics.json"
    random_seed: int = 123


@dataclass
class AppConfig:
    project_root: Path
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    api: ApiConfig
    eval: EvalConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    @property
    def raw_file(self) -> Path:
        return self.resolve(self.data.raw_dir) / self.data.raw_filename

    @property
    def processed_file(self) -> Path:
        return self.resolve(self.data.processed_path)

    @property
    def index_dir(self) -> Path:
        return self.resolve(self.index.index_dir)

    @property
    def uploads_dir(self) -> Path:
        return self.resolve(self.api.uploads_dir)


def _section(cls, payload: dict[str, Any], key: str):
    data = payload.get(key) or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config and apply environment variable overrides."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve() if not cfg_path.exists() else cfg_path.resolve()
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = AppConfig(
        project_root=root,
        data=_section(DataConfig, raw, "data"),
        chunking=_section(ChunkingConfig, raw, "chunking"),
        embedding=_section(EmbeddingConfig, raw, "embedding"),
        index=_section(IndexConfig, raw, "index"),
        retrieval=_section(RetrievalConfig, raw, "retrieval"),
        llm=_section(LLMConfig, raw, "llm"),
        api=_section(ApiConfig, raw, "api"),
        eval=_section(EvalConfig, raw, "eval"),
        raw=raw,
    )

    if os.environ.get("OLLAMA_BASE_URL"):
        cfg.llm.base_url = os.environ["OLLAMA_BASE_URL"].rstrip("/")
    if os.environ.get("OLLAMA_HOST") and not os.environ.get("OLLAMA_BASE_URL"):
        host = os.environ["OLLAMA_HOST"].rstrip("/")
        cfg.llm.base_url = host if host.startswith("http") else f"http://{host}"
    if os.environ.get("OLLAMA_MODEL"):
        cfg.llm.model = os.environ["OLLAMA_MODEL"]
    if os.environ.get("ARXIV_RAG_INDEX_DIR"):
        cfg.index.index_dir = os.environ["ARXIV_RAG_INDEX_DIR"]
    if os.environ.get("ARXIV_RAG_SAMPLE_SIZE"):
        cfg.data.sample_size = int(os.environ["ARXIV_RAG_SAMPLE_SIZE"])
    if os.environ.get("API_CORS_ORIGINS"):
        cfg.api.cors_origins = [
            origin.strip()
            for origin in os.environ["API_CORS_ORIGINS"].split(",")
            if origin.strip()
        ]
    return cfg
