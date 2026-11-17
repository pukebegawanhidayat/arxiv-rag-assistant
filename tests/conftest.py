"""Shared fixtures: tiny paper corpus, hashing embedder, mocked LLM pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from arxiv_rag.config import load_config
from arxiv_rag.data.chunking import chunk_papers
from arxiv_rag.embeddings import HashingEmbedder
from arxiv_rag.llm import StaticLLMClient
from arxiv_rag.pipeline import RAGPipeline, build_hybrid
from arxiv_rag.reranker import LexicalReranker
from arxiv_rag.sessions import SessionStore


def sample_papers() -> list[dict]:
    return [
        {
            "arxiv_id": "1706.03762",
            "title": "Attention Is All You Need",
            "abstract": (
                "We propose a new simple network architecture, the Transformer, "
                "based solely on attention mechanisms, dispensing with recurrence and "
                "convolutions entirely. Experiments on machine translation tasks show "
                "the Transformer is superior in quality while being more parallelizable."
            ),
            "authors": "Vaswani et al.",
            "categories": ["cs.CL", "cs.LG"],
            "year": 2017,
            "doi": "10.48550/arXiv.1706.03762",
        },
        {
            "arxiv_id": "1810.04805",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": (
                "BERT introduces a bidirectional encoder representation from transformers "
                "that is pre-trained with a masked language model objective and next "
                "sentence prediction. The model obtains new state-of-the-art results on "
                "GLUE and SQuAD after fine-tuning."
            ),
            "authors": "Devlin et al.",
            "categories": ["cs.CL"],
            "year": 2018,
            "doi": "",
        },
        {
            "arxiv_id": "2005.14165",
            "title": "Language Models are Few-Shot Learners",
            "abstract": (
                "GPT-3 is an autoregressive language model with 175 billion parameters. "
                "It exhibits strong few-shot in-context learning without gradient updates, "
                "including translation, question answering, and cloze tasks."
            ),
            "authors": "Brown et al.",
            "categories": ["cs.CL", "cs.LG"],
            "year": 2020,
            "doi": "",
        },
        {
            "arxiv_id": "1509.06461",
            "title": "Deep Reinforcement Learning with Double Q-learning",
            "abstract": (
                "Double Q-learning reduces overestimation bias in deep reinforcement "
                "learning by decoupling action selection from action evaluation. The "
                "algorithm is evaluated on Atari 2600 games and improves stability."
            ),
            "authors": "van Hasselt et al.",
            "categories": ["cs.AI", "cs.LG"],
            "year": 2015,
            "doi": "",
        },
        {
            "arxiv_id": "2203.15556",
            "title": "Training Compute-Optimal Large Language Models",
            "abstract": (
                "Chinchilla scaling laws show that for a given compute budget, smaller "
                "models trained on more tokens outperform under-trained larger models. "
                "The paper recommends a compute-optimal allocation between parameters and data."
            ),
            "authors": "Hoffmann et al.",
            "categories": ["cs.LG"],
            "year": 2022,
            "doi": "",
        },
    ]


@pytest.fixture
def papers() -> list[dict]:
    return sample_papers()


@pytest.fixture
def chunks(papers) -> list[dict]:
    return chunk_papers(papers, max_chars=1600, overlap_chars=200)


@pytest.fixture
def cfg(tmp_path: Path):
    cfg = load_config()
    cfg.api.uploads_dir = str(tmp_path / "uploads")
    return cfg


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64, normalize=True, seed=0)


@pytest.fixture
def llm() -> StaticLLMClient:
    return StaticLLMClient(
        reply=(
            "The Transformer relies on self-attention instead of recurrence "
            "[1706.03762]."
        )
    )


@pytest.fixture
def pipeline(cfg, chunks, embedder, llm) -> RAGPipeline:
    hybrid = build_hybrid(chunks, embedder, cfg)
    sessions = SessionStore(embedder, cfg)
    return RAGPipeline(
        cfg=cfg,
        hybrid=hybrid,
        reranker=LexicalReranker(),
        llm=llm,
        embedder=embedder,
        sessions=sessions,
        n_arxiv_chunks=len(chunks),
    )
