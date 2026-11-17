"""Synthetic known-item retrieval evaluation and optional LLM-as-judge."""

from __future__ import annotations

import json
import logging
import random
import re
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from arxiv_rag.data.chunking import split_sentences
from arxiv_rag.llm import LLMClient
from arxiv_rag.retrievers.base import RetrievedChunk

logger = logging.getLogger(__name__)

GENERIC_LEAD = re.compile(
    r"^(we (present|propose|introduce|describe|show|study)|"
    r"in this (paper|work|article)|this paper|this work)\b",
    re.IGNORECASE,
)


def pick_query_span(abstract: str) -> str | None:
    """Choose a distinctive sentence from an abstract to use as a query."""
    sentences = split_sentences(abstract)
    decent = [
        sent
        for sent in sentences
        if 80 <= len(sent) <= 400 and not GENERIC_LEAD.match(sent)
    ]
    if not decent:
        decent = [sent for sent in sentences if 40 <= len(sent) <= 500]
    if not decent:
        text = abstract.strip()
        return text if len(text) >= 40 else None
    # Prefer a later sentence (contribution / results) over the opening hook.
    return decent[len(decent) // 2]


def build_eval_questions(
    papers: Sequence[dict[str, Any]],
    n_questions: int = 100,
    seed: int = 123,
) -> list[dict[str, Any]]:
    """Build a known-item eval set: each query is a span whose gold id is known."""
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for paper in papers:
        span = pick_query_span(str(paper.get("abstract") or ""))
        if not span:
            continue
        candidates.append(
            {
                "id": f"q-{paper['arxiv_id']}",
                "query": span,
                "gold_arxiv_id": paper["arxiv_id"],
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "categories": list(paper.get("categories") or []),
            }
        )
    if len(candidates) < n_questions:
        raise ValueError(
            f"Only {len(candidates)} papers produced a usable query span; "
            f"need {n_questions}."
        )
    sample = rng.sample(candidates, n_questions)
    sample.sort(key=lambda item: item["gold_arxiv_id"])
    return sample


def first_relevant_rank(chunks: Sequence[RetrievedChunk], gold_arxiv_id: str) -> int | None:
    """1-based rank of the first chunk whose paper id matches, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.arxiv_id == gold_arxiv_id:
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    values = [1.0 / rank if rank else 0.0 for rank in ranks]
    return sum(values) / len(values)


def evaluate_retrievers(
    questions: Sequence[dict[str, Any]],
    retrieve_fn: Callable[[str, str, int], list[RetrievedChunk]],
    methods: Sequence[str] = ("bm25", "dense", "hybrid", "hybrid+rerank"),
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Run each retrieval method on the eval set and compute Recall@k and MRR.

    ``retrieve_fn(query, method, k)`` must return a ranked list of chunks.
    """
    max_k = max(ks)
    results: dict[str, Any] = {"n_questions": len(questions), "ks": list(ks), "methods": {}}
    for method in methods:
        ranks: list[int | None] = []
        per_question: list[dict[str, Any]] = []
        for item in questions:
            chunks = retrieve_fn(item["query"], method, max_k)
            rank = first_relevant_rank(chunks, item["gold_arxiv_id"])
            ranks.append(rank)
            per_question.append(
                {
                    "id": item["id"],
                    "gold_arxiv_id": item["gold_arxiv_id"],
                    "rank": rank,
                    "top_ids": [c.arxiv_id for c in chunks[:max_k]],
                }
            )
        metrics = {f"recall@{k}": round(recall_at_k(ranks, k), 4) for k in ks}
        metrics["mrr"] = round(mean_reciprocal_rank(ranks), 4)
        # Also report a few descriptive stats.
        found = [r for r in ranks if r is not None]
        metrics["found"] = len(found)
        metrics["median_rank_found"] = float(statistics.median(found)) if found else None
        results["methods"][method] = {"metrics": metrics, "per_question": per_question}
        logger.info(
            "%s: %s",
            method,
            ", ".join(f"{key}={value}" for key, value in metrics.items() if key != "median_rank_found"),
        )
    return results


def summarise_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Drop per-question rows so the summary is easy to paste into a README."""
    return {
        "n_questions": results.get("n_questions"),
        "ks": results.get("ks"),
        "methods": {
            name: payload["metrics"]
            for name, payload in results.get("methods", {}).items()
        },
    }


JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers.
Score the assistant's answer using only the provided context excerpts.
Return a single JSON object with these integer fields (0 or 1) and a short reason:
- faithfulness: 1 if every factual claim in the answer is supported by the context, otherwise 0.
- relevance: 1 if the answer addresses the user's question, otherwise 0.
Do not add any text outside the JSON object."""


def judge_answer(
    llm: LLMClient,
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """Optional LLM-as-judge for faithfulness and answer relevance."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\n"
                "Return JSON only."
            ),
        },
    ]
    raw = llm.generate(messages)
    payload = _extract_json(raw)
    faithfulness = int(payload.get("faithfulness", 0))
    relevance = int(payload.get("relevance", 0))
    return {
        "faithfulness": 1 if faithfulness else 0,
        "relevance": 1 if relevance else 0,
        "reason": str(payload.get("reason", "")),
        "raw": raw,
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
