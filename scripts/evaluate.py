#!/usr/bin/env python3
"""Run known-item retrieval evaluation (and optional LLM-as-judge generation eval)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_rag.config import load_config  # noqa: E402
from arxiv_rag.data.loader import load_processed_papers  # noqa: E402
from arxiv_rag.evaluation import (  # noqa: E402
    build_eval_questions,
    evaluate_retrievers,
    judge_answer,
    load_json,
    save_json,
    summarise_metrics,
)
from arxiv_rag.pipeline import load_pipeline  # noqa: E402
from arxiv_rag.prompts import format_context  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--rebuild-questions", action="store_true")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also generate answers and score faithfulness/relevance with the local Ollama model.",
    )
    parser.add_argument("--llm-n", type=int, default=10, help="How many questions to judge when --with-llm is set.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    questions_path = cfg.resolve(cfg.eval.questions_path)
    metrics_path = cfg.resolve(cfg.eval.metrics_path)
    n_questions = args.n_questions or cfg.eval.n_questions

    if questions_path.exists() and not args.rebuild_questions:
        questions = load_json(questions_path)
        logger.info("Loaded %s cached eval questions from %s", len(questions), questions_path)
    else:
        papers = load_processed_papers(cfg.processed_file)
        questions = build_eval_questions(papers, n_questions=n_questions, seed=cfg.eval.random_seed)
        save_json(questions_path, questions)
        logger.info("Wrote %s eval questions to %s", len(questions), questions_path)

    pipeline = load_pipeline(cfg, require_index=True)

    def retrieve_fn(query: str, method: str, k: int):
        return pipeline.retrieve(query, method=method, k=k)  # type: ignore[arg-type]

    results = evaluate_retrievers(questions, retrieve_fn, ks=cfg.eval.ks)
    summary = summarise_metrics(results)
    payload = {
        "summary": summary,
        "details": results,
        "embedding_model": cfg.embedding.model_name,
        "reranker_model": cfg.retrieval.reranker_model,
        "n_chunks": pipeline.n_arxiv_chunks,
        "index_dir": str(cfg.index_dir),
    }

    if args.with_llm:
        if not pipeline.llm.is_available():
            logger.warning("Ollama is not available at %s; skipping LLM-as-judge.", cfg.llm.base_url)
            payload["llm_judge"] = {"status": "skipped", "reason": "ollama_unavailable"}
        else:
            judge_rows = []
            subset = questions[: max(0, args.llm_n)]
            for item in subset:
                result = pipeline.answer(item["query"], method="hybrid+rerank")
                context = format_context(result.chunks)
                try:
                    scores = judge_answer(pipeline.llm, item["query"], result.answer, context)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Judge failed for %s: %s", item["id"], exc)
                    scores = {"faithfulness": None, "relevance": None, "reason": str(exc)}
                judge_rows.append(
                    {
                        "id": item["id"],
                        "gold_arxiv_id": item["gold_arxiv_id"],
                        "refused": result.refused,
                        "answer": result.answer,
                        **{k: scores.get(k) for k in ("faithfulness", "relevance", "reason")},
                    }
                )
            valid_f = [r["faithfulness"] for r in judge_rows if r["faithfulness"] is not None]
            valid_r = [r["relevance"] for r in judge_rows if r["relevance"] is not None]
            payload["llm_judge"] = {
                "status": "ok",
                "model": cfg.llm.model,
                "n": len(judge_rows),
                "faithfulness": round(sum(valid_f) / len(valid_f), 4) if valid_f else None,
                "relevance": round(sum(valid_r) / len(valid_r), 4) if valid_r else None,
                "rows": judge_rows,
            }

    save_json(metrics_path, payload)
    save_json(cfg.resolve(cfg.eval.output_dir) / "retrieval_metrics_summary.json", summary)
    logger.info("Summary:\n%s", json_dumps(summary))
    logger.info("Wrote metrics to %s", metrics_path)
    return 0


def json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
