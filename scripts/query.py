#!/usr/bin/env python3
"""Command-line query against a built index (requires a running Ollama server to generate)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_rag.config import load_config  # noqa: E402
from arxiv_rag.pipeline import load_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="The question to ask.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--category", choices=["cs.AI", "cs.CL", "cs.LG"], default=None)
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "hybrid+rerank"],
        default="hybrid+rerank",
    )
    parser.add_argument("--retrieve-only", action="store_true", help="Skip generation and print retrieved chunks.")
    parser.add_argument("--json", action="store_true", help="Print a JSON payload instead of text.")
    args = parser.parse_args()

    question = " ".join(args.question)
    cfg = load_config(args.config)
    pipeline = load_pipeline(cfg, require_index=True)
    filters = {}
    if args.category:
        filters["category"] = args.category
    if args.year_min is not None:
        filters["year_min"] = args.year_min
    if args.year_max is not None:
        filters["year_max"] = args.year_max

    if args.retrieve_only:
        chunks = pipeline.retrieve(question, filters=filters or None, method=args.method)
        rows = [c.to_dict() for c in chunks]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for i, row in enumerate(rows, start=1):
                print(f"{i}. [{row['arxiv_id']}] {row['title']} ({row.get('year')})  score={row['score']:.4f}")
                print(f"   {row['url']}")
                print(f"   {row['text'][:280].replace(chr(10), ' ')}...")
                print()
        return 0

    result = pipeline.answer(question, filters=filters or None, method=args.method)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(result.answer)
    print()
    print("Sources:")
    for src in result.sources:
        flag = "*" if src.get("cited") else " "
        print(f" {flag} [{src['arxiv_id']}] {src.get('title')} — {src.get('url')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
