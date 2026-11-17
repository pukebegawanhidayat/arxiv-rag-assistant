#!/usr/bin/env python3
"""Filter the arXiv snapshot, chunk papers, and build FAISS + BM25 indexes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_rag.config import load_config  # noqa: E402
from arxiv_rag.data.chunking import chunk_papers  # noqa: E402
from arxiv_rag.data.loader import prepare_papers  # noqa: E402
from arxiv_rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from arxiv_rag.retrievers.bm25 import BM25Retriever  # noqa: E402
from arxiv_rag.vectorstore import FaissVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-filter papers even if a processed file exists.")
    parser.add_argument("--sample-size", type=int, default=None, help="Override data.sample_size.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sample_size is not None:
        cfg.data.sample_size = args.sample_size

    if not cfg.raw_file.exists():
        logger.error("Raw snapshot not found at %s. Run: python scripts/download_data.py", cfg.raw_file)
        return 1

    papers = prepare_papers(cfg, force=args.force)
    if not papers:
        logger.error("No papers matched the configured filters.")
        return 1

    chunks = chunk_papers(
        papers,
        max_chars=cfg.chunking.max_chars,
        overlap_chars=cfg.chunking.overlap_chars,
        source_fields=cfg.chunking.source_fields,
    )
    logger.info("Built %s chunks from %s papers", len(chunks), len(papers))

    embedder = SentenceTransformerEmbedder(cfg.embedding, cache_root=cfg.project_root)
    vectors = embedder.encode([c["text"] for c in chunks])
    store = FaissVectorStore(dim=embedder.dim, index_type=cfg.index.faiss_index_type)
    store.add(vectors, [dict(c) for c in chunks], [c["chunk_id"] for c in chunks])

    index_dir = cfg.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)
    BM25Retriever.from_chunks(chunks).save(index_dir)

    years = [p["year"] for p in papers if p.get("year")]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_papers": len(papers),
        "n_chunks": len(chunks),
        "embedding_model": cfg.embedding.model_name,
        "embedding_dim": embedder.dim,
        "categories": cfg.data.categories,
        "sample_size": cfg.data.sample_size,
        "random_seed": cfg.data.random_seed,
        "min_year": cfg.data.min_year,
        "max_year": cfg.data.max_year,
        "year_min_observed": min(years) if years else None,
        "year_max_observed": max(years) if years else None,
        "faiss_index_type": cfg.index.faiss_index_type,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote index manifest to %s", index_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
