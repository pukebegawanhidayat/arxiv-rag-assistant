"""Stream, filter, and sample the Kaggle arXiv metadata dump."""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from arxiv_rag.config import AppConfig, DataConfig

logger = logging.getLogger(__name__)

YEAR_RE = re.compile(r"(19|20)\d{2}")


def paper_year(record: dict[str, Any]) -> int | None:
    """Best-effort publication year from versions, update_date, or the arXiv id."""
    versions = record.get("versions") or []
    if versions:
        created = versions[0].get("created") or ""
        match = YEAR_RE.search(created)
        if match:
            return int(match.group(0))
    update = record.get("update_date") or ""
    match = YEAR_RE.search(update)
    if match:
        return int(match.group(0))
    # Legacy ids like "cs/0001001"; modern ids start with YYMM.
    arxiv_id = str(record.get("id") or "")
    if re.match(r"^\d{4}\.", arxiv_id):
        yy, mm = int(arxiv_id[:2]), int(arxiv_id[2:4])
        if 1 <= mm <= 12:
            return 1900 + yy if yy >= 91 else 2000 + yy
    return None


def _category_set(record: dict[str, Any]) -> set[str]:
    raw = record.get("categories") or ""
    if isinstance(raw, list):
        return {str(c).strip() for c in raw if str(c).strip()}
    return {tok for tok in str(raw).split() if tok}


def matches_categories(record: dict[str, Any], targets: Iterable[str]) -> bool:
    wanted = set(targets)
    return bool(_category_set(record) & wanted)


def in_year_range(record: dict[str, Any], min_year: int | None, max_year: int | None) -> bool:
    if min_year is None and max_year is None:
        return True
    year = paper_year(record)
    if year is None:
        return False
    if min_year is not None and year < min_year:
        return False
    return max_year is None or year <= max_year


def normalise_paper(record: dict[str, Any]) -> dict[str, Any]:
    """Keep a compact, stable schema used everywhere downstream."""
    title = re.sub(r"\s+", " ", str(record.get("title") or "")).strip()
    abstract = re.sub(r"\s+", " ", str(record.get("abstract") or "")).strip()
    authors = record.get("authors") or ""
    categories = sorted(_category_set(record))
    year = paper_year(record)
    return {
        "arxiv_id": str(record.get("id") or "").strip(),
        "title": title,
        "abstract": abstract,
        "authors": authors.strip() if isinstance(authors, str) else json.dumps(authors),
        "categories": categories,
        "year": year,
        "doi": (record.get("doi") or "") or "",
        "update_date": record.get("update_date") or "",
    }


def iter_raw_papers(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSON-Lines dump (the Kaggle snapshot format)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw arXiv metadata not found at {path}. "
            "Run: python scripts/download_data.py"
        )
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON on line %s of %s", line_no, path)


def reservoir_sample(
    records: Iterator[dict[str, Any]],
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Algorithm R: one-pass sample of k items from a stream of unknown length."""
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for i, item in enumerate(records):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def filter_and_sample_papers(
    raw_path: Path,
    cfg: DataConfig,
    progress_every: int = 200_000,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter the snapshot by category/year and optionally sample.

    If ``stats`` is provided it is filled with scan totals, multi-label
    category counts, and year counts over *all* matching papers (before sampling).
    """
    category_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    seen = 0
    kept = 0

    def filtered() -> Iterator[dict[str, Any]]:
        nonlocal seen, kept
        for record in iter_raw_papers(raw_path):
            seen += 1
            if seen % progress_every == 0:
                logger.info("Scanned %s raw records (%s matched so far)", seen, kept)
            if not matches_categories(record, cfg.categories):
                continue
            if not in_year_range(record, cfg.min_year, cfg.max_year):
                continue
            paper = normalise_paper(record)
            if not paper["arxiv_id"] or not paper["abstract"] or not paper["title"]:
                continue
            kept += 1
            for cat in paper["categories"]:
                if cat in cfg.categories:
                    category_counts[cat] += 1
            if paper["year"] is not None:
                year_counts[int(paper["year"])] += 1
            yield paper
        logger.info("Finished scan: %s raw records, %s matched filters", seen, kept)

    stream = filtered()
    if cfg.sample_size is None or cfg.sample_size <= 0:
        papers = list(stream)
    else:
        papers = reservoir_sample(stream, cfg.sample_size, cfg.random_seed)
    papers.sort(key=lambda p: p["arxiv_id"])
    logger.info("Retained %s papers after sampling", len(papers))
    if stats is not None:
        stats.update(
            {
                "raw_scanned": seen,
                "matched": kept,
                "retained": len(papers),
                "categories": cfg.categories,
                "sample_size": cfg.sample_size,
                "random_seed": cfg.random_seed,
                "category_counts": dict(sorted(category_counts.items())),
                "year_counts": {str(k): v for k, v in sorted(year_counts.items())},
            }
        )
    return papers


def save_processed_papers(papers: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for paper in papers:
            fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
    logger.info("Wrote %s papers to %s", len(papers), path)


def load_processed_papers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed paper file not found at {path}. "
            "Run: python scripts/build_index.py"
        )
    papers: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def prepare_papers(cfg: AppConfig, force: bool = False) -> list[dict[str, Any]]:
    """Load cached processed papers or build them from the raw snapshot."""
    processed = cfg.processed_file
    if processed.exists() and not force:
        papers = load_processed_papers(processed)
        logger.info("Loaded %s cached papers from %s", len(papers), processed)
        return papers
    stats: dict[str, Any] = {}
    papers = filter_and_sample_papers(cfg.raw_file, cfg.data, stats=stats)
    save_processed_papers(papers, processed)
    if stats:
        stats_path = processed.with_name("ingest_stats.json")
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote ingest stats to %s", stats_path)
    return papers
