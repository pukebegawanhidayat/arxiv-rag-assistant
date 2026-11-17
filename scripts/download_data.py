#!/usr/bin/env python3
"""Download the Kaggle arXiv metadata snapshot, reusing the kagglehub cache if present."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a script without an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arxiv_rag.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_data")


def find_cached_snapshot() -> Path | None:
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "Cornell-University" / "arxiv"
    if not cache_root.exists():
        return None
    matches = list(cache_root.rglob("arxiv-metadata-oai-snapshot.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def download_with_kagglehub(dataset: str) -> Path:
    import kagglehub

    logger.info("Downloading %s via kagglehub (will reuse cache if possible)...", dataset)
    folder = Path(kagglehub.dataset_download(dataset))
    snapshot = folder / "arxiv-metadata-oai-snapshot.json"
    if snapshot.exists():
        return snapshot
    matches = list(folder.rglob("arxiv-metadata-oai-snapshot.json"))
    if not matches:
        raise FileNotFoundError(f"kagglehub downloaded {folder} but the snapshot JSON was not inside it.")
    return matches[0]


def link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src)
        logger.info("Symlinked %s -> %s", dest, src)
    except OSError:
        import shutil

        shutil.copy2(src, dest)
        logger.info("Copied %s -> %s", src, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional path to config.yaml")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Call kagglehub even if a cached snapshot is already on disk.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dest = cfg.raw_file

    if dest.exists() and not args.force_download:
        logger.info("Raw snapshot already present at %s (%.1f GB)", dest, dest.stat().st_size / 1e9)
        return 0

    cached = find_cached_snapshot()
    if cached is not None and not args.force_download:
        logger.info("Found kagglehub cache at %s (%.1f GB)", cached, cached.stat().st_size / 1e9)
        link_or_copy(cached, dest)
        return 0

    snapshot = download_with_kagglehub(cfg.data.kaggle_dataset)
    link_or_copy(snapshot, dest)
    logger.info("Ready: %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
