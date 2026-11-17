"""Extract and link arXiv citations from generated answers."""

from __future__ import annotations

import re

from arxiv_rag.retrievers.base import RetrievedChunk

# Modern ids: 1706.03762 or 1706.03762v7. Legacy: cs/0001001, hep-th/9901001.
MODERN_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
LEGACY_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}"
ARXIV_ID = rf"(?:{MODERN_ID}|{LEGACY_ID})"
BRACKET_CITE_RE = re.compile(rf"\[({ARXIV_ID})\]", re.IGNORECASE)
NAKED_CITE_RE = re.compile(rf"(?:arxiv:|arXiv:)({ARXIV_ID})", re.IGNORECASE)
# PDF uploads are cited as [pdf:filename] or [pdf:session:n].
PDF_CITE_RE = re.compile(r"\[(pdf:[^\]]+)\]", re.IGNORECASE)


def extract_citation_ids(text: str) -> list[str]:
    """Return unique citation keys in the order they first appear."""
    found: list[str] = []
    seen: set[str] = set()
    for match in BRACKET_CITE_RE.finditer(text):
        key = match.group(1)
        if key not in seen:
            seen.add(key)
            found.append(key)
    for match in PDF_CITE_RE.finditer(text):
        key = match.group(1)
        if key not in seen:
            seen.add(key)
            found.append(key)
    if not found:
        for match in NAKED_CITE_RE.finditer(text):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                found.append(key)
    return found


def arxiv_abs_url(arxiv_id: str) -> str | None:
    if arxiv_id.startswith("pdf:"):
        return None
    return f"https://arxiv.org/abs/{arxiv_id}"


def linkify_citations(text: str) -> str:
    """Turn [arXiv id] mentions into Markdown links for the chat UI."""

    def _repl(match: re.Match[str]) -> str:
        arxiv_id = match.group(1)
        return f"[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"

    return BRACKET_CITE_RE.sub(_repl, text)


def sources_from_chunks(
    chunks: list[RetrievedChunk],
    cited_ids: list[str] | None = None,
) -> list[dict]:
    """Build a de-duplicated source list, cited papers first when ids are given."""
    cited = set(cited_ids or [])
    by_id: dict[str, dict] = {}
    for chunk in chunks:
        arxiv_id = chunk.arxiv_id
        if arxiv_id in by_id:
            continue
        entry = chunk.to_dict()
        entry["cited"] = arxiv_id in cited or any(
            arxiv_id.endswith(cid) or cid.endswith(arxiv_id) or cid == arxiv_id
            for cid in cited
        )
        entry["url"] = arxiv_abs_url(arxiv_id)
        by_id[arxiv_id] = entry
    if not cited:
        return list(by_id.values())
    cited_rows = [row for row in by_id.values() if row.get("cited")]
    other_rows = [row for row in by_id.values() if not row.get("cited")]
    return cited_rows + other_rows
