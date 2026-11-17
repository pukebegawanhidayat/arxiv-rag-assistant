"""Streamlit chat UI for the arXiv RAG Assistant. Talks to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_URL") or os.environ.get("ARXIV_RAG_API_URL") or "http://localhost:8000"
CATEGORIES = ["(all)", "cs.AI", "cs.CL", "cs.LG"]
SOURCES = {
    "arXiv index only": "arxiv",
    "Uploaded PDF only": "pdf",
    "arXiv + uploaded PDF": "both",
}


def api_url() -> str:
    return st.session_state.get("api_url") or DEFAULT_API


def request(method: str, path: str, **kwargs) -> httpx.Response:
    url = api_url().rstrip("/") + path
    timeout = kwargs.pop("timeout", 180.0)
    with httpx.Client(timeout=timeout) as client:
        return client.request(method, url, **kwargs)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("pdf_name", None)
    st.session_state.setdefault("pdf_chunks", 0)
    st.session_state.setdefault("api_url", DEFAULT_API)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources returned.")
        return
    for src in sources:
        arxiv_id = src.get("arxiv_id") or ""
        title = src.get("title") or "Untitled"
        year = src.get("year") or "n.d."
        url = src.get("url")
        cited = "cited" if src.get("cited") else "context"
        header = f"[{arxiv_id}] {title} ({year}) — {cited}"
        with st.expander(header, expanded=bool(src.get("cited"))):
            if url:
                st.markdown(f"[Open on arXiv]({url})")
            cats = src.get("categories") or []
            if cats:
                st.caption("Categories: " + ", ".join(cats))
            excerpt = src.get("text") or ""
            if excerpt:
                st.write(excerpt)


def main() -> None:
    st.set_page_config(page_title="arXiv RAG Assistant", page_icon="📄", layout="wide")
    init_state()

    st.title("arXiv RAG Assistant")
    st.caption("Domain-specific retrieval-augmented assistant for AI, ML, and NLP papers.")

    with st.sidebar:
        st.header("Settings")
        st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
        if st.button("Check API health"):
            try:
                res = request("GET", "/health", timeout=5.0)
                res.raise_for_status()
                health = res.json()
                st.success(
                    f"status={health.get('status')} · index={health.get('n_chunks')} chunks · "
                    f"ollama={'yes' if health.get('ollama') else 'no'}"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Cannot reach API: {exc}")

        st.subheader("Filters")
        year_range = st.slider("Publication year", 1990, 2026, (2015, 2026))
        apply_year = st.checkbox("Apply year filter", value=False)
        category = st.selectbox("Category", CATEGORIES, index=0)
        source_label = st.radio("Search in", list(SOURCES.keys()), index=0)
        source = SOURCES[source_label]

        st.subheader("Upload PDF")
        uploaded = st.file_uploader("Index a PDF (full text) for this session", type=["pdf"])
        if uploaded is not None and st.button("Index PDF"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                res = request("POST", "/upload-pdf", files=files, timeout=180.0)
                if res.status_code >= 400:
                    st.error(res.json().get("detail") if res.headers.get("content-type", "").startswith("application/json") else res.text)
                else:
                    payload = res.json()
                    st.session_state.session_id = payload["session_id"]
                    st.session_state.pdf_name = payload["filename"]
                    st.session_state.pdf_chunks = payload["n_chunks"]
                    st.success(f"Indexed {payload['filename']} into {payload['n_chunks']} chunks.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")
        if st.session_state.session_id:
            st.info(
                f"Active PDF: {st.session_state.pdf_name} "
                f"({st.session_state.pdf_chunks} chunks)\n`{st.session_state.session_id}`"
            )
            if st.button("Clear PDF session"):
                st.session_state.session_id = None
                st.session_state.pdf_name = None
                st.session_state.pdf_chunks = 0
                st.rerun()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("refused"):
                st.warning("The assistant refused because the retrieved context was insufficient.")

    prompt = st.chat_input("Ask about AI, ML, or NLP papers…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    body: dict[str, Any] = {"question": prompt, "source": source}
    if apply_year:
        body["year_min"], body["year_max"] = year_range
    if category != "(all)":
        body["category"] = category
    if st.session_state.session_id and source in ("pdf", "both"):
        body["session_id"] = st.session_state.session_id
    elif source in ("pdf", "both") and not st.session_state.session_id:
        with st.chat_message("assistant"):
            st.error("Upload and index a PDF first, or switch the search scope to the arXiv index.")
        return

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating an answer…"):
            try:
                res = request("POST", "/query", json=body, timeout=180.0)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                return
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except Exception:  # noqa: BLE001
                detail = res.text
            st.error(detail)
            return
        payload = res.json()
        answer = payload.get("answer") or ""
        st.markdown(answer)
        sources = payload.get("sources") or []
        if sources:
            st.subheader("Sources")
            render_sources(sources)
        refused = bool(payload.get("refused"))
        if refused:
            st.warning("The assistant refused because the retrieved context was insufficient.")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "refused": refused}
        )


if __name__ == "__main__":
    main()
