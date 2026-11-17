"""Prompt templates for grounded, citation-required answers."""

from __future__ import annotations

from arxiv_rag.retrievers.base import RetrievedChunk

SYSTEM_PROMPT = """You are a research assistant for academic papers in artificial intelligence, natural language processing, and machine learning.

Answer the user's question using ONLY the provided context excerpts from arXiv papers.
Rules:
- Cite supporting sources with the arXiv id in square brackets, for example [1706.03762].
- If several sources support a claim, cite all of them.
- If the context is insufficient to answer, say so clearly. Do not guess and do not use knowledge that is not in the context.
- Do not invent papers, results, or citations.
- Be concise, precise, and technical when the context allows it.
- Prefer English unless the user writes in another language."""

REFUSAL_MESSAGE = (
    "I do not have enough relevant context in the indexed papers to answer that "
    "question reliably. Please rephrase the question, broaden the year or "
    "category filters, or try a topic covered by cs.AI, cs.CL, or cs.LG abstracts."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        year = chunk.metadata.get("year") or "n.d."
        title = chunk.metadata.get("title") or "Untitled"
        arxiv_id = chunk.arxiv_id
        blocks.append(
            f"[{i}] arXiv:{arxiv_id} | {title} ({year})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = format_context(chunks)
    user = (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Write an answer grounded in the context and include arXiv id citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
