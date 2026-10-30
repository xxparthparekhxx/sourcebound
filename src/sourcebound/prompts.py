"""Prompt text and passage formatting."""

from __future__ import annotations

from typing import Sequence

from .models import Passage

PIPELINE_SYSTEM = """You answer questions using only the numbered passages you are given.

Rules:
- Base every statement on the passages. Do not use outside knowledge.
- After each claim, cite the passage that supports it with its number in square brackets, like [1] or [2][3]. Use only numbers that appear in the passages.
- If the passages do not contain the answer, say that you could not find it in the documents and say what is missing. Do not guess.
- Be concise and write plain text."""

AGENT_SYSTEM = """You answer questions about a set of documents. You have tools to look things up.

Tools:
- search_documents(query): finds passages and numbers them.
- read_passage(passage): shows the full text of one numbered passage.
- related_entities(entity): shows entities linked to an entity in the knowledge graph, with passages that mention them.

How to work:
- Search first. If the results do not answer the question, search again with different words. You have at most {max_steps} rounds of tool calls.
- Do not write any text before or between tool calls. Write the answer only once you have the evidence.
- Answer only from what the tools returned. Cite each claim with the passage number in square brackets, like [1] or [2][3]. Never cite a number the tools did not return.
- If the documents do not contain the answer, say so plainly. Do not guess."""

AGENT_FORCE_ANSWER = (
    "The tool budget is used up. Answer the question now, using only the passages found so far "
    "and citing them as [n]. If they do not contain the answer, say so."
)


def format_passage(n: int, p: Passage, max_chars: int | None = None) -> str:
    text = p.text
    if max_chars and len(text) > max_chars:  # never silent: the cut is marked in the prompt
        text = text[:max_chars].rsplit(" ", 1)[0] + " ...[truncated]"
    head = f"[{n}] Source: {p.label()}"
    return f"{head}\n{text}"


def pipeline_user_message(question: str, passages: Sequence[Passage], max_chars: int | None = None) -> str:
    body = "\n\n".join(format_passage(i, p, max_chars) for i, p in enumerate(passages, start=1))
    return f"Passages:\n\n{body}\n\nQuestion: {question}"
