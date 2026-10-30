"""Plain data classes shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Passage:
    """A chunk of a document as returned by retrieval."""

    id: int
    doc_id: int
    document: str  # document title
    source: str  # path the file was ingested from
    page: int | None
    page_end: int | None
    heading: str
    text: str
    ord: int = 0
    score: float = 0.0
    found_by: list[str] = field(default_factory=list)  # which stages surfaced it: vector / bm25 / graph
    vector_score: float | None = None  # cosine similarity to the query, if the vector stage saw it
    rerank_score: float | None = None

    @property
    def location(self) -> str:
        if self.page is None:
            return ""
        if self.page_end and self.page_end != self.page:
            return f"pp. {self.page}-{self.page_end}"
        return f"p. {self.page}"

    @property
    def filename(self) -> str:
        return Path(self.source).name

    def label(self) -> str:
        """Short human-readable origin: 'paper.pdf, p. 3, Methods'."""
        parts = [self.filename or self.document]
        if self.location:
            parts.append(self.location)
        if self.heading:
            parts.append(self.heading)
        return ", ".join(parts)

    def to_dict(self, *, text: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "doc_id": self.doc_id,
            "document": self.document,
            "file": self.filename,
            "page": self.page,
            "page_end": self.page_end,
            "heading": self.heading,
        }
        if text:
            d["text"] = self.text
        return d


@dataclass
class Entity:
    id: int
    name: str
    type: str
    mentions: int = 0
