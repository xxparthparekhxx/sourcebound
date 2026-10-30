"""Structure-aware chunking.

Input is a list of body paragraphs, each already tagged with the heading it sits under, a
section number (incremented at every heading) and, for PDFs, a page number. Chunks are packed
from whole paragraphs up to ``size`` characters, never cross a heading, and each chunk after
the first in a section starts with a short tail of the previous one (``overlap`` characters).
A paragraph longer than ``size`` is split at sentence boundaries first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
MIN_CHUNK_CHARS = 20


@dataclass
class Block:
    """One body paragraph."""

    text: str
    page: int | None = None
    heading: str = ""
    section: int = 0


@dataclass
class ChunkSpec:
    text: str
    page: int | None
    page_end: int | None
    heading: str


def split_long(text: str, size: int) -> list[str]:
    """Split text into pieces of at most `size` characters, preferring sentence boundaries."""
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    cur = ""
    for sent in _SENT_SPLIT.split(text):
        while len(sent) > size:  # a single sentence longer than the limit: cut at whitespace
            cut = sent.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            head, sent = sent[:cut].rstrip(), sent[cut:].lstrip()
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(head)
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) > size:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        pieces.append(cur)
    return pieces


def overlap_tail(text: str, overlap: int) -> str:
    """The last <= `overlap` characters of `text`, starting at a sentence or word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 and len(text) <= overlap else ""
    tail = text[-overlap:]
    m = _SENT_SPLIT.search(tail)
    if m:
        return tail[m.end():].strip()
    sp = tail.find(" ")
    return tail[sp + 1:].strip() if sp != -1 else ""


@dataclass
class _Piece:
    text: str
    page: int | None


def chunk_blocks(blocks: list[Block], size: int = 1200, overlap: int = 200) -> list[ChunkSpec]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[ChunkSpec] = []

    # group by section (a heading change always starts a new group)
    groups: list[list[Block]] = []
    for b in blocks:
        if not b.text.strip():
            continue
        if not groups or groups[-1][0].section != b.section or groups[-1][0].heading != b.heading:
            groups.append([])
        groups[-1].append(b)

    for group in groups:
        heading = group[0].heading
        pieces: list[_Piece] = []
        for b in group:
            for part in split_long(b.text.strip(), size):
                pieces.append(_Piece(part, b.page))

        cur: list[_Piece] = []  # pieces in the chunk being built (overlap seed first, if any)
        new_content = False

        def flush() -> None:
            nonlocal cur, new_content
            if cur and new_content:
                text = "\n\n".join(p.text for p in cur)
                if len(text.strip()) >= MIN_CHUNK_CHARS:
                    pages = [p.page for p in cur if p.page is not None]
                    chunks.append(
                        ChunkSpec(text, min(pages) if pages else None, max(pages) if pages else None, heading)
                    )
            tail = overlap_tail(cur[-1].text, overlap) if cur and new_content else ""
            cur = [_Piece(tail, cur[-1].page)] if tail else []
            new_content = False

        for piece in pieces:
            length = sum(len(p.text) for p in cur) + 2 * len(cur)
            if cur and new_content and length + len(piece.text) > size:
                flush()
                # the seed may push the new piece over the limit: drop it rather than exceed `size`
                if cur and sum(len(p.text) for p in cur) + 2 + len(piece.text) > size:
                    cur = []
            cur.append(piece)
            new_content = True
        flush()
    return chunks
