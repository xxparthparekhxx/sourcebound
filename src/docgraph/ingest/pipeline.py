"""Ingest orchestration: files -> parsed -> chunked -> stored (and optionally graph-extracted)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Config
from ..graph import GraphStats, extract_graph
from ..llm import ChatClient
from ..store import Index
from .chunking import chunk_blocks
from .loaders import UnsupportedFormat, detect_format, load_file
from .ocr import make_ocr

log = logging.getLogger("docgraph.ingest")


@dataclass
class DocReport:
    path: str
    status: str  # ingested | replaced | unchanged | failed | unsupported
    title: str = ""
    n_chunks: int = 0
    n_pages: int | None = None
    skipped_pages: list[dict] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)
    error: str = ""


@dataclass
class IngestReport:
    documents: list[DocReport] = field(default_factory=list)
    graph: GraphStats | None = None

    def count(self, status: str) -> int:
        return sum(1 for d in self.documents if d.status == status)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_files(path: Path) -> list[Path]:
    """A file, or every supported file under a directory (hidden directories are skipped)."""
    if path.is_file():
        return [path]
    out = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts):
            out.append(p)
    return out


def ingest_path(
    index: Index,
    path: str | Path,
    cfg: Config,
    client: ChatClient | None = None,
    *,
    graph: bool | None = None,
    ocr: bool | None = None,
    force: bool = False,
    progress: Callable[[DocReport], None] | None = None,
) -> IngestReport:
    """Ingest a file or directory. Unchanged files (same SHA-256) are skipped; a changed file
    replaces the chunks, vectors and graph rows of its earlier version."""
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"no such file or directory: {root}")
    use_graph = cfg.ingest.graph.enabled if graph is None else graph
    use_ocr = cfg.ingest.ocr.enabled if ocr is None else ocr
    if (use_graph or use_ocr) and client is None:
        raise ValueError("graph extraction and OCR need a chat client")
    ocr_fn = make_ocr(client, cfg.ingest.ocr) if use_ocr and client else None

    report = IngestReport()
    new_doc_ids: list[int] = []
    explicit = root.is_file()
    for f in find_files(root):
        src = str(f.resolve())
        try:
            detect_format(f)
        except UnsupportedFormat as e:
            if explicit:  # a file the user named explicitly deserves an error; directory noise does not
                report.documents.append(DocReport(src, "unsupported", error=str(e)))
            continue
        rep = DocReport(src, "failed")
        try:
            sha = sha256_file(f)
            existing = index.get_document(src)
            if existing is not None and existing.sha256 == sha and not force:
                rep = DocReport(src, "unchanged", existing.title, existing.n_chunks, existing.n_pages, existing.skipped_pages)
                if graph is not False and use_graph:
                    new_doc_ids.append(existing.id)  # pick up chunks that still lack a graph
            else:
                parsed = load_file(f, ocr=ocr_fn, min_chars=cfg.ingest.ocr.min_chars, ocr_dpi=cfg.ingest.ocr.dpi)
                chunks = chunk_blocks(parsed.blocks, cfg.ingest.chunk_size, cfg.ingest.chunk_overlap)
                doc_id, _, replaced = index.upsert_document(
                    src, parsed, chunks, sha, context_header=cfg.ingest.context_header
                )
                new_doc_ids.append(doc_id)
                rep = DocReport(
                    src, "replaced" if replaced else "ingested", parsed.title, len(chunks), parsed.n_pages,
                    [{"page": s.page, "reason": s.reason} for s in parsed.skipped_pages], parsed.ocr_pages,
                )
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a directory ingest
            log.exception("failed to ingest %s", f)
            rep = DocReport(src, "failed", error=f"{type(e).__name__}: {e}")
        report.documents.append(rep)
        if progress:
            progress(rep)

    if use_graph and new_doc_ids and client is not None:
        report.graph = extract_graph(index, client, cfg.ingest.graph, new_doc_ids)
    return report
