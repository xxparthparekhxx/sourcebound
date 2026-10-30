"""File loaders: PDF, Markdown, plain text and HTML -> body paragraphs tagged with heading and page."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from ..textutil import normalize_ws
from .chunking import Block

SUPPORTED = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".html": "html",
    ".htm": "html",
}

OcrFn = Callable[[bytes], str]


class UnsupportedFormat(ValueError):
    pass


@dataclass
class SkippedPage:
    page: int
    reason: str


@dataclass
class ParsedDocument:
    title: str
    format: str
    blocks: list[Block]
    n_pages: int | None = None
    skipped_pages: list[SkippedPage] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)


def detect_format(path: Path) -> str:
    fmt = SUPPORTED.get(path.suffix.lower())
    if not fmt:
        raise UnsupportedFormat(f"unsupported file type {path.suffix or '(none)'}: {path.name}")
    return fmt


def load_file(path: Path, *, ocr: OcrFn | None = None, min_chars: int = 20, ocr_dpi: int = 150) -> ParsedDocument:
    fmt = detect_format(path)
    if fmt == "pdf":
        return parse_pdf(path, ocr=ocr, min_chars=min_chars, dpi=ocr_dpi)
    text = _read_text(path)
    if fmt == "markdown":
        return parse_markdown(text, default_title=path.stem)
    if fmt == "html":
        return parse_html(text, default_title=path.stem)
    return parse_text(text, default_title=path.stem)


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------------------- Markdown

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MD_INLINE_RE = re.compile(r"[*_`]+")


def markdown_blocks(text: str, *, page: int | None = None, start_section: int = 0, heading: str = "") -> tuple[list[Block], str | None, int, str]:
    """Parse Markdown into paragraphs. Returns (blocks, first_h1, last_section, last_heading)."""
    blocks: list[Block] = []
    first_h1: str | None = None
    section = start_section
    para: list[str] = []
    in_fence = False
    fence: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Block("\n".join(para).strip(), page, heading, section))
            para.clear()

    for line in text.splitlines():
        if in_fence:
            fence.append(line)
            if _FENCE_RE.match(line):
                in_fence = False
                blocks.append(Block("\n".join(fence).strip(), page, heading, section))
                fence = []
            continue
        if _FENCE_RE.match(line):
            flush()
            in_fence = True
            fence = [line]
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush()
            heading = _MD_INLINE_RE.sub("", m.group(2)).strip()
            section += 1
            if first_h1 is None and len(m.group(1)) == 1:
                first_h1 = heading
            continue
        if not line.strip():
            flush()
            continue
        para.append(line.rstrip())
    flush()
    if fence:
        blocks.append(Block("\n".join(fence).strip(), page, heading, section))
    return blocks, first_h1, section, heading


def parse_markdown(text: str, *, default_title: str = "") -> ParsedDocument:
    title = None
    m = _FRONT_MATTER_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip("\"'")
        text = text[m.end():]
    blocks, h1, _, _ = markdown_blocks(text)
    return ParsedDocument(title or h1 or default_title, "markdown", blocks)


# ---------------------------------------------------------------------------- plain text


def parse_text(text: str, *, default_title: str = "") -> ParsedDocument:
    blocks: list[Block] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = normalize_ws(para)
        if para:
            blocks.append(Block(para))
    return ParsedDocument(default_title, "text", blocks)


# ---------------------------------------------------------------------------- HTML

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "head", "nav", "footer", "aside", "form"}
_BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "tr", "table", "pre", "blockquote", "section", "article", "main",
    "dd", "dt", "dl", "figure", "figcaption", "br", "hr", "header", "details", "summary",
}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _HTMLBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title = ""
        self.first_h1: str | None = None
        self._buf: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_pre = 0
        self._heading_tag: str | None = None
        self._heading = ""
        self._section = 0

    def _flush(self) -> None:
        text = "".join(self._buf)
        self._buf.clear()
        text = text.strip("\n") if self._in_pre else normalize_ws(text)
        if text.strip():
            self.blocks.append(Block(text, None, self._heading, self._section))

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in _SKIP_TAGS and tag != "head":
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADINGS:
            self._flush()
            self._heading_tag = tag
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self._in_pre += 1
        elif tag in ("td", "th") and self._buf:
            self._buf.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in _HEADINGS and self._heading_tag == tag:
            text = normalize_ws("".join(self._buf))
            self._buf.clear()
            self._heading_tag = None
            if text:
                self._heading = text
                self._section += 1
                if tag == "h1" and self.first_h1 is None:
                    self.first_h1 = text
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self._in_pre = max(0, self._in_pre - 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def parse_html(text: str, *, default_title: str = "") -> ParsedDocument:
    p = _HTMLBlocks()
    p.feed(text)
    p.close()
    title = normalize_ws(p.title) or p.first_h1 or default_title
    return ParsedDocument(title, "html", p.blocks)


# ---------------------------------------------------------------------------- PDF

_HYPHEN_END = re.compile(r"[A-Za-z]-$")


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif _HYPHEN_END.search(out) and ln[:1].islower():
            out = out[:-1] + ln  # de-hyphenate a word broken across lines
        else:
            out += " " + ln
    return out


@dataclass
class _PdfBlock:
    text: str
    size: float
    bold: float  # fraction of characters in a bold face
    n_lines: int
    y0: float
    y1: float


def _pdf_blocks(page) -> list[_PdfBlock]:
    out: list[_PdfBlock] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        lines, chars, size_sum, bold_chars = [], 0, 0.0, 0
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            text = "".join(s.get("text", "") for s in spans)
            if not text.strip():
                continue
            lines.append(text)
            for s in spans:
                n = len(s.get("text", "").strip())
                chars += n
                size_sum += s.get("size", 0.0) * n
                if (s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower():
                    bold_chars += n
        if not lines or chars == 0:
            continue
        x0, y0, x1, y1 = b["bbox"]
        out.append(_PdfBlock(_join_lines(lines), size_sum / chars, bold_chars / chars, len(lines), y0, y1))
    return out


def _is_heading(b: _PdfBlock, body: float) -> bool:
    t = b.text
    if len(t) > 140 or b.n_lines > 3:
        return False
    if b.size >= body * 1.15:
        return True
    return b.bold >= 0.9 and len(t) <= 90 and not t.rstrip().endswith((".", ",", ";")) and b.size >= body * 0.95


def parse_pdf(path: Path, *, ocr: OcrFn | None = None, min_chars: int = 20, dpi: int = 150) -> ParsedDocument:
    import pymupdf

    doc = pymupdf.open(path)
    try:
        pages = [_pdf_blocks(p) for p in doc]
        # Which pages have no usable text layer?
        no_text: list[int] = []
        for i, p in enumerate(doc):
            if sum(len(b.text) for b in pages[i]) < min_chars:
                no_text.append(i)

        # Body font size: the most common size, weighted by character count.
        sizes: Counter[float] = Counter()
        for blocks in pages:
            for b in blocks:
                sizes[round(b.size * 2) / 2] += len(b.text)
        body = sizes.most_common(1)[0][0] if sizes else 10.0

        # Running headers / footers: short text repeating in the page margins on many pages.
        margin: Counter[str] = Counter()
        for i, blocks in enumerate(pages):
            h = doc[i].rect.height
            for b in blocks:
                if (b.y1 < h * 0.09 or b.y0 > h * 0.91) and len(b.text) < 120:
                    margin[re.sub(r"\d+", "#", b.text.strip().lower())] += 1
        repeated = {k for k, n in margin.items() if n >= max(3, 0.5 * len(pages))}

        parsed = ParsedDocument("", "pdf", [], n_pages=len(doc))
        title_guess = (doc.metadata or {}).get("title", "").strip()
        section, heading = 0, ""
        first_big: str | None = None

        for i, blocks in enumerate(pages):
            page_no = i + 1
            if i in no_text:
                has_images = bool(doc[i].get_images(full=True)) or bool(doc[i].get_drawings())
                if not has_images:
                    continue  # a blank page is not worth reporting
                if ocr is None:
                    parsed.skipped_pages.append(SkippedPage(page_no, "no text layer (scanned page); OCR is off"))
                    continue
                png = doc[i].get_pixmap(dpi=dpi).tobytes("png")
                try:
                    text = ocr(png)
                except Exception as e:  # noqa: BLE001  - one bad page must not abort the document
                    parsed.skipped_pages.append(SkippedPage(page_no, f"OCR failed: {type(e).__name__}: {e}"[:200]))
                    continue
                if not text.strip():
                    parsed.skipped_pages.append(SkippedPage(page_no, "OCR returned no text"))
                    continue
                new, _, section, heading = markdown_blocks(text, page=page_no, start_section=section, heading=heading)
                parsed.blocks.extend(new)
                parsed.ocr_pages.append(page_no)
                continue
            for b in blocks:
                key = re.sub(r"\d+", "#", b.text.strip().lower())
                if key in repeated or re.fullmatch(r"[\divxlc\s.\-–]{1,8}", b.text.strip().lower()):
                    continue  # running header/footer or a bare page number
                if _is_heading(b, body):
                    heading = b.text.strip()
                    section += 1
                    if first_big is None and b.size >= body * 1.3:
                        first_big = heading
                    continue
                parsed.blocks.append(Block(b.text, page_no, heading, section))
        parsed.title = title_guess or first_big or path.stem
        return parsed
    finally:
        doc.close()
