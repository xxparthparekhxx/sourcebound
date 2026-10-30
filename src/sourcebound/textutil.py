"""Small text helpers shared by several modules."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN, _CLOSE = "<think>", "</think>"


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (some reasoning models inline them in the content)."""
    text = _THINK_RE.sub("", text)
    # An unterminated block: everything after the opening tag is reasoning.
    i = text.lower().find(_OPEN)
    if i != -1:
        text = text[:i]
    return text.strip()


class ThinkFilter:
    """Streaming version of :func:`strip_think`; tags may be split across chunks."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._inside:
                i = self._buf.lower().find(_CLOSE)
                if i == -1:
                    # keep a tail that could be the start of "</think>"
                    keep = _partial_suffix(self._buf.lower(), _CLOSE)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[i + len(_CLOSE):]
                self._inside = False
                continue
            i = self._buf.lower().find(_OPEN)
            if i == -1:
                keep = _partial_suffix(self._buf.lower(), _OPEN)
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            out.append(self._buf[:i])
            self._buf = self._buf[i + len(_OPEN):]
            self._inside = True
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        self._inside = False
        return rest


def _partial_suffix(s: str, tag: str) -> int:
    """Length of the longest proper prefix of `tag` that `s` ends with."""
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:n]):
            return n
    return 0


_WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


STOPWORDS = frozenset(
    """a an and are as at be but by can did do does for from had has have how i if in into is it its
    me my no not of on or our she so than that the their them then there these they this to us was
    we were what when where which who whom why will with you your about also been being both could
    during each he her his just more most other over some such only own same should too very""".split()
)


def content_terms(text: str) -> list[str]:
    """Lower-cased alphanumeric terms without stop words (used for BM25 queries and the relevance gate)."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", text.lower())
    return [t for t in terms if t not in STOPWORDS and len(t) > 1]
