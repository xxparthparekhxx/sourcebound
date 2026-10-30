"""Citation checking: every [n] in an answer must name a passage the model was actually given.

A marker that does not (for example [7] when only five passages were provided) is removed from
the text and reported as flagged. The check works on a stream: text is passed through
:class:`CitationFilter` as it arrives, and a marker is held back only until its closing bracket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection

_BRACKET_CHARS = set("0123456789,;- –")
_TOKEN = re.compile(r"^\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?\s*$")
_MAX_MARKER = 24
_AFTER_REMOVED = ".,;:!?)"


@dataclass
class CitationCheck:
    text: str
    used: list[int] = field(default_factory=list)  # valid numbers, in order of first appearance
    flagged: list[int] = field(default_factory=list)  # numbers that were removed


class CitationFilter:
    """Incremental citation validator. ``allowed`` may be a live set that grows while streaming."""

    def __init__(self, allowed: Collection[int]):
        self.allowed = allowed
        self.used: list[int] = []
        self.flagged: list[int] = []
        self._bracket: str | None = None  # text of an open "[..." that might be a citation
        self._ws = ""  # a single space held back so a removed marker does not leave " ."
        self._removed_after_ws = False

    # -- helpers ---------------------------------------------------------------------------------

    def _parse(self, inner: str) -> list[int] | None:
        nums: list[int] = []
        for part in re.split(r"[,;]", inner):
            m = _TOKEN.match(part)
            if not m:
                return None
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            if b < a or b - a > 20:
                return None
            nums.extend(range(a, b + 1))
        return nums or None

    def _resolve(self, bracket: str) -> str:
        """`bracket` is "[...]". Returns the text to emit for it ("" when it is removed)."""
        nums = self._parse(bracket[1:-1])
        if nums is None:
            return bracket  # not a citation ("[ ]", "[,]", ...): ordinary text
        good = [n for n in nums if n in self.allowed]
        bad = [n for n in nums if n not in self.allowed]
        for n in good:
            if n not in self.used:
                self.used.append(n)
        for n in bad:
            if n not in self.flagged:
                self.flagged.append(n)
        if not bad:
            return bracket
        if not good:
            return ""
        return "[" + ", ".join(str(n) for n in dict.fromkeys(good)) + "]"

    # -- streaming -------------------------------------------------------------------------------

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self._bracket is not None:
                if ch == "]":
                    resolved = self._resolve(self._bracket + "]")
                    self._bracket = None
                    if resolved:
                        out.append(self._ws + resolved)
                        self._ws = ""
                    else:
                        self._removed_after_ws = bool(self._ws)
                    continue
                if ch in _BRACKET_CHARS and len(self._bracket) < _MAX_MARKER:
                    self._bracket += ch
                    continue
                # not a citation after all: the buffered text is ordinary text
                out.append(self._ws + self._bracket)
                self._ws = ""
                self._bracket = None
                # fall through: `ch` is processed normally (it may open a new bracket)
            if ch == "[":
                self._bracket = "["
                continue
            if ch == " ":
                if self._removed_after_ws:
                    continue  # "a [9] b": keep the one space already held, not two
                if self._ws:
                    out.append(self._ws)
                self._ws = " "
                continue
            if self._ws:
                if not (self._removed_after_ws and ch in _AFTER_REMOVED):
                    out.append(self._ws)
                self._ws = ""
            self._removed_after_ws = False
            out.append(ch)
        return "".join(out)

    def flush(self) -> str:
        out = self._ws + (self._bracket or "")
        self._ws, self._bracket, self._removed_after_ws = "", None, False
        return out


def validate_citations(text: str, allowed: Collection[int]) -> CitationCheck:
    """Validate a complete answer: remove markers that name passages that were not provided."""
    f = CitationFilter(allowed)
    cleaned = f.feed(text) + f.flush()
    return CitationCheck(cleaned, f.used, f.flagged)
