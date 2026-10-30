"""Shared helpers for the Qasper evaluation: gold-evidence matching, metrics, result files."""

from __future__ import annotations

import datetime as dt
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_DATA = ROOT / "data" / "qasper"
DEFAULT_WORK = ROOT / "work"

NGRAM = 5
COVER_THRESHOLD = 0.5


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngram_set(toks: Sequence[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


class Evidence:
    """One gold evidence paragraph, prepared for overlap matching."""

    def __init__(self, text: str):
        self.text = text
        self.toks = tokens(text)
        self.grams = ngram_set(self.toks)
        self.flat = " " + " ".join(self.toks) + " "

    def covered_by(self, chunk: "ChunkText") -> bool:
        """A chunk covers an evidence paragraph when at least half of the paragraph's word 5-grams
        occur in the chunk (very short paragraphs: the whole paragraph must occur in the chunk)."""
        if not self.toks:
            return False
        if self.grams:
            return len(self.grams & chunk.grams) / len(self.grams) >= COVER_THRESHOLD
        return self.flat in chunk.flat


class ChunkText:
    def __init__(self, text: str):
        toks = tokens(text)
        self.grams = ngram_set(toks)
        self.flat = " " + " ".join(toks) + " "


def question_metrics(
    ranked: Sequence[ChunkText], annotators: Sequence[Sequence[Evidence]], ks: Iterable[int] = (5, 10)
) -> dict:
    """Retrieval metrics for one question.

    recall@k   = max over annotators of the fraction of that annotator's evidence paragraphs that
                 some chunk in the top k covers (Qasper's own convention is also max over annotators)
    rr         = 1 / rank of the first chunk that covers any annotator's evidence (0 if none in the list)
    """
    out: dict = {}
    cover = [[[ev.covered_by(c) for c in ranked] for ev in ann] for ann in annotators]  # [ann][ev][rank]
    for k in ks:
        best = 0.0
        for ann in cover:
            if ann:
                best = max(best, sum(any(flags[:k]) for flags in ann) / len(ann))
        out[f"recall@{k}"] = best
    first = None
    for rank in range(len(ranked)):
        if any(flags[rank] for ann in cover for flags in ann):
            first = rank + 1
            break
    out["rr"] = 1.0 / first if first else 0.0
    out["first_hit_rank"] = first
    return out


def bootstrap_ci(values: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): mean and a 95% percentile bootstrap interval over questions."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def environment() -> dict:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=ROOT).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": sh("git", "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain", "--", "src", "eval")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT.parent) if path.is_relative_to(ROOT.parent) else path}")


def load_questions(data: Path, name: str = "questions.json") -> dict:
    return json.loads((data / name).read_text())


def sample(items: Sequence, n: int, seed: int) -> list:
    items = list(items)
    random.Random(seed).shuffle(items)
    return items[:n]
