"""Per-sentence faithfulness: is each cited sentence entailed by the passages it cites?

Every answer sentence that carries a ``[n]`` marker naming a passage that was shown to the model is
scored with openjev-flash against the text of the passages it cites (premise) with the markers removed
(hypothesis). P(entailment) >= the configured threshold counts as supported. Sentences without a valid
marker, or too short to judge, are reported as not scored.
"""

from __future__ import annotations

import re
from typing import Mapping

from .rerank import split_sentences

_MARK = re.compile(r"\[(\d+(?:\s*[,;-]\s*\d+)*)\]")
MIN_SENTENCE_CHARS = 8


def marker_numbers(sentence: str) -> list[int]:
    nums: list[int] = []
    for m in _MARK.finditer(sentence):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            if "-" in part:
                a, b = part.split("-", 1)
                nums.extend(range(int(a), int(b) + 1))
            else:
                nums.append(int(part))
    return nums


def strip_markers(sentence: str) -> str:
    return re.sub(r"\s+", " ", _MARK.sub("", sentence)).strip()


def sentence_pairs(answer: str, passages: Mapping[int, str]) -> list[dict]:
    """Split an answer into sentences and attach the cited passage numbers and the (premise, hypothesis) pair."""
    out = []
    for sent in split_sentences(answer):
        nums = [n for n in dict.fromkeys(marker_numbers(sent)) if n in passages]
        hyp = strip_markers(sent)
        pair = ("\n\n".join(passages[n] for n in nums), hyp) if nums and len(hyp) >= MIN_SENTENCE_CHARS else None
        out.append({"text": sent, "cited": nums, "pair": pair})
    return out


def check_answer(scorer, answer: str, passages: Mapping[int, str], threshold: float = 0.5) -> list[dict]:
    """One dict per sentence: text, cited, p_entail / supported (None when the sentence was not scored)."""
    items = sentence_pairs(answer, passages)
    scored = [(i, it["pair"]) for i, it in enumerate(items) if it["pair"]]
    probs = scorer.predict([p for _, p in scored]) if scored else []
    result = {i: p for (i, _), p in zip(scored, probs)}
    out = []
    for i, it in enumerate(items):
        p = result.get(i)
        out.append({
            "text": it["text"], "cited": it["cited"],
            "p_entail": None if p is None else round(float(p[1]), 4),
            "supported": None if p is None else bool(p[1] >= threshold),
        })
    return out
