"""openjev-flash as a reranker and a faithfulness scorer.

openjev-flash (https://huggingface.co/xxparthparekhxx/openjev-flash) is a 150M-parameter
cross-encoder that reads a premise and a hypothesis and returns P(contradiction / entailment /
neutral). This follows the model card's recipe directly (plain tokenizer pair, truncation of the
premise only, probabilities reordered to [contradiction, entailment, neutral]) instead of
importing the repository's ``openjev_flash.py`` at runtime, so no downloaded code is executed.

Needs the optional dependencies: ``uv sync --extra rerank``.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .config import RerankConfig

CON, ENT, NEU = 0, 1, 2


class RerankUnavailable(RuntimeError):
    pass


class FlashScorer:
    def __init__(self, cfg: RerankConfig | None = None):
        self.cfg = cfg or RerankConfig()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RerankUnavailable("reranking needs the optional dependencies: uv sync --extra rerank") from e
        self._torch = torch
        device = self.cfg.device
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model, dtype=dtype).to(device).eval()
        label_to_idx = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self._perm = [label_to_idx["contradiction"], label_to_idx["entailment"], label_to_idx["neutral"]]
        self.device = device

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Softmax probabilities [contradiction, entailment, neutral] per (premise, hypothesis)."""
        torch = self._torch
        order = np.argsort([len(p) + len(h) for p, h in pairs], kind="stable")  # similar lengths per batch
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        bs = self.cfg.batch_size
        with torch.inference_mode():
            for i in range(0, len(pairs), bs):
                idx = order[i : i + bs]
                enc = self.tok(
                    [pairs[j][0].strip() for j in idx],
                    [pairs[j][1].strip() for j in idx],
                    truncation="only_first",
                    max_length=self.cfg.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()[:, self._perm]
                out[idx] = torch.softmax(logits, -1).cpu().numpy()
        return out

    # -- reranking -------------------------------------------------------------------------------

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        """P(entailment) that each passage answers the question."""
        hyp = self.cfg.hypothesis.format(question=question.strip())
        return self.predict([(p, hyp) for p in passages])[:, ENT].tolist()

    # -- faithfulness ----------------------------------------------------------------------------

    def entailment(self, premise: str, hypothesis: str) -> float:
        return float(self.predict([(premise, hypothesis)])[0, ENT])


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n+", text.strip()):
        parts.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return parts
