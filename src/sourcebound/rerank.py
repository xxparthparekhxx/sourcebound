"""openjev-flash as a reranker and a faithfulness scorer.

openjev-flash (https://huggingface.co/xxparthparekhxx/openjev-flash) is a 150M-parameter
cross-encoder that reads a premise and a hypothesis and returns P(contradiction / entailment /
neutral). This follows the model card's recipe directly (plain tokenizer pair, truncation of the
premise only, probabilities reordered to [contradiction, entailment, neutral]) instead of
importing the repository's ``openjev_flash.py`` at runtime, so no downloaded code is executed.

Two backends compute the same thing: ``torch`` (``uv sync --extra rerank``) and ``onnx``
(``uv sync --extra rerank-onnx``), which runs the int8 export of the model under ONNX Runtime and needs no
PyTorch. ``rerank.backend`` in the config picks one; :func:`make_scorer` builds it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import RerankConfig

CON, ENT, NEU = 0, 1, 2


class RerankUnavailable(RuntimeError):
    pass


class _Scorer:
    """What both backends share: batching order, the relevance and entailment helpers."""

    cfg: RerankConfig

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Softmax probabilities [contradiction, entailment, neutral] per (premise, hypothesis)."""
        order = np.argsort([len(p) + len(h) for p, h in pairs], kind="stable")  # similar lengths per batch
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        bs = self.cfg.batch_size
        for i in range(0, len(pairs), bs):
            idx = order[i : i + bs]
            out[idx] = self._probs([pairs[j][0].strip() for j in idx], [pairs[j][1].strip() for j in idx])
        return out

    def _probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        raise NotImplementedError

    # -- reranking -------------------------------------------------------------------------------

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        """P(entailment) that each passage answers the question."""
        hyp = self.cfg.hypothesis.format(question=question.strip())
        return self.predict([(p, hyp) for p in passages])[:, ENT].tolist()

    # -- faithfulness ----------------------------------------------------------------------------

    def entailment(self, premise: str, hypothesis: str) -> float:
        return float(self.predict([(premise, hypothesis)])[0, ENT])


def _permutation(id2label: dict) -> list[int]:
    """Column of each of [contradiction, entailment, neutral] in the model's own logit order."""
    label_to_idx = {str(v).lower(): int(k) for k, v in id2label.items()}
    return [label_to_idx["contradiction"], label_to_idx["entailment"], label_to_idx["neutral"]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


class FlashScorer(_Scorer):
    """The reference implementation: transformers + torch."""

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
        rev = {"revision": self.cfg.revision} if self.cfg.revision else {}
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model, **rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model, dtype=dtype, **rev).to(device).eval()
        self._perm = _permutation(self.model.config.id2label)
        self.device = device

    def _probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            enc = self.tok(
                premises,
                hypotheses,
                truncation="only_first",
                max_length=self.cfg.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.float()[:, self._perm]
            return torch.softmax(logits, -1).cpu().numpy()


class OnnxFlashScorer(_Scorer):
    """The int8 ONNX export of openjev-flash under ONNX Runtime: no torch, no transformers.

    The export lives under ``onnx/`` in the model repository (``model_quantized.onnx`` with its
    ``config.json`` and ``tokenizer.json``). Tokenization is the same as the torch path: the premise
    is truncated, never the hypothesis; probabilities are reordered through the config's id2label.
    ``rerank.onnx_dir`` points at a local copy of that directory instead of downloading it.
    """

    def __init__(self, cfg: RerankConfig | None = None):
        self.cfg = cfg or RerankConfig()
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RerankUnavailable("the onnx backend needs the optional dependencies: uv sync --extra rerank-onnx") from e
        model_path = self._file(Path(self.cfg.onnx_file).name, self.cfg.onnx_file)
        config = json.loads(Path(self._file("config.json")).read_text())
        self._perm = _permutation(config["id2label"])
        self.tok = Tokenizer.from_file(self._file("tokenizer.json"))
        pad = self._pad_token()
        self.tok.no_padding()
        self.tok.enable_padding(pad_id=self.tok.token_to_id(pad), pad_token=pad)
        self.tok.enable_truncation(max_length=self.cfg.max_length, strategy="only_first")
        so = ort.SessionOptions()
        if self.cfg.threads:
            so.intra_op_num_threads = self.cfg.threads
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self._inputs = [i.name for i in self.session.get_inputs()]

    def _file(self, name: str, repo_path: str | None = None) -> str:
        """A local path to `name`: from onnx_dir, or downloaded from the model repository at the pinned revision."""
        if self.cfg.onnx_dir is not None:
            return str(Path(self.cfg.onnx_dir) / name)
        from huggingface_hub import hf_hub_download

        folder = str(Path(self.cfg.onnx_file).parent)
        candidates = [repo_path] if repo_path else [f"{folder}/{name}", name]
        last: Exception | None = None
        for c in candidates:
            try:
                return hf_hub_download(self.cfg.model, c, revision=self.cfg.revision)
            except Exception as e:  # noqa: BLE001 - try the next location
                last = e
        raise RerankUnavailable(f"could not download {name} from {self.cfg.model}: {last}")

    def _pad_token(self) -> str:
        try:
            return json.loads(Path(self._file("tokenizer_config.json")).read_text()).get("pad_token") or "[PAD]"
        except Exception:  # noqa: BLE001 - the config is optional
            return "[PAD]"

    def _probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(list(zip(premises, hypotheses)))
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0].astype(np.float64)
        return _softmax(logits[:, self._perm]).astype(np.float32)


def make_scorer(cfg: RerankConfig | None = None) -> _Scorer:
    cfg = cfg or RerankConfig()
    return OnnxFlashScorer(cfg) if cfg.backend == "onnx" else FlashScorer(cfg)


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    parts = []
    for para in re.split(r"\n+", text.strip()):
        parts.extend(s.strip() for s in _SENT.split(para) if s.strip())
    return parts
