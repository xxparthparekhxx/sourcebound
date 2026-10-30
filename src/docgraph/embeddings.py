"""Embedding backends: a small local CPU model (fastembed) or an OpenAI-compatible /v1/embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import httpx
import numpy as np

from .config import EmbeddingsConfig


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    provider: str
    model: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


class LocalEmbedder:
    """fastembed (ONNX runtime, CPU). No torch."""

    provider = "local"

    def __init__(self, model: str, *, batch_size: int = 32, cache_dir: str | Path | None = None):
        from fastembed import TextEmbedding  # imported lazily: it is slow to import

        supported = {m["model"]: m for m in TextEmbedding.list_supported_models()}
        if model not in supported:
            raise EmbeddingError(
                f"{model!r} is not a fastembed model. Pick one from fastembed's list "
                "(for example BAAI/bge-small-en-v1.5) or set embeddings.provider = \"openai\"."
            )
        self.model = model
        self.batch_size = batch_size
        self._dim = int(supported[model]["dim"])
        cache = Path(cache_dir or os.environ.get("FASTEMBED_CACHE_PATH") or Path.home() / ".cache" / "docgraph" / "fastembed")
        cache.mkdir(parents=True, exist_ok=True)
        self._te = TextEmbedding(model_name=model, cache_dir=str(cache))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return _normalize(np.array(list(self._te.embed(list(texts), batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._te.query_embed([text]))))[0]


class OpenAIEmbedder:
    """Any server that implements POST {base_url}/embeddings (OpenAI, Ollama, vLLM, ...)."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key_env: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout: float = 120.0,
        env: Mapping[str, str] | None = None,
        http: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.env = env
        self._http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self._request(["dimension probe"]).shape[1])
        return self._dim

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = (self.env if self.env is not None else os.environ).get(self.api_key_env)
            if not key:
                raise EmbeddingError(f"environment variable {self.api_key_env} is not set")
            h["Authorization"] = f"Bearer {key}"
        return h

    def _request(self, inputs: list[str]) -> np.ndarray:
        try:
            resp = self._http.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": inputs},
                headers=self._headers(),
            )
        except httpx.TransportError as e:
            raise EmbeddingError(f"embedding endpoint {self.base_url} unreachable: {e}") from e
        if resp.status_code != 200:
            raise EmbeddingError(f"embedding endpoint returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            rows = sorted(resp.json()["data"], key=lambda r: r.get("index", 0))
            vecs = np.array([r["embedding"] for r in rows], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as e:
            raise EmbeddingError(f"unexpected embeddings response: {resp.text[:200]}") from e
        if len(vecs) != len(inputs):
            raise EmbeddingError(f"asked for {len(inputs)} embeddings, got {len(vecs)}")
        return _normalize(vecs)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = [self.document_prefix + t for t in texts]
        out = [self._request(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        if not out:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self._request([self.query_prefix + text])[0]


def make_embedder(cfg: EmbeddingsConfig, env: Mapping[str, str] | None = None) -> Embedder:
    if cfg.provider == "local":
        return LocalEmbedder(cfg.model, batch_size=cfg.batch_size)
    assert cfg.base_url is not None
    return OpenAIEmbedder(
        cfg.model,
        cfg.base_url,
        api_key_env=cfg.api_key_env,
        query_prefix=cfg.query_prefix,
        document_prefix=cfg.document_prefix,
        batch_size=cfg.batch_size,
        timeout=cfg.timeout,
        env=env,
    )
