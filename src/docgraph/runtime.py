"""Wires configuration, index, chat client, retriever and answerer together."""

from __future__ import annotations

from typing import Mapping, Sequence

from .answer import Answerer
from .config import Config
from .embeddings import Embedder, make_embedder
from .llm import ChatClient
from .rerank import FlashScorer
from .retrieval import Retriever
from .store import Index


class LazyReranker:
    """Loads openjev-flash on first use, so nothing heavy is imported unless reranking is requested."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._scorer: FlashScorer | None = None

    @property
    def scorer(self) -> FlashScorer:
        if self._scorer is None:
            self._scorer = FlashScorer(self.cfg)
        return self._scorer

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        return self.scorer.score(question, passages)


class Runtime:
    def __init__(self, cfg: Config, *, embedder: Embedder | None = None, client: ChatClient | None = None,
                 env: Mapping[str, str] | None = None):
        self.cfg = cfg
        self.env = env
        self.embedder: Embedder = embedder or make_embedder(cfg.embeddings, env)
        self.index = Index.open(cfg, self.embedder)  # raises EmbeddingMismatch if the index was built with another model
        self.client = client or ChatClient(
            cfg.chat.endpoints, timeout=cfg.chat.timeout, max_tokens=cfg.chat.max_tokens,
            passes=cfg.chat.passes, backoff=cfg.chat.backoff, env=env,
        )
        self.reranker = LazyReranker(cfg.rerank)
        self.retriever = Retriever(self.index, cfg.retrieval, self.reranker)
        self.answerer = Answerer(self.index, self.retriever, self.client, cfg)

    def close(self) -> None:
        self.index.close()
        self.client.close()
