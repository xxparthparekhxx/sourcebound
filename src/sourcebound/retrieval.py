"""Hybrid retrieval: vector (+MMR), BM25, graph expansion, RRF fusion, optional rerank.

Every stage can be switched on or off through :class:`~sourcebound.config.RetrievalConfig`
(or per call), which is what the ablation in ``eval/`` does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Hashable, Protocol, Sequence

import numpy as np

from .config import RetrievalConfig
from .graph import graph_candidates
from .models import Passage
from .store import Index
from .textutil import content_terms


# ---------------------------------------------------------------------------- pure fusion helpers


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60, weights: Sequence[float] | None = None) -> list[tuple[Hashable, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of weight / (k + rank), rank starting at 1.

    Ties are broken by the best rank the item reached in any list, then by first appearance."""
    scores: dict[Hashable, float] = {}
    best: dict[Hashable, int] = {}
    first: dict[Hashable, int] = {}
    for li, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[li]
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a duplicate within one list counts once, at its best rank
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
            best[item] = min(best.get(item, rank), rank)
            first.setdefault(item, len(first))
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], first[kv[0]]))


def mmr(query_vec: np.ndarray, vectors: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    """Maximal marginal relevance over unit-norm vectors. Returns indices into `vectors`, in pick order.

    Each step picks the candidate maximising  lam * sim(query, d) - (1 - lam) * max sim(d, already picked).
    """
    n = len(vectors)
    if n == 0:
        return []
    rel = vectors @ query_vec
    pair = vectors @ vectors.T
    picked: list[int] = []
    remaining = set(range(n))
    while remaining and len(picked) < k:
        def value(i: int) -> float:
            redundancy = max((pair[i, j] for j in picked), default=0.0)
            return lam * float(rel[i]) - (1.0 - lam) * float(redundancy)

        nxt = max(remaining, key=lambda i: (value(i), -i))
        picked.append(nxt)
        remaining.discard(nxt)
    return picked


# ---------------------------------------------------------------------------- reranker interface


class Reranker(Protocol):
    def score(self, question: str, passages: Sequence[str]) -> list[float]: ...


# ---------------------------------------------------------------------------- retriever


@dataclass
class RetrievalResult:
    passages: list[Passage]
    relevant: bool
    reason: str
    stages: dict[str, int] = field(default_factory=dict)  # candidates contributed per stage
    best_vector: float | None = None  # highest cosine similarity the vector stage saw


def _stem(t: str) -> str:
    return t[:5] if len(t) > 5 else t


def term_overlap(terms: Sequence[str], text: str) -> tuple[int, int]:
    """(matched, total) distinct query terms found in `text` (crude prefix stemming)."""
    uniq = list(dict.fromkeys(terms))
    words = {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9_\-]*", text.lower())}
    matched = sum(1 for t in uniq if _stem(t) in words)
    return matched, len(uniq)


class Retriever:
    def __init__(self, index: Index, cfg: RetrievalConfig | None = None, reranker: Reranker | None = None):
        self.index = index
        self.cfg = cfg or RetrievalConfig()
        self.reranker = reranker

    def options(self, **overrides) -> RetrievalConfig:
        return self.cfg.model_copy(update={k: v for k, v in overrides.items() if v is not None})

    def search(self, query: str, *, doc_ids: Sequence[int] | None = None, **overrides) -> RetrievalResult:
        o = self.options(**overrides)
        index = self.index
        rankings: dict[str, list[int]] = {}
        vec_scores: dict[int, float] = {}
        best_vec = None

        if o.vector:
            qv = index.embedder.embed_query(query) if index.embedder else None
            if qv is None:
                raise RuntimeError("vector search needs an embedder")
            hits = index.vector_search(qv, o.vector_fetch, doc_ids, with_embeddings=o.mmr)
            vec_scores = {h.id: h.similarity for h in hits}
            best_vec = max(vec_scores.values(), default=None)
            if o.mmr and hits:
                order = mmr(qv, np.vstack([h.embedding for h in hits]), o.candidates, o.mmr_lambda)
                rankings["vector"] = [hits[i].id for i in order]
            else:
                rankings["vector"] = [h.id for h in hits[: o.candidates]]

        terms = content_terms(query)
        if o.bm25:
            rankings["bm25"] = [cid for cid, _ in index.bm25_search(terms, o.candidates, doc_ids)]

        graph_hits = []
        if o.graph:
            graph_hits = graph_candidates(index, query, o.graph_max_chunks, doc_ids)
            rankings["graph"] = [h.chunk_id for h in graph_hits]

        if not rankings:
            raise ValueError("every retrieval stage is switched off")

        fused = rrf(list(rankings.values()), k=o.rrf_k)
        pool_size = max(o.top_k, o.rerank_candidates) if o.rerank else o.top_k
        ids = [cid for cid, _ in fused[:pool_size]]
        passages = index.get_passages(ids)
        out: list[Passage] = []
        for cid, score in fused[:pool_size]:
            p = passages.get(cid)
            if p is None:
                continue
            p.score = score
            p.vector_score = vec_scores.get(cid)
            p.found_by = [name for name, r in rankings.items() if cid in r]
            out.append(p)

        if o.rerank and out:
            if self.reranker is None:
                raise RuntimeError("rerank is enabled but no reranker was provided")
            scores = self.reranker.score(query, [p.text for p in out])
            for p, s in zip(out, scores):
                p.rerank_score = float(s)
            out = sorted(out, key=lambda p: (-(p.rerank_score or 0.0), -p.score))
        out = out[: o.top_k]

        relevant, reason = self._assess(o, terms, out, best_vec, bool(graph_hits))
        return RetrievalResult(out, relevant, reason, {k: len(v) for k, v in rankings.items()}, best_vec)

    @staticmethod
    def _assess(o: RetrievalConfig, terms: list[str], passages: list[Passage], best_vec: float | None, has_graph: bool) -> tuple[bool, str]:
        """Decide whether retrieval found anything worth answering from.

        Relevant when any of these holds: the best vector similarity reaches `min_vector_score`;
        a returned passage contains enough of the question's content words (at least
        `min_term_overlap` of them, and at least two unless the question has only one); or the
        question names an entity that the graph links to a passage.
        """
        if not passages:
            return False, "no passages matched the question"
        if best_vec is not None and best_vec >= o.min_vector_score:
            return True, f"vector similarity {best_vec:.2f} >= {o.min_vector_score}"
        if o.bm25 and terms:
            for p in passages:
                matched, total = term_overlap(terms, p.text + " " + p.heading)
                if matched >= min(2, total) and matched / total >= o.min_term_overlap:
                    return True, f"{matched}/{total} question terms found in a passage"
        if has_graph:
            return True, "question mentions entities in the graph"
        bits = []
        if best_vec is not None:
            bits.append(f"best vector similarity {best_vec:.2f} < {o.min_vector_score}")
        if o.bm25:
            bits.append("no passage shares enough question terms")
        return False, "; ".join(bits) or "no signal above threshold"
