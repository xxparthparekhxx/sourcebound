"""Knowledge graph: LLM extraction at ingest time, expansion at question time. SQLite only."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import GraphConfig
from .llm import ChatClient, ChatError
from .models import Passage
from .store import Index, norm_entity

SYSTEM_PROMPT = """You extract a knowledge graph from one passage of a document.
Reply with a single JSON object on one line, with no code fence and no other text, in exactly this shape:
{"entities": [{"name": "...", "type": "..."}], "relations": [{"source": "...", "predicate": "...", "target": "..."}]}

Rules:
- Entities are specific things named in the passage: people, organisations, methods, models, datasets, metrics, tools, places, technical concepts. Use the name as written. No pronouns, no generic words such as "the paper" or "results". No bare numbers or percentages, no citation keys, no references to tables or figures.
- "type" is one lowercase word (person, organization, method, model, dataset, metric, tool, place, concept, other).
- At most {max_entities} entities and {max_relations} relations.
- A relation links two entities from your own list. "predicate" is a short verb phrase of at most five words, taken from what the passage says.
- Do not add facts the passage does not state. If there is nothing worth extracting, return {"entities": [], "relations": []}."""

_GENERIC = frozenset(
    "it this that these those they them he she we i you the paper article document section result results "
    "approach method model data one two work study authors author".split()
)


class _Ent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=2, max_length=80)
    type: str = Field(default="", max_length=40)

    @field_validator("name", "type", mode="before")
    @classmethod
    def _strip(cls, v):
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("must be a string")
        return v.strip()


class _Rel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str = Field(min_length=2, max_length=80)
    predicate: str = Field(min_length=1, max_length=60)
    target: str = Field(min_length=2, max_length=80)

    @field_validator("source", "predicate", "target", mode="before")
    @classmethod
    def _strip(cls, v):
        if not isinstance(v, str):
            raise ValueError("must be a string")
        return v.strip()


class Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entities: list[_Ent] = Field(default_factory=list)
    relations: list[_Rel] = Field(default_factory=list)


def parse_extraction(raw: str, max_entities: int = 12, max_relations: int = 12) -> Extraction | None:
    """Parse and validate a model reply against the schema. Returns None when the reply is not valid,
    in which case the chunk's extraction is dropped.

    The reply must be a JSON object whose "entities" and "relations" are lists. Each item is then
    validated on its own and an item that fails is discarded; if items were present but none of them
    validate, the whole reply counts as invalid.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    lists = {}
    for key, limit in (("entities", max_entities), ("relations", max_relations)):
        value = data.get(key, [])
        if not isinstance(value, list):
            return None
        lists[key] = value[:limit]  # models routinely ignore count limits; clip rather than reject

    entities: list[_Ent] = []
    for item in lists["entities"]:
        try:
            entities.append(_Ent.model_validate(item))
        except ValidationError:
            pass
    relations: list[_Rel] = []
    for item in lists["relations"]:
        try:
            relations.append(_Rel.model_validate(item))
        except ValidationError:
            pass
    if (lists["entities"] or lists["relations"]) and not entities and not relations:
        return None

    # Semantic clean-up: drop generic/duplicate entities, and relations whose ends are not in the list.
    seen: dict[str, str] = {}
    ents: list[_Ent] = []
    for e in entities:
        n = norm_entity(e.name)
        if not n or all(w in _GENERIC for w in n.split()) or n in seen:
            continue
        seen[n] = e.name
        ents.append(e)
    rels = [r for r in relations if norm_entity(r.source) in seen and norm_entity(r.target) in seen
            and norm_entity(r.source) != norm_entity(r.target)]
    return Extraction(entities=ents, relations=rels)


@dataclass
class GraphStats:
    chunks: int = 0
    extracted: int = 0
    failed: int = 0  # reply was not valid JSON for the schema: dropped
    errors: int = 0  # the LLM call itself failed: left pending, can be retried
    entities: int = 0
    relations: int = 0
    failures: list[str] = field(default_factory=list)


def extract_chunk(client: ChatClient, text: str, cfg: GraphConfig, *, heading: str = "") -> Extraction | None:
    system = SYSTEM_PROMPT.replace("{max_entities}", str(cfg.max_entities)).replace("{max_relations}", str(cfg.max_relations))
    user = (f"Section: {heading}\n\n" if heading else "") + f"Passage:\n{text}"
    result = client.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        params={"max_tokens": 600},
        # JSON output: repetition penalties push the model away from repeating keys and break the shape
        force={"temperature": 0.1, "presence_penalty": 0.0, "frequency_penalty": 0.0},
    )
    return parse_extraction(result.content, cfg.max_entities, cfg.max_relations)


def extract_graph(
    index: Index,
    client: ChatClient,
    cfg: GraphConfig,
    doc_ids: Sequence[int] | None = None,
    progress: Callable[[GraphStats], None] | None = None,
) -> GraphStats:
    """Extract entities and relations for every chunk that has not been processed yet.

    LLM calls run in a small thread pool; results are written from this thread. A reply that does
    not validate is dropped (the chunk is marked failed and not retried); a failed call leaves the
    chunk pending so a later run picks it up.
    """
    pending = index.pending_graph_chunks(doc_ids)
    stats = GraphStats(chunks=len(pending))

    def work(p: Passage):
        try:
            return p, extract_chunk(client, p.text, cfg, heading=p.heading), None
        except ChatError as e:
            return p, None, e

    if not pending:
        return stats
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        for p, ex, err in pool.map(work, pending):
            if err is not None:
                stats.errors += 1
                if len(stats.failures) < 5:
                    stats.failures.append(f"chunk {p.id}: {err}")
            elif ex is None:
                stats.failed += 1
                index.mark_graph(p.id, 2)
            else:
                index.store_graph(
                    p.id,
                    [(e.name, e.type) for e in ex.entities],
                    [(r.source, r.predicate, r.target) for r in ex.relations],
                )
                stats.extracted += 1
                stats.entities += len(ex.entities)
                stats.relations += len(ex.relations)
            if progress:
                progress(stats)
    return stats


# ---------------------------------------------------------------------------- expansion


@dataclass
class GraphHit:
    chunk_id: int
    score: float
    via: list[str]  # entity names that led here


def _weight(mentions: int) -> float:
    """Entities that appear everywhere say little; weight by how specific they are."""
    return 1.0 / math.log(2.0 + mentions)


NEIGHBOUR_DISCOUNT = 0.5


def graph_candidates(
    index: Index, question: str, limit: int = 20, doc_ids: Sequence[int] | None = None
) -> list[GraphHit]:
    """Chunks mentioning entities found in the question, plus chunks mentioning their 1-hop neighbours."""
    matched = index.match_entities(question)
    if not matched:
        return []
    scores: dict[int, float] = {}
    via: dict[int, list[str]] = {}

    def add(chunk_ids: list[int], w: float, name: str) -> None:
        for cid in chunk_ids:
            scores[cid] = scores.get(cid, 0.0) + w
            names = via.setdefault(cid, [])
            if name not in names:
                names.append(name)

    matched_ids = {e.id for e in matched}
    for e in matched:
        add(index.entity_chunks(e.id, doc_ids), _weight(e.mentions), e.name)
        done: set[int] = set()
        for nb in index.neighbours(e.id):
            if nb["id"] in matched_ids or nb["id"] in done:
                continue
            done.add(nb["id"])
            add(index.entity_chunks(nb["id"], doc_ids), NEIGHBOUR_DISCOUNT * _weight(nb["n"]), nb["name"])
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [GraphHit(cid, s, via[cid]) for cid, s in ranked]
