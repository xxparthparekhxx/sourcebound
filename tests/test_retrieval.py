import numpy as np
import pytest

from docgraph.config import Config, EndpointConfig
from docgraph.ingest.pipeline import ingest_path
from docgraph.llm import ChatClient
from docgraph.retrieval import Retriever, mmr, rrf, term_overlap
from docgraph.store import Index
from fakes import Reply

import json


# ------------------------------------------------------------------------ pure functions

def test_rrf_scores_and_order():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf([["a", "b", "c"], ["b", "d"]])] == ["b", "a", "d", "c"]


def test_rrf_with_one_list_keeps_its_order_and_weights_apply():
    assert [d for d, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]
    top = rrf([["a"], ["b"]], weights=[1.0, 3.0])
    assert top[0][0] == "b"


def test_rrf_counts_a_duplicate_in_one_list_once():
    assert dict(rrf([["a", "a", "b"]]))["a"] == pytest.approx(1 / 61)


def test_rrf_ties_prefer_the_better_best_rank():
    # a: rank 1 in list 1 only; b: rank 1 in list 2 only -> equal scores, first appearance decides
    assert [d for d, _ in rrf([["a"], ["b"]])] == ["a", "b"]


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_mmr_prefers_a_diverse_second_pick():
    q = unit(1, 0.2, 0)
    vecs = np.vstack([unit(1, 0.3, 0), unit(1, 0.31, 0), unit(0.6, 0, 1)])   # 0 and 1 are near-duplicates
    assert mmr(q, vecs, 2, lam=1.0) == [0, 1]       # pure relevance picks the duplicate
    assert mmr(q, vecs, 2, lam=0.5) == [0, 2]       # with a diversity penalty the different one wins
    assert mmr(q, vecs, 5, lam=0.5)[0] == 0 and len(mmr(q, vecs, 5, lam=0.5)) == 3
    assert mmr(q, np.zeros((0, 3), dtype=np.float32), 3) == []


def test_term_overlap_stems_crudely():
    assert term_overlap(["transformers", "attention"], "A transformer uses attention heads") == (2, 2)
    assert term_overlap(["mongolia", "capital"], "nothing relevant") == (0, 2)


# ------------------------------------------------------------------------ retriever

DOCS = {
    "cats.md": "# Cats\n\n## Diet\n\nCats are obligate carnivores and eat meat, fish and specially formulated kibble.\n\n## Sleep\n\nCats sleep for up to sixteen hours every day, mostly in sunny spots.\n",
    "dogs.md": "# Dogs\n\n## Training\n\nDogs respond well to positive reinforcement and short daily training sessions.\n\n## Diet\n\nDogs are omnivores and can eat meat, rice and vegetables.\n",
    "space.md": "# Space\n\n## Moons\n\nJupiter has dozens of moons, including Ganymede, the largest moon in the solar system.\n",
}


@pytest.fixture
def setup(tmp_path, embedder):
    cfg = Config(data_dir=tmp_path / "idx")
    cfg.retrieval.min_vector_score = 0.2   # the hashing embedder gives lower cosines than a real model
    index = Index.open(cfg, embedder)
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text)
    ingest_path(index, d, cfg)
    yield cfg, index
    index.close()


def titles(res):
    return [p.document for p in res.passages]


def test_each_stage_can_be_switched_on_and_off(setup):
    cfg, index = setup
    r = Retriever(index, cfg.retrieval)
    q = "How many moons does Jupiter have?"
    both = r.search(q)
    assert titles(both)[0] == "Space" and set(both.passages[0].found_by) == {"vector", "bm25"}
    v = r.search(q, bm25=False)
    assert v.stages == {"vector": 5} or set(v.stages) == {"vector"}
    assert all(p.found_by == ["vector"] for p in v.passages)
    b = r.search(q, vector=False)
    assert set(b.stages) == {"bm25"} and titles(b) == ["Space"]        # BM25 only returns chunks that match a term
    assert all(p.found_by == ["bm25"] for p in b.passages)
    with pytest.raises(ValueError):
        r.search(q, vector=False, bm25=False)


def test_mmr_toggle_changes_vector_ordering_but_not_the_candidate_set(setup):
    cfg, index = setup
    r = Retriever(index, cfg.retrieval)
    a = r.search("what do cats and dogs eat", bm25=False, mmr=True, top_k=5)
    b = r.search("what do cats and dogs eat", bm25=False, mmr=False, top_k=5)
    assert {p.id for p in a.passages} == {p.id for p in b.passages}


def test_top_k_and_document_filter(setup):
    cfg, index = setup
    r = Retriever(index, cfg.retrieval)
    assert len(r.search("diet meat", top_k=2).passages) == 2
    dog_id = index.doc_ids_for_sources([d.source for d in index.documents() if d.title == "Dogs"])
    res = r.search("diet meat", doc_ids=dog_id)
    assert set(titles(res)) == {"Dogs"}


def test_relevance_gate_says_no_for_unrelated_questions(setup):
    cfg, index = setup
    r = Retriever(index, cfg.retrieval)
    assert r.search("How many moons does Jupiter have?").relevant
    nothing = r.search("zebra quantum mortgage")
    assert not nothing.relevant and nothing.reason
    # question with one matching generic word out of two is not enough lexically
    assert not r.search("capital of Mongolia", min_vector_score=0.99).relevant
    strict = r.search("How many moons does Jupiter have?", min_vector_score=0.99, min_term_overlap=0.99)
    assert not strict.relevant


class StubReranker:
    def __init__(self, prefer: str):
        self.prefer, self.calls = prefer, []

    def score(self, question, passages):
        self.calls.append((question, list(passages)))
        return [1.0 if self.prefer in p else 0.0 for p in passages]


def test_rerank_reorders_the_fused_candidates(setup):
    cfg, index = setup
    stub = StubReranker("positive reinforcement")
    r = Retriever(index, cfg.retrieval, reranker=stub)
    base = r.search("diet of cats and dogs", top_k=3)
    assert "positive reinforcement" not in base.passages[0].text
    rr = r.search("diet of cats and dogs", top_k=3, rerank=True, rerank_candidates=5)
    assert "positive reinforcement" in rr.passages[0].text and rr.passages[0].rerank_score == 1.0
    assert len(stub.calls[0][1]) == 5   # the whole pool of five chunks was scored
    with pytest.raises(RuntimeError):
        Retriever(index, cfg.retrieval).search("x", rerank=True)


def test_graph_stage_adds_chunks_that_text_search_misses(setup, tmp_path, llm_servers):
    cfg, index = setup
    # Entities: Jupiter appears in the Space chunk; a relation links it to "Great Red Spot" mentioned in a chunk
    # that shares no words with the question.
    d = tmp_path / "docs2"
    d.mkdir()
    (d / "storm.md").write_text("# Storms\n\nThe Great Red Spot is a persistent anticyclone larger than Earth.\n")
    (d / "planet.md").write_text("# Planet\n\nJupiter is the biggest planet and hosts the Great Red Spot storm.\n")

    def respond(body):
        t = body["messages"][-1]["content"]
        if "persistent anticyclone" in t:
            return Reply(content=json.dumps({"entities": [{"name": "Great Red Spot", "type": "concept"}], "relations": []}))
        if "biggest planet" in t:
            return Reply(content=json.dumps({
                "entities": [{"name": "Jupiter", "type": "place"}, {"name": "Great Red Spot", "type": "concept"}],
                "relations": [{"source": "Jupiter", "predicate": "hosts", "target": "Great Red Spot"}]}))
        return Reply(content='{"entities": [], "relations": []}')

    srv = llm_servers(respond)
    client = ChatClient([EndpointConfig(name="g", base_url=srv.base_url, model="m")])
    ingest_path(index, d, cfg, client, graph=True)

    r = Retriever(index, cfg.retrieval)
    q = "Tell me about Jupiter"
    without = r.search(q, vector=False, graph=False)
    assert not any("anticyclone" in p.text for p in without.passages)
    with_graph = r.search(q, vector=False, graph=True)
    assert any("anticyclone" in p.text for p in with_graph.passages)          # reached through the 1-hop neighbour
    assert any("graph" in p.found_by for p in with_graph.passages)
    assert with_graph.relevant
