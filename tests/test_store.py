import json

import pytest

from docgraph.config import Config, EndpointConfig
from docgraph.graph import graph_candidates, parse_extraction
from docgraph.ingest.pipeline import ingest_path
from docgraph.llm import ChatClient
from docgraph.store import EmbeddingMismatch, Index
from docgraph.textutil import content_terms
from fakes import FakeEmbedder, Reply


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=tmp_path / "idx")


@pytest.fixture
def index(cfg, embedder):
    idx = Index.open(cfg, embedder)
    yield idx
    idx.close()


def write(tmp_path, name, text):
    p = tmp_path / "docs" / name
    p.parent.mkdir(exist_ok=True)
    p.write_text(text)
    return p


ALPHA = "# Alpha Guide\n\n## Setup\n\nInstall the widget compiler before anything else.\n\n## Usage\n\nRun the compiler with the verbose flag to see tracing output.\n"
BETA = "# Beta Notes\n\n## Storage\n\nThe archive stores snapshots in a column database nightly.\n"


def test_ingest_stores_chunks_fts_and_vectors(tmp_path, cfg, index, embedder):
    write(tmp_path, "alpha.md", ALPHA)
    write(tmp_path, "beta.md", BETA)
    rep = ingest_path(index, tmp_path / "docs", cfg)
    assert [d.status for d in rep.documents] == ["ingested", "ingested"]
    assert index.stats()["documents"] == 2 and index.stats()["chunks"] == 3
    assert index.collection.count() == 3
    hits = index.bm25_search(content_terms("verbose tracing compiler"), 5)
    top = index.get_passage(hits[0][0])
    assert top.heading == "Usage" and top.document == "Alpha Guide"
    v = index.vector_search(embedder.embed_query("snapshots column database"), 3)
    assert index.get_passage(v[0].id).document == "Beta Notes"
    assert 0 < v[0].similarity <= 1.0
    # restricting to one document works for both searches
    alpha_id = index.doc_ids_for_sources([str((tmp_path / "docs" / "alpha.md").resolve())])
    assert {index.get_passage(h.id).document for h in index.vector_search(embedder.embed_query("snapshots"), 5, alpha_id)} == {"Alpha Guide"}
    assert all(index.get_passage(i).document == "Alpha Guide" for i, _ in index.bm25_search(["snapshots", "compiler"], 5, alpha_id))


def test_unchanged_file_is_skipped_and_changed_file_replaces_its_chunks(tmp_path, cfg, index):
    f = write(tmp_path, "alpha.md", ALPHA)
    ingest_path(index, f, cfg)
    first_ids = [p.id for p in index.document_passages(index.documents()[0].id)]

    assert ingest_path(index, f, cfg).documents[0].status == "unchanged"
    assert index.stats()["chunks"] == 2

    f.write_text("# Alpha Guide\n\n## Usage\n\nCompletely different text about gardening tomatoes and herbs.\n")
    rep = ingest_path(index, f, cfg)
    assert rep.documents[0].status == "replaced"
    assert index.stats()["documents"] == 1 and index.stats()["chunks"] == 1
    assert index.collection.count() == 1
    new = index.document_passages(index.documents()[0].id)
    assert "gardening" in new[0].text and new[0].id not in first_ids
    assert index.bm25_search(["compiler"], 5) == []          # old text is gone from FTS
    assert index.bm25_search(["tomatoes"], 5)[0][0] == new[0].id
    assert index.get_passage(first_ids[0]) is None


def test_unsupported_and_failing_files_do_not_stop_a_directory_ingest(tmp_path, cfg, index):
    write(tmp_path, "good.md", ALPHA)
    write(tmp_path, "image.png", "not text")
    (tmp_path / "docs" / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    rep = ingest_path(index, tmp_path / "docs", cfg)
    by = {d.path.rsplit("/", 1)[-1]: d for d in rep.documents}
    assert by["good.md"].status == "ingested"
    assert by["broken.pdf"].status == "failed" and by["broken.pdf"].error
    assert "image.png" not in by  # silently ignored inside a directory


def test_embedding_model_mismatch_is_refused(tmp_path, cfg, index):
    write(tmp_path, "alpha.md", ALPHA)
    ingest_path(index, tmp_path / "docs", cfg)
    assert index.embedding_info() == {"model": "fake-hash", "dimension": 128, "provider": "fake"}
    index.close()

    with pytest.raises(EmbeddingMismatch) as e:                       # different model name
        Index.open(cfg, FakeEmbedder("other-model", 128))
    assert "fake-hash" in str(e.value) and "other-model" in str(e.value)
    with pytest.raises(EmbeddingMismatch) as e:                       # same name, different dimension
        Index.open(cfg, FakeEmbedder("fake-hash", 64))
    assert "dimension" in str(e.value)

    ok = Index.open(cfg, FakeEmbedder("fake-hash", 128))               # the same model is accepted
    assert ok.stats()["chunks"] == 2
    ok.close()


def test_empty_index_may_switch_embedding_model_and_reset_clears_the_record(tmp_path, cfg):
    a = Index.open(cfg, FakeEmbedder("m1", 32))
    a.close()
    b = Index.open(cfg, FakeEmbedder("m2", 64))      # nothing indexed yet: allowed
    assert b.embedding_info()["model"] == "m2"
    write(tmp_path, "alpha.md", ALPHA)
    ingest_path(b, tmp_path / "docs", cfg)
    b.reset()
    assert b.stats()["chunks"] == 0 and b.collection.count() == 0
    b.close()
    c = Index.open(cfg, FakeEmbedder("m3", 16))      # reset index accepts another model
    assert c.embedding_info()["model"] == "m3"
    c.close()


def test_delete_document_removes_everything(tmp_path, cfg, index):
    f = write(tmp_path, "alpha.md", ALPHA)
    ingest_path(index, f, cfg)
    assert index.delete_document(str(f.resolve()))
    assert index.stats()["chunks"] == 0 and index.collection.count() == 0
    assert not index.delete_document(str(f.resolve()))


# ------------------------------------------------------------------------ knowledge graph

GRAPH_DOC = (
    "# Team\n\nAlice Chen founded Acme Robotics in Berlin and leads the vision team.\n\n"
    "# Products\n\nAcme Robotics builds the Atlas arm, which Bob Diaz tests every week.\n\n"
    "# Misc\n\nThis paragraph is about nothing in particular, only filler words here.\n"
)


def graph_llm(llm_servers):
    def respond(body):
        text = body["messages"][-1]["content"]
        if "Alice Chen" in text:
            return Reply(content=json.dumps({
                "entities": [{"name": "Alice Chen", "type": "person"}, {"name": "Acme Robotics", "type": "organization"}, {"name": "Berlin", "type": "place"}],
                "relations": [{"source": "Alice Chen", "predicate": "founded", "target": "Acme Robotics"},
                              {"source": "Acme Robotics", "predicate": "is based in", "target": "Berlin"},
                              {"source": "Alice Chen", "predicate": "knows", "target": "Nobody Listed"}],
            }))
        if "Atlas" in text:
            return Reply(content='Sure! ```json\n' + json.dumps({
                "entities": [{"name": "Acme Robotics", "type": "organization"}, {"name": "Atlas arm", "type": "product"}, {"name": "Bob Diaz", "type": "person"}],
                "relations": [{"source": "Acme Robotics", "predicate": "builds", "target": "Atlas arm"},
                              {"source": "Bob Diaz", "predicate": "tests", "target": "Atlas arm"}],
            }) + '\n```')
        return Reply(content="I cannot produce JSON for this one, sorry.")
    return llm_servers(respond)


def test_graph_extraction_validates_stores_and_expands(tmp_path, cfg, index, llm_servers):
    srv = graph_llm(llm_servers)
    client = ChatClient([EndpointConfig(name="g", base_url=srv.base_url, model="m")])
    f = write(tmp_path, "team.md", GRAPH_DOC)
    rep = ingest_path(index, f, cfg, client, graph=True)

    assert rep.graph.chunks == 3 and rep.graph.extracted == 2 and rep.graph.failed == 1 and rep.graph.errors == 0
    st = index.stats()
    assert st["entities"] == 5          # Acme Robotics is shared between two chunks
    assert st["relations"] == 4         # the relation to an entity that was never listed was dropped
    assert st["mentions"] == 6
    acme = index.find_entity("acme robotics")
    assert acme.mentions == 2
    assert {n["name"] for n in index.neighbours(acme.id)} == {"Alice Chen", "Berlin", "Atlas arm"}

    # Question names Alice: her chunk (direct) and the chunk about Acme's other products (1-hop) come back
    hits = graph_candidates(index, "What did Alice Chen found?", 10)
    chunks = index.get_passages([h.chunk_id for h in hits])
    assert "Alice Chen founded" in chunks[hits[0].chunk_id].text
    assert any("Atlas" in c.text for c in chunks.values())
    assert hits[0].score > hits[-1].score
    assert not any("filler" in c.text for c in chunks.values())

    # re-running does not call the model again for finished or dropped chunks
    n = len(srv.requests)
    assert ingest_path(index, f, cfg, client, graph=True).graph.chunks == 0
    assert len(srv.requests) == n


def test_graph_rows_disappear_when_the_document_is_replaced(tmp_path, cfg, index, llm_servers):
    srv = graph_llm(llm_servers)
    client = ChatClient([EndpointConfig(name="g", base_url=srv.base_url, model="m")])
    f = write(tmp_path, "team.md", GRAPH_DOC)
    ingest_path(index, f, cfg, client, graph=True)
    f.write_text("# Other\n\nNothing about the team remains in this replacement document text.\n")
    ingest_path(index, f, cfg, None)
    st = index.stats()
    assert st["entities"] == 0 and st["relations"] == 0 and st["mentions"] == 0


def test_failed_llm_calls_leave_chunks_pending(tmp_path, cfg, index, llm_servers):
    srv = llm_servers(Reply(status=500))
    client = ChatClient([EndpointConfig(name="g", base_url=srv.base_url, model="m")])
    rep = ingest_path(index, write(tmp_path, "team.md", GRAPH_DOC), cfg, client, graph=True)
    assert rep.graph.errors == 3 and rep.graph.extracted == 0
    assert len(index.pending_graph_chunks()) == 3


@pytest.mark.parametrize("raw", [
    "not json at all",
    '{"entities": "Alice", "relations": []}',
    '{"entities": [{"name": 5}], "relations": []}',
    '{"entities": [{"name": "A"}], "relations": []}',        # the only item is invalid (name shorter than 2 characters)
    '[1, 2, 3]',
    '{"entities": [{"name": "Alice"}, {"name": "Bob"}], "relations": "none"}',   # relations is not a list
    '{"entities": [], "relations": [{"name": "Ellipsis", "type": "tool"}]}',     # nothing valid at all
])
def test_invalid_extractions_are_dropped(raw):
    assert parse_extraction(raw) is None


def test_invalid_items_are_dropped_but_valid_ones_kept():
    ex = parse_extraction(json.dumps({
        "entities": [{"name": "Alice"}, {"name": 5}, {"name": "Acme"}],
        "relations": [{"source": "Alice"}, {"name": "Acme", "type": "x"}, {"source": "Alice", "predicate": "works at", "target": "Acme"}],
    }))
    assert [e.name for e in ex.entities] == ["Alice", "Acme"]
    assert [(r.source, r.predicate, r.target) for r in ex.relations] == [("Alice", "works at", "Acme")]


def test_empty_extraction_is_valid():
    ex = parse_extraction('{"entities": [], "relations": []}')
    assert ex is not None and ex.entities == [] and ex.relations == []


def test_extraction_clean_up_and_limits():
    ex = parse_extraction(json.dumps({
        "entities": [{"name": "The paper"}, {"name": "BERT", "type": "model"}, {"name": "bert"}, {"name": "GLUE", "type": "dataset"}] + [{"name": f"Thing {i}"} for i in range(20)],
        "relations": [{"source": "BERT", "predicate": "evaluated on", "target": "GLUE"}, {"source": "BERT", "predicate": "x", "target": "Missing"}],
    }), max_entities=4)
    # clipped to 4 entities first; then the generic "The paper" and the duplicate "bert" are dropped
    assert [e.name for e in ex.entities] == ["BERT", "GLUE"]
    assert [(r.source, r.target) for r in ex.relations] == [("BERT", "GLUE")]
