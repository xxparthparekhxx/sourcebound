import json

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient

from docgraph.api import create_app
from docgraph.config import Config, EndpointConfig
from docgraph.ingest.pipeline import ingest_path
from docgraph.llm import ChatClient
from docgraph.runtime import Runtime
from fakes import Reply

DOC = "# Cats\n\n## Sleep\n\nCats sleep for up to sixteen hours every day, mostly in sunny spots.\n\n## Diet\n\nCats are obligate carnivores and eat meat and fish.\n"


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events


@pytest.fixture
def served(tmp_path, embedder, llm_servers):
    srv = llm_servers(Reply(content="Cats sleep sixteen hours [1]. Extra [5]."))
    cfg = Config(data_dir=tmp_path / "idx", chat={"endpoints": [{"name": "fake", "base_url": srv.base_url, "model": "m"}]})
    cfg.retrieval.min_vector_score = 0.2
    client = ChatClient(cfg.chat.endpoints)
    rt = Runtime(cfg, embedder=embedder, client=client)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cats.md").write_text(DOC)
    ingest_path(rt.index, docs, cfg)
    with TestClient(create_app(runtime=rt)) as tc:
        yield tc, rt, srv
    rt.close()


def test_healthz_reports_index_and_endpoints_without_secrets(served):
    tc, _, _ = served
    r = tc.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["documents"] == 1 and body["chunks"] == 2
    assert body["embedding"]["model"] == "fake-hash" and body["embedding"]["dimension"] == 128
    assert body["chat_endpoints"] == [{"name": "fake", "model": "m", "usable": True}]
    assert "api_key" not in r.text and "http" not in json.dumps(body["chat_endpoints"])


def test_ask_streams_tokens_then_citations_then_done(served):
    tc, _, srv = served
    with tc.stream("POST", "/ask", json={"question": "How many hours do cats sleep every day?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = parse_sse(text)
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done" and names[-2] == "citations"
    assert names.count("token") > 3
    assert max(i for i, n in enumerate(names) if n == "token") < names.index("citations")   # tokens first
    answer = "".join(d["text"] for n, d in events if n == "token")
    assert answer == "Cats sleep sixteen hours [1]. Extra."                                   # [5] was removed
    cit = dict(events)["citations"]
    assert cit["flagged"] == [5]
    c = cit["citations"][0]
    assert c["n"] == 1 and c["document"] == "Cats" and c["file"] == "cats.md" and c["heading"] == "Sleep"
    assert "sixteen hours" in c["text"] and "page" in c
    assert dict(events)["done"]["mode"] == "pipeline"


def test_passage_endpoint_returns_document_page_text_and_neighbours(served):
    tc, rt, _ = served
    docs = rt.index.documents()
    first, second = rt.index.document_passages(docs[0].id)
    r = tc.get(f"/passages/{first.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == first.text and body["document"] == "Cats" and body["heading"] == "Sleep"
    assert body["file"] == "cats.md" and body["page"] is None and body["next_id"] == second.id and body["prev_id"] is None
    assert "/" not in body["file"]            # a file name, not a server path
    assert tc.get(f"/passages/{second.id}").json()["prev_id"] == first.id
    assert tc.get("/passages/999999").status_code == 404
    assert tc.get("/passages/not-a-number").status_code == 422


def test_ask_validates_input_and_reports_an_empty_index(served, tmp_path, embedder):
    tc, rt, _ = served
    assert tc.post("/ask", json={"question": ""}).status_code == 422
    assert tc.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422
    rt2 = Runtime(Config(data_dir=tmp_path / "empty"), embedder=embedder, client=rt.client)
    with TestClient(create_app(runtime=rt2)) as tc2:
        r = tc2.post("/ask", json={"question": "anything"})
        assert parse_sse(r.text) == [("error", {"message": "The index is empty. Run `docgraph ingest <path>` first."})]
    rt2.close()


def test_stage_switches_in_the_request_reach_the_retriever(served):
    tc, _, _ = served
    r = tc.post("/ask", json={"question": "sixteen hours cats sleep", "vector": False, "top_k": 1})
    retrieval = dict(parse_sse(r.text))["retrieval"]
    assert retrieval["stages"] == {"bm25": 1} or set(retrieval["stages"]) == {"bm25"}


def test_chat_failure_is_reported_in_band(tmp_path, embedder, llm_servers):
    srv = llm_servers(Reply(status=500))
    cfg = Config(data_dir=tmp_path / "idx")
    cfg.retrieval.min_vector_score = 0.2
    rt = Runtime(cfg, embedder=embedder, client=ChatClient([EndpointConfig(name="f", base_url=srv.base_url, model="m")]))
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "cats.md").write_text(DOC)
    ingest_path(rt.index, tmp_path / "d", cfg)
    with TestClient(create_app(runtime=rt)) as tc:
        events = parse_sse(tc.post("/ask", json={"question": "How many hours do cats sleep?"}).text)
    assert events[-1][0] == "error" and "all chat endpoints failed" in events[-1][1]["message"]
    rt.close()


def test_ask_dependencies_hook_runs_before_the_handler(served, tmp_path, embedder):
    _, rt, _ = served

    def deny():
        raise HTTPException(429, "daily limit reached")

    with TestClient(create_app(runtime=rt, ask_dependencies=[deny])) as tc:
        r = tc.post("/ask", json={"question": "hi"})
        assert r.status_code == 429
        assert tc.get("/healthz").status_code == 200


def test_the_ui_is_served_and_is_self_contained(served):
    tc, _, _ = served
    r = tc.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert "fetch(\"/ask\"" in html and "/passages/" in html
    assert "http://" not in html and "https://" not in html            # no CDN, no external assets
