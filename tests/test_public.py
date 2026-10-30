import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sourcebound.config import Config, ConfigError
from sourcebound.ingest.pipeline import ingest_path
from sourcebound.llm import ChatClient
from sourcebound.public import GatewayAuth, Slots, create_public_app
from sourcebound.runtime import Runtime
from fakes import Reply
from test_api import DOC, parse_sse

SECRET = "s3cret-value-for-the-gateway"
H = {"X-Gateway-Secret": SECRET}


class FakeScorer:
    """Stands in for openjev-flash: a sentence is 'entailed' when it contains the word sixteen."""

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.array([[0.1, 0.9, 0.0] if "sixteen" in h else [0.4, 0.1, 0.5] for _, h in pairs], dtype=np.float32)

    def score(self, q, passages):
        return [0.5] * len(passages)


def build(tmp_path, embedder, srv, **serve):
    cfg = Config(data_dir=tmp_path / "idx", chat={"endpoints": [{"name": "fake", "base_url": srv.base_url, "model": "m"}]}, serve=serve)
    cfg.retrieval.min_vector_score = 0.2
    rt = Runtime(cfg, embedder=embedder, client=ChatClient(cfg.chat.endpoints))
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "cats.md").write_text(DOC)
    ingest_path(rt.index, docs, cfg)
    rt.reranker._scorer = FakeScorer()
    return cfg, rt


@pytest.fixture
def pub(tmp_path, embedder, llm_servers):
    srv = llm_servers(Reply(content="Cats sleep sixteen hours every day [1]. They eat meat [2]. Fish is nice [9]."))
    cfg, rt = build(tmp_path, embedder, srv)
    with TestClient(create_public_app(cfg, runtime=rt, env={"SOURCEBOUND_GATEWAY_SECRET": SECRET})) as tc:
        yield tc, rt, srv
    rt.close()


def ask(tc, question="How many hours do cats sleep every day?", headers=H):
    with tc.stream("POST", "/ask", json={"question": question}, headers=headers) as r:
        return r.status_code, "".join(r.iter_text())


def test_healthz_is_open_and_reveals_nothing(pub):
    tc, _, _ = pub
    r = tc.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-Gateway-Secret": "wrong"}, {"X-Gateway-Secret": SECRET + "x"}, {"X-Gateway-Secret": ""}])
def test_everything_else_needs_the_exact_secret(pub, headers):
    tc, _, srv = pub
    for method, path in (("POST", "/ask"), ("GET", "/passages/1"), ("GET", "/"), ("GET", "/openapi.json"), ("GET", "/docs")):
        r = tc.request(method, path, json={"question": "hi"} if method == "POST" else None, headers=headers)
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}, (method, path)
    assert srv.requests == []  # nothing reached retrieval or the model


def test_refuses_to_start_without_a_secret(tmp_path, embedder, llm_servers):
    cfg, rt = build(tmp_path, embedder, llm_servers(Reply(content="x")))
    for env in ({}, {"SOURCEBOUND_GATEWAY_SECRET": "short"}):
        with pytest.raises(ConfigError, match="SOURCEBOUND_GATEWAY_SECRET"):
            create_public_app(cfg, runtime=rt, env=env)
    rt.close()


def test_secret_env_name_is_configurable(tmp_path, embedder, llm_servers):
    cfg, rt = build(tmp_path, embedder, llm_servers(Reply(content="x")), gateway_secret_env="MY_SECRET")
    with TestClient(create_public_app(cfg, runtime=rt, env={"MY_SECRET": SECRET})) as tc:
        assert tc.post("/ask", json={"question": "cats"}, headers=H).status_code == 200
    rt.close()


def test_question_only_and_length_limited(pub):
    tc, _, srv = pub
    assert tc.post("/ask", json={"question": "q" * 501}, headers=H).status_code == 422
    assert tc.post("/ask", json={"question": ""}, headers=H).status_code == 422
    assert tc.post("/ask", json={"question": "cats", "top_k": 20}, headers=H).status_code == 422   # no client-side settings
    assert tc.post("/ask", json={"question": "cats", "mode": "agent"}, headers=H).status_code == 422
    assert tc.post("/ask", json={}, headers=H).status_code == 422
    assert srv.requests == []
    status, _ = ask(tc, "c" * 500)
    assert status == 200


def test_stream_order_citations_and_per_sentence_faithfulness(pub):
    tc, rt, _ = pub
    status, text = ask(tc)
    assert status == 200
    events = parse_sse(text)
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-3:] == ["citations", "faithfulness", "done"]
    assert set(names) == {"retrieval", "token", "citations", "faithfulness", "done"}
    answer = "".join(d["text"] for n, d in events if n == "token")
    assert answer == "Cats sleep sixteen hours every day [1]. They eat meat [2]. Fish is nice."   # [9] was never shown
    d = dict(events)
    cits = d["citations"]["citations"]
    assert [c["n"] for c in cits] == [1, 2] and set(cits[0]) == {"n", "id", "document", "file", "heading", "text"}
    assert d["citations"]["flagged"] == [9]
    sents = d["faithfulness"]["sentences"]
    assert [s["cited"] for s in sents] == [[1], [2], []]
    assert [s["supported"] for s in sents] == [True, False, None]            # the uncited sentence is not scored
    assert sents[2]["p_entail"] is None and sents[0]["p_entail"] == 0.9
    premise, hyp = rt.reranker._scorer.calls[-1][0]
    assert "sixteen hours" in premise and "[1]" not in hyp and hyp.startswith("Cats sleep sixteen")
    done = d["done"]
    assert done["no_answer"] is False and set(done["timings"]) == {"retrieval_s", "llm_s", "faithfulness_s", "total_s"}
    assert "api_key" not in text and "base_url" not in text and "http" not in json.dumps(done)


def test_off_topic_question_answers_without_a_model_call_and_without_faithfulness_scoring(pub):
    tc, rt, srv = pub
    status, text = ask(tc, "zzzz qqqq xxxx")
    events = parse_sse(text)
    assert [n for n, _ in events][-1] == "done" and dict(events)["done"]["no_answer"] is True
    assert dict(events)["faithfulness"]["sentences"] == [] and srv.requests == []


def test_model_failure_is_one_generic_error_event(tmp_path, embedder, llm_servers):
    srv = llm_servers(Reply(status=429, body='{"error": {"message": "secret-provider-detail free-models-per-day"}}'))
    cfg, rt = build(tmp_path, embedder, srv)
    with TestClient(create_public_app(cfg, runtime=rt, env={"SOURCEBOUND_GATEWAY_SECRET": SECRET})) as tc:
        status, text = ask(tc)
    rt.close()
    events = parse_sse(text)
    assert status == 200 and events[-1][0] == "error"
    assert "secret-provider-detail" not in text and "rate limited" in events[-1][1]["message"]


def test_busy_returns_503_and_the_slot_frees_afterwards(tmp_path, embedder, llm_servers):
    gate, entered = threading.Event(), threading.Event()

    def respond(body):
        entered.set()
        gate.wait(10)
        return Reply(content="Cats sleep sixteen hours [1].")

    srv = llm_servers(respond)
    cfg, rt = build(tmp_path, embedder, srv, max_concurrent=1)
    with TestClient(create_public_app(cfg, runtime=rt, env={"SOURCEBOUND_GATEWAY_SECRET": SECRET})) as tc:
        first: dict = {}
        t = threading.Thread(target=lambda: first.update(zip(("status", "text"), ask(tc))))
        t.start()
        assert entered.wait(10)
        r = tc.post("/ask", json={"question": "How many hours do cats sleep?"}, headers=H)
        assert r.status_code == 503 and r.headers["retry-after"] == "5" and r.json() == {"error": "busy"}
        assert tc.get("/healthz").status_code == 200                       # health stays up while busy
        gate.set()
        t.join(10)
        assert first["status"] == 200 and parse_sse(first["text"])[-1][0] == "done"
        status, _ = ask(tc)
        assert status == 200                                              # the slot was released
    rt.close()


def test_slots_release_is_idempotent_and_bounded():
    s = Slots(2)
    a, b = s.take(), s.take()
    assert a and b and s.take() is None
    a.release(); a.release()
    c = s.take()
    assert c is not None and s.take() is None


def test_faithfulness_can_be_switched_off(tmp_path, embedder, llm_servers):
    srv = llm_servers(Reply(content="Cats sleep sixteen hours [1]."))
    cfg, rt = build(tmp_path, embedder, srv, faithfulness=False)
    with TestClient(create_public_app(cfg, runtime=rt, env={"SOURCEBOUND_GATEWAY_SECRET": SECRET})) as tc:
        _, text = ask(tc)
    assert dict(parse_sse(text))["faithfulness"]["sentences"] == [] and rt.reranker._scorer.calls == []
    rt.close()


def test_auth_middleware_ignores_non_http_scopes():
    import asyncio

    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(GatewayAuth(inner, "x" * 16)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]
