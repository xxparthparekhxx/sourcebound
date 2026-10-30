
import pytest

from docgraph.answer import Answerer
from docgraph.config import Config, EndpointConfig
from docgraph.ingest.pipeline import ingest_path
from docgraph.llm import ChatClient, PromptTooLong, estimate_tokens
from docgraph.retrieval import Retriever
from docgraph.store import Index
from fakes import Reply

DOCS = {
    "cats.md": "# Cats\n\n## Sleep\n\nCats sleep for up to sixteen hours every day, mostly in sunny spots.\n\n## Diet\n\nCats are obligate carnivores and eat meat and fish.\n",
    "space.md": "# Space\n\n## Moons\n\nJupiter has dozens of moons, including Ganymede, the largest moon in the solar system.\n",
}
SLEEP_Q = "How many hours do cats sleep every day?"


@pytest.fixture
def env(tmp_path, embedder, llm_servers):
    cfg = Config(data_dir=tmp_path / "idx")
    cfg.retrieval.min_vector_score = 0.2
    cfg.answer.max_steps = 3
    index = Index.open(cfg, embedder)
    d = tmp_path / "docs"
    d.mkdir()
    for name, text in DOCS.items():
        (d / name).write_text(text)
    ingest_path(index, d, cfg)

    def build(responder, **endpoint_kw) -> tuple[Answerer, object]:
        srv = llm_servers(responder)
        client = ChatClient([EndpointConfig(name="fake", base_url=srv.base_url, model="m", **endpoint_kw)], max_tokens=256)
        return Answerer(index, Retriever(index, cfg.retrieval), client, cfg), srv

    yield build, index, cfg
    index.close()


def events(answerer, q, **kw):
    return list(answerer.stream(q, **kw))


# ------------------------------------------------------------------------ pipeline

def test_pipeline_streams_tokens_then_citations_with_document_page_and_text(env):
    build, index, _ = env
    answerer, srv = build(Reply(content="Cats sleep for up to sixteen hours a day [1]."))
    evs = events(answerer, SLEEP_Q)
    kinds = [e["type"] for e in evs]
    assert kinds[0] == "retrieval" and kinds[-1] == "done" and kinds[-2] == "citations"
    assert kinds.count("token") > 3                       # streamed in pieces, not one blob
    assert "".join(e["text"] for e in evs if e["type"] == "token") == "Cats sleep for up to sixteen hours a day [1]."
    cit = evs[-2]["citations"]
    assert [c["n"] for c in cit] == [1]
    assert cit[0]["document"] == "Cats" and cit[0]["heading"] == "Sleep" and "sixteen hours" in cit[0]["text"]
    assert cit[0]["file"] == "cats.md" and "id" in cit[0]
    assert evs[-1]["mode"] == "pipeline" and not evs[-1]["no_answer"] and evs[-1]["endpoint"] == "fake"
    body = srv.requests[0]
    assert body["stream"] is True and "tools" not in body
    user = body["messages"][-1]["content"]
    assert user.startswith("Passages:") and "[1] Source: cats.md" in user and user.rstrip().endswith(f"Question: {SLEEP_Q}")
    assert "[1] Source: cats.md, Sleep" in user or "Sleep" in user


def test_citation_to_a_passage_that_was_not_given_is_flagged_and_removed(env):
    build, _, _ = env
    answerer, _ = build(Reply(content="They sleep sixteen hours [1]. Jupiter has 95 moons [42]."))
    ans = answerer.ask(SLEEP_Q)
    assert ans.text == "They sleep sixteen hours [1]. Jupiter has 95 moons."
    assert ans.flagged == [42] and [c["n"] for c in ans.citations] == [1]
    given = {p["n"] for p in ans.passages}
    assert 42 not in given and {c["n"] for c in ans.citations} <= given


def test_answer_without_any_citation_is_marked_uncited(env):
    build, _, _ = env
    answerer, _ = build(Reply(content="Sixteen hours."))
    ans = answerer.ask(SLEEP_Q)
    assert ans.uncited and ans.citations == []


def test_nothing_relevant_says_so_without_calling_the_model(env):
    build, _, cfg = env
    answerer, srv = build(Reply(content="should never be used"))
    ans = answerer.ask("zebra quantum mortgage derivatives")
    assert ans.no_answer and ans.text == cfg.answer.no_answer_message
    assert ans.citations == [] and srv.requests == []


def test_pipeline_drops_lowest_ranked_passages_to_fit_a_small_context(env):
    build, _, cfg = env
    q = "meat fish sleep moons Jupiter cats"
    a1, srv1 = build(Reply(content="ok [1]"), context_tokens=100000)
    a1.ask(q, top_k=3)
    msgs = srv1.requests[0]["messages"]
    full_tokens = estimate_tokens(msgs)
    assert msgs[-1]["content"].count("Source:") == 3

    # a window that holds the reply reserve (256) plus slightly less than the full prompt
    a2, srv2 = build(Reply(content="ok [1]"), context_tokens=256 + full_tokens - 20)
    ans = a2.ask(q, top_k=3)
    assert any("left out to fit" in n for n in ans.notices)
    sent = srv2.requests[0]["messages"][-1]["content"]
    assert 1 <= sent.count("Source:") < 3

    a3, srv3 = build(Reply(content="never"), context_tokens=200)         # not even one passage fits
    with pytest.raises(PromptTooLong):
        a3.ask(SLEEP_Q)
    assert srv3.requests == []


# ------------------------------------------------------------------------ agent

def agent_script(*turns):
    """Responder that plays `turns` in order, keyed by how many assistant turns the conversation has."""
    def respond(body):
        done = sum(1 for m in body["messages"] if m["role"] == "assistant")
        return turns[min(done, len(turns) - 1)]
    return respond


def call(name, **args):
    return {"name": name, "arguments": args}


def test_agent_uses_tools_then_answers_with_checked_citations(env):
    build, _, _ = env
    answerer, srv = build(agent_script(
        Reply(tool_calls=[call("search_documents", query="cats sleep hours")]),
        Reply(tool_calls=[call("read_passage", passage=1)]),
        Reply(content="Cats sleep up to sixteen hours [1]. Moons [8]."),
    ))
    evs = events(answerer, SLEEP_Q, mode="agent")
    tools = [e for e in evs if e["type"] == "tool"]
    assert [t["name"] for t in tools] == ["search_documents", "read_passage"]
    done = evs[-1]
    assert done["mode"] == "agent" and done["fallback"] is None and len(done["steps"]) == 2
    text = "".join(e["text"] for e in evs if e["type"] == "token")
    assert text == "Cats sleep up to sixteen hours [1]. Moons."
    cit = next(e for e in evs if e["type"] == "citations")
    assert [c["n"] for c in cit["citations"]] == [1] and cit["flagged"] == [8]
    assert "sixteen hours" in cit["citations"][0]["text"]
    # the tool schemas were sent, and tool results came back to the model as role=tool messages
    assert [t["function"]["name"] for t in srv.requests[0]["tools"]] == ["search_documents", "read_passage", "related_entities"]
    tool_msgs = [m for m in srv.requests[-1]["messages"] if m["role"] == "tool"]
    assert "Found" in tool_msgs[0]["content"] and "sixteen hours" in tool_msgs[1]["content"]


def test_agent_related_entities_without_a_graph_explains_itself(env):
    build, _, _ = env
    answerer, srv = build(agent_script(
        Reply(tool_calls=[call("related_entities", entity="Jupiter")]),
        Reply(tool_calls=[call("search_documents", query="jupiter moons")]),
        Reply(content="Jupiter has dozens of moons [1]."),
    ))
    ans = answerer.ask("How many moons does Jupiter have?", mode="agent")
    tool_msgs = [m["content"] for m in srv.requests[-1]["messages"] if m["role"] == "tool"]
    assert "knowledge graph is empty" in tool_msgs[0]
    assert ans.mode == "agent" and ans.citations[0]["document"] == "Space"


def test_agent_step_cap_forces_a_final_answer_without_tools(env):
    build, _, cfg = env
    answerer, srv = build(agent_script(
        Reply(tool_calls=[call("search_documents", query="cats")]),
        Reply(tool_calls=[call("search_documents", query="cats again")]),
        Reply(tool_calls=[call("search_documents", query="and again")]),
        Reply(content="Based on what I found, cats sleep a lot [1]."),
    ))
    ans = answerer.ask(SLEEP_Q, mode="agent")
    assert len(srv.requests) == cfg.answer.max_steps + 1
    assert "tools" in srv.requests[cfg.answer.max_steps - 1] and "tools" not in srv.requests[-1]
    assert ans.mode == "agent" and ans.text.startswith("Based on what I found") and len(ans.steps) == 3


def test_agent_text_written_before_a_tool_call_is_retracted(env):
    build, _, _ = env
    answerer, _ = build(agent_script(
        Reply(content="Let me look that up.", tool_calls=[call("search_documents", query="cats sleep")]),
        Reply(content="Sixteen hours [1]."),
    ))
    evs = events(answerer, SLEEP_Q, mode="agent")
    kinds = [e["type"] for e in evs]
    assert "retract" in kinds
    i = kinds.index("retract")
    pre = "".join(e["text"] for e in evs[:i] if e["type"] == "token")
    post = "".join(e["text"] for e in evs[i:] if e["type"] == "token")
    assert pre == "Let me look that up." and post == "Sixteen hours [1]."


def test_agent_with_no_relevant_passages_may_answer_that_it_found_nothing(env):
    build, _, _ = env
    answerer, _ = build(agent_script(
        Reply(tool_calls=[call("search_documents", query="zebra quantum mortgage")]),
        Reply(content="The documents do not contain this."),
    ))
    ans = answerer.ask("zebra quantum mortgage", mode="agent")
    assert ans.mode == "agent" and ans.text == "The documents do not contain this." and not ans.uncited


# ------------------------------------------------------------------------ agent -> pipeline fallback

def tools_rejected(body):
    if body.get("tools"):
        return Reply(status=400, body='{"error": {"message": "this model does not support tools"}}')
    return Reply(content="Sixteen hours [1].")


def test_endpoint_that_rejects_tools_falls_back_to_pipeline(env):
    build, _, _ = env
    answerer, srv = build(tools_rejected)
    evs = events(answerer, SLEEP_Q, mode="agent")
    assert any(e["type"] == "notice" and "pipeline mode" in e["message"] for e in evs)
    done = evs[-1]
    assert done["mode"] == "pipeline" and done["requested_mode"] == "agent" and "ToolsUnsupported" in done["fallback"]
    ans = answerer.ask(SLEEP_Q, mode="agent")
    assert ans.text == "Sixteen hours [1]." and ans.citations and ans.fallback_reason
    assert srv.requests[0].get("tools") and not srv.requests[1].get("tools")


@pytest.mark.parametrize("bad", [
    Reply(tool_calls=[{"name": "search_documents", "arguments": "{not json"}]),
    Reply(tool_calls=[{"name": "delete_everything", "arguments": {}}]),
    Reply(tool_calls=[{"name": "search_documents", "arguments": {}}]),                       # missing required argument
    Reply(tool_calls=[{"name": "read_passage", "arguments": {"passage": "first"}}]),         # wrong type
    Reply(tool_calls=[{"name": "search_documents", "arguments": "[1, 2]"}]),                 # not an object
])
def test_malformed_tool_calls_fall_back_to_pipeline(env, bad):
    build, _, _ = env

    def respond(body):
        return bad if body.get("tools") else Reply(content="Pipeline answer [1].")

    answerer, _ = build(respond)
    ans = answerer.ask(SLEEP_Q, mode="agent")
    assert ans.mode == "pipeline" and ans.requested_mode == "agent"
    assert "MalformedToolCall" in ans.fallback_reason
    assert ans.text == "Pipeline answer [1]." and ans.citations


def test_agent_that_never_searches_falls_back_and_retracts_its_text(env):
    build, _, _ = env

    def respond(body):
        return Reply(content="From memory: 8 hours.") if body.get("tools") else Reply(content="Grounded answer [1].")

    answerer, _ = build(respond)
    evs = events(answerer, SLEEP_Q, mode="agent")
    kinds = [e["type"] for e in evs]
    assert kinds.index("retract") < len(kinds) - 1
    assert evs[-1]["mode"] == "pipeline" and "AgentDidNotSearch" in evs[-1]["fallback"]
    final = "".join(e["text"] for e in evs[max(i for i, k in enumerate(kinds) if k == "retract"):] if e["type"] == "token")
    assert final == "Grounded answer [1]."


def test_unknown_mode_is_rejected(env):
    build, _, _ = env
    answerer, _ = build(Reply(content="x"))
    with pytest.raises(ValueError):
        answerer.ask("q", mode="magic")
