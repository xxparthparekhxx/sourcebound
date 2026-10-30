import pytest

from sourcebound.config import EndpointConfig
from sourcebound.embeddings import OpenAIEmbedder
from sourcebound.llm import (
    AllEndpointsFailed, ChatClient, EndpointRejected, Final, NoUsableEndpoint, PromptTooLong,
    TextDelta, ToolsUnsupported, estimate_tokens,
)
from fakes import Reply

MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]


def ep(srv, name, **kw):
    return EndpointConfig(name=name, base_url=srv.base_url, model=f"{name}-model", **kw)


def test_429_on_the_first_endpoint_moves_to_the_second(llm_servers):
    a = llm_servers(Reply(status=429))
    b = llm_servers(Reply(content="from b"))
    client = ChatClient([ep(a, "a"), ep(b, "b")])
    res = client.complete(MSGS)
    assert res.content == "from b" and res.endpoint == "b" and res.model == "b-model"
    assert len(a.requests) == 1 and len(b.requests) == 1
    assert a.requests[0]["model"] == "a-model" and b.requests[0]["model"] == "b-model"
    assert client.request_count == 2


def test_5xx_and_connection_errors_also_fall_back(llm_servers):
    dead = llm_servers(Reply())
    dead_url = dead.base_url
    dead.close()  # nothing listens here any more -> connection refused
    s500 = llm_servers(Reply(status=503))
    good = llm_servers(Reply(content="ok"))
    client = ChatClient([
        EndpointConfig(name="dead", base_url=dead_url, model="m"), ep(s500, "s500"), ep(good, "good"),
    ])
    assert client.complete(MSGS).endpoint == "good"
    assert len(s500.requests) == 1


def test_order_is_respected_when_the_first_endpoint_works(llm_servers):
    a = llm_servers(Reply(content="from a"))
    b = llm_servers(Reply(content="from b"))
    assert ChatClient([ep(a, "a"), ep(b, "b")]).complete(MSGS).content == "from a"
    assert b.requests == []


def test_other_4xx_errors_are_not_retried_elsewhere(llm_servers):
    a = llm_servers(Reply(status=401, body='{"error":"bad key"}'))
    b = llm_servers(Reply(content="never"))
    with pytest.raises(EndpointRejected) as e:
        ChatClient([ep(a, "a"), ep(b, "b")]).complete(MSGS)
    assert e.value.status == 401 and b.requests == []


def test_all_endpoints_failing_lists_every_attempt(llm_servers):
    a = llm_servers(Reply(status=429))
    b = llm_servers(Reply(status=500))
    with pytest.raises(AllEndpointsFailed) as e:
        ChatClient([ep(a, "a"), ep(b, "b")]).complete(MSGS)
    assert [x.endpoint for x in e.value.attempts] == ["a", "b"]


def test_endpoint_without_its_key_is_skipped_without_a_request(llm_servers):
    a = llm_servers(Reply(content="from a"))
    b = llm_servers(Reply(content="from b"))
    client = ChatClient([ep(a, "a", api_key_env="NEEDS_KEY"), ep(b, "b")], env={})
    assert client.complete(MSGS).endpoint == "b" and a.requests == []
    with pytest.raises(NoUsableEndpoint) as e:
        ChatClient([ep(a, "a", api_key_env="NEEDS_KEY")], env={}).complete(MSGS)
    assert "NEEDS_KEY" in str(e.value)


def test_bearer_key_is_read_from_env_and_options_are_sent(llm_servers):
    a = llm_servers(Reply(content="x"))
    client = ChatClient(
        [ep(a, "a", api_key_env="K", options={"reasoning_effort": "none", "temperature": 0.7})],
        env={"K": "sekret"},
    )
    client.complete(MSGS, params={"temperature": 0.0, "top_p": 0.5})
    assert a.headers[0]["authorization"] == "Bearer sekret"
    body = a.requests[0]
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.7  # endpoint options win over call defaults
    assert body["top_p"] == 0.5


def test_force_wins_over_endpoint_options_and_does_not_leak_between_calls(llm_servers):
    a = llm_servers(Reply(content="x"))
    client = ChatClient([ep(a, "a", options={"presence_penalty": 1.5, "temperature": 0.7})])
    client.complete(MSGS, force={"presence_penalty": 0.0})
    client.complete(MSGS)
    assert a.requests[0]["presence_penalty"] == 0.0 and a.requests[0]["temperature"] == 0.7
    assert a.requests[1]["presence_penalty"] == 1.5


def test_streaming_yields_deltas_then_final_and_strips_think(llm_servers):
    a = llm_servers(Reply(content="<think>secret reasoning</think>The answer is 42."))
    events = list(ChatClient([ep(a, "a")]).stream(MSGS))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "The answer is 42."
    assert isinstance(events[-1], Final) and events[-1].result.content == "The answer is 42."
    assert len([e for e in events if isinstance(e, TextDelta)]) > 2  # really streamed in pieces
    assert a.requests[0]["stream"] is True


def test_streaming_falls_back_before_the_first_delta(llm_servers):
    a = llm_servers(Reply(status=429))
    b = llm_servers(Reply(content="streamed from b"))
    events = list(ChatClient([ep(a, "a"), ep(b, "b")]).stream(MSGS))
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "streamed from b"
    assert events[-1].result.endpoint == "b"


def test_streamed_tool_calls_are_assembled(llm_servers):
    a = llm_servers(Reply(tool_calls=[{"name": "search_documents", "arguments": {"query": "graph rag"}}]))
    events = list(ChatClient([ep(a, "a")]).stream(MSGS, tools=TOOLS))
    final = events[-1].result
    assert [tc.name for tc in final.tool_calls] == ["search_documents"]
    assert final.tool_calls[0].parsed_arguments() == {"query": "graph rag"}
    assert a.requests[0]["tools"] == TOOLS


def test_non_streamed_tool_calls(llm_servers):
    a = llm_servers(Reply(tool_calls=[{"name": "t", "arguments": {"x": 1}}]))
    res = ChatClient([ep(a, "a")]).complete(MSGS, tools=TOOLS)
    assert res.tool_calls[0].parsed_arguments() == {"x": 1}
    assert res.assistant_message()["tool_calls"][0]["function"]["name"] == "t"


def test_endpoints_that_reject_tools_raise_tools_unsupported(llm_servers):
    reject = lambda body: Reply(status=400, body='{"error":"tools not supported"}') if body.get("tools") else Reply(content="plain")
    a = llm_servers(reject)
    b = llm_servers(reject)
    client = ChatClient([ep(a, "a"), ep(b, "b")])
    with pytest.raises(ToolsUnsupported):
        client.complete(MSGS, tools=TOOLS)
    assert len(a.requests) == 1 and len(b.requests) == 1  # each was asked once
    assert client.complete(MSGS).content == "plain"  # without tools the same endpoints work


def test_tool_rejection_on_one_endpoint_tries_the_next(llm_servers):
    a = llm_servers(Reply(status=404, body="no tool support"))
    b = llm_servers(Reply(tool_calls=[{"name": "t", "arguments": {}}]))
    res = ChatClient([ep(a, "a"), ep(b, "b")]).complete(MSGS, tools=TOOLS)
    assert res.endpoint == "b" and res.tool_calls


def test_prompt_that_exceeds_the_declared_context_is_never_sent(llm_servers):
    small = llm_servers(Reply(content="small"))
    big = llm_servers(Reply(content="big"))
    long_msgs = [{"role": "user", "content": "word " * 2000}]  # ~2900 tokens
    client = ChatClient([ep(small, "small", context_tokens=1024), ep(big, "big", context_tokens=16000)], max_tokens=256)
    assert client.complete(long_msgs).endpoint == "big"
    assert small.requests == []
    with pytest.raises(PromptTooLong):
        ChatClient([ep(small, "small", context_tokens=1024)], max_tokens=256).complete(long_msgs)
    assert small.requests == []


def test_max_input_tokens_uses_the_smallest_declared_window(llm_servers):
    a = llm_servers()
    c = ChatClient([ep(a, "a", context_tokens=4096), ep(a, "b", context_tokens=8192), ep(a, "c")], max_tokens=1000)
    assert c.max_input_tokens() == 3096
    assert ChatClient([ep(a, "c")]).max_input_tokens() is None


LONG = [{"role": "user", "content": "alpha beta gamma " * 400}]  # 6800 characters, about 1950 estimated tokens


@pytest.mark.parametrize("stream", [False, True])
def test_a_prompt_the_server_reports_as_cut_short_is_flagged(llm_servers, stream):
    # What Ollama does when the prompt exceeds its context: it drops the start and reports the shorter count.
    srv = llm_servers(Reply(content="answer", prompt_tokens=700))
    client = ChatClient([ep(srv, "o", context_tokens=3000)], max_tokens=256)
    if stream:
        res = [e for e in client.stream(LONG) if isinstance(e, Final)][0].result
    else:
        res = client.complete(LONG)
    assert res.truncated and "700 prompt tokens" in res.truncated and "OLLAMA_CONTEXT_LENGTH" in res.truncated


def test_a_plausible_prompt_token_count_is_not_flagged(llm_servers):
    srv = llm_servers(Reply(content="answer", prompt_tokens=1900))
    assert ChatClient([ep(srv, "o", context_tokens=3000)], max_tokens=256).complete(LONG).truncated is None


def test_truncation_is_not_checked_without_a_declared_context_or_for_short_prompts(llm_servers):
    srv = llm_servers(Reply(content="answer", prompt_tokens=3))
    assert ChatClient([ep(srv, "o")], max_tokens=256).complete(LONG).truncated is None
    assert ChatClient([ep(srv, "o", context_tokens=3000)], max_tokens=256).complete(MSGS).truncated is None


def test_usage_is_requested_in_streams_only_when_a_context_is_declared(llm_servers):
    srv = llm_servers(Reply(content="x"))
    list(ChatClient([ep(srv, "o", context_tokens=3000)]).stream(MSGS))
    list(ChatClient([ep(srv, "p")]).stream(MSGS))
    assert srv.requests[0].get("stream_options") == {"include_usage": True}
    assert "stream_options" not in srv.requests[1]


def test_estimate_tokens_counts_images():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x" * 350}, {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    assert estimate_tokens(msgs) > 1200


def test_passes_retry_after_a_rate_limit(llm_servers):
    a = llm_servers([Reply(status=429), Reply(content="second try")])
    client = ChatClient([ep(a, "a")], passes=2, backoff=0.01)
    assert client.complete(MSGS).content == "second try" and len(a.requests) == 2


def test_openai_embedder_uses_prefixes_and_probes_dimension(llm_servers):
    srv = llm_servers()
    srv.embed_dim = 24
    e = OpenAIEmbedder("nomic", srv.base_url, query_prefix="search_query: ", document_prefix="search_document: ", batch_size=2)
    assert e.dimension == 24
    docs = e.embed_documents(["a b", "c d", "e f"])
    assert docs.shape == (3, 24)
    assert [r["input"] for r in srv.embed_requests[-2:]] == [["search_document: a b", "search_document: c d"], ["search_document: e f"]]
    assert e.embed_query("q").shape == (24,)
    assert srv.embed_requests[-1]["input"] == ["search_query: q"]
