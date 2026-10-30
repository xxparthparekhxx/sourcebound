"""OpenAI-compatible chat client with an ordered list of endpoints.

Endpoints are tried in order. A connection error, timeout, HTTP 408/429 or 5xx moves on to the
next endpoint; any other HTTP error is the caller's problem and is raised straight away.
A streamed response can only fall back before its first delta: once text has been handed to
the caller, a failure is reported instead of silently restarting somewhere else.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import httpx

from .config import EndpointConfig
from .textutil import ThinkFilter, strip_think

log = logging.getLogger("docgraph.llm")

RETRYABLE_STATUS = frozenset({408, 429}) | frozenset(range(500, 600))


class ChatError(Exception):
    """A chat request failed."""


class NoUsableEndpoint(ChatError):
    """No endpoint is configured, or all configured ones are missing their API key."""


class AllEndpointsFailed(ChatError):
    def __init__(self, attempts: list["Attempt"]):
        self.attempts = attempts
        super().__init__("all chat endpoints failed: " + "; ".join(a.describe() for a in attempts))


class EndpointRejected(ChatError):
    """The endpoint answered with a non-retryable HTTP error (bad key, unknown model, bad request)."""

    def __init__(self, endpoint: str, status: int, body: str):
        self.endpoint, self.status, self.body = endpoint, status, body
        super().__init__(f"{endpoint} returned HTTP {status}: {body[:300]}")


class PromptTooLong(ChatError):
    """The prompt would not fit in the context window declared for any usable endpoint."""


class ToolsUnsupported(ChatError):
    """No endpoint accepted a request that carried tool definitions."""


class MalformedToolCall(ChatError):
    """The model produced a tool call that cannot be executed."""


@dataclass
class Attempt:
    endpoint: str
    error: str
    status: int | None = None

    def describe(self) -> str:
        return f"{self.endpoint}: {self.error}"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as sent by the model

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments) if self.arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise MalformedToolCall(f"tool call {self.name!r} has invalid JSON arguments: {e}") from e
        if not isinstance(value, dict):
            raise MalformedToolCall(f"tool call {self.name!r} arguments are not a JSON object")
        return value


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    endpoint: str = ""
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def assistant_message(self) -> dict[str, Any]:
        """The message to append to the conversation before sending tool results."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


IMAGE_TOKEN_ESTIMATE = 1200


def estimate_tokens(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None) -> int:
    """Rough upper-leaning token count (3.5 characters per token; a flat cost per image)."""
    chars = 0
    images = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    chars += len(str(part.get("text", "")))
                elif isinstance(part, Mapping) and "image" in str(part.get("type", "")):
                    images += 1
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(tc))
        chars += 16  # role / framing overhead
    if tools:
        chars += len(json.dumps(list(tools)))
    return int(chars / 3.5) + 1 + images * IMAGE_TOKEN_ESTIMATE


@dataclass
class TextDelta:
    text: str


@dataclass
class Final:
    result: ChatResult


class _Retry(Exception):
    """Internal: this endpoint failed in a way that justifies trying the next one."""

    def __init__(self, attempt: Attempt, wait: float = 0.0):
        self.attempt, self.wait = attempt, wait


class _TooLong(Exception):
    def __init__(self, attempt: Attempt):
        self.attempt = attempt


class _ToolsRejected(Exception):
    def __init__(self, attempt: Attempt):
        self.attempt = attempt


class ChatClient:
    def __init__(
        self,
        endpoints: Sequence[EndpointConfig],
        *,
        timeout: float = 120.0,
        max_tokens: int = 1024,
        passes: int = 1,
        backoff: float = 2.0,
        env: Mapping[str, str] | None = None,
        http: httpx.Client | None = None,
    ):
        self.endpoints = list(endpoints)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.passes = max(1, passes)
        self.backoff = backoff
        self.env = env
        self._http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))
        self.request_count = 0  # HTTP requests sent, including ones that failed

    # ------------------------------------------------------------------ public API

    def usable_endpoints(self, only: str | None = None) -> list[EndpointConfig]:
        eps = [e for e in self.endpoints if e.available(self.env)]
        if only:
            eps = [e for e in eps if e.name == only]
        return eps

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        params: Mapping[str, Any] | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        for event in self._run(messages, tools, params, endpoint, model, stream=False):
            if isinstance(event, Final):
                return event.result
        raise ChatError("no result")  # pragma: no cover

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        params: Mapping[str, Any] | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> Iterator[TextDelta | Final]:
        yield from self._run(messages, tools, params, endpoint, model, stream=True)

    def max_input_tokens(self, only: str | None = None) -> int | None:
        """Smallest declared prompt budget (context window minus the reply reserve) over usable endpoints."""
        budgets = [e.context_tokens - self.max_tokens for e in self.usable_endpoints(only) if e.context_tokens]
        return min(budgets) if budgets else None

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ fallback loop

    def _run(self, messages, tools, params, only, model, *, stream: bool) -> Iterator[TextDelta | Final]:
        usable = self.usable_endpoints(only)
        if not usable:
            names = [e.api_key_env for e in self.endpoints if e.api_key_env and not e.available(self.env)]
            hint = f" (set {', '.join(sorted(set(names)))})" if names else ""
            raise NoUsableEndpoint("no usable chat endpoint configured" + hint)

        attempts: list[Attempt] = []
        tools_rejected: list[Attempt] = []
        too_long: list[Attempt] = []
        candidates = list(usable)
        for n_pass in range(self.passes):
            wait = 0.0
            failed = False  # a retryable failure happened in this pass
            for ep in list(candidates):
                try:
                    yield from self._one(ep, messages, tools, params, model, stream)
                    return
                except _Retry as r:
                    log.warning("endpoint %s failed: %s", ep.label, r.attempt.error)
                    attempts.append(r.attempt)
                    wait = max(wait, r.wait)
                    failed = True
                except _ToolsRejected as r:  # a capability mismatch, not an outage: do not retry this endpoint
                    log.warning("endpoint %s rejected tools: %s", ep.label, r.attempt.error)
                    tools_rejected.append(r.attempt)
                    candidates.remove(ep)
                except _TooLong as r:
                    log.warning("endpoint %s skipped: %s", ep.label, r.attempt.error)
                    too_long.append(r.attempt)
                    candidates.remove(ep)
            if not failed or not candidates:
                break
            if n_pass + 1 < self.passes:
                time.sleep(min(max(wait, self.backoff * (n_pass + 1)), 30.0))
        if tools_rejected and tools:  # the caller can retry without tools, which tries every endpoint again
            raise ToolsUnsupported("; ".join(a.describe() for a in tools_rejected))
        if attempts:
            raise AllEndpointsFailed(attempts)
        if too_long:
            raise PromptTooLong("; ".join(a.describe() for a in too_long))
        raise AllEndpointsFailed(tools_rejected)

    # ------------------------------------------------------------------ one endpoint

    def _body(self, ep: EndpointConfig, messages, tools, params, model, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model or ep.model, "messages": list(messages), "max_tokens": self.max_tokens}
        if params:
            body.update(params)
        body.update(ep.options)  # endpoint settings win over per-call defaults
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        body["stream"] = stream
        return body

    def _headers(self, ep: EndpointConfig) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = ep.api_key(self.env)
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _one(self, ep: EndpointConfig, messages, tools, params, model, stream: bool) -> Iterator[TextDelta | Final]:
        url = ep.base_url.rstrip("/") + "/chat/completions"
        body = self._body(ep, messages, tools, params, model, stream)
        used_model = body["model"]
        if ep.context_tokens:
            need = estimate_tokens(messages, tools) + int(body.get("max_tokens") or 0)
            if need > ep.context_tokens:
                raise _TooLong(
                    Attempt(ep.label, f"prompt of about {need} tokens (including the reply reserve) does not fit "
                                      f"the declared context of {ep.context_tokens}")
                )
        self.request_count += 1
        try:
            if not stream:
                resp = self._http.post(url, json=body, headers=self._headers(ep))
                self._check_status(ep, resp.status_code, resp.text, resp.headers, bool(tools))
                result = self._parse_completion(resp.text, ep, used_model)
                yield Final(result)
                return
            with self._http.stream("POST", url, json=body, headers=self._headers(ep)) as resp:
                if resp.status_code != 200:
                    resp.read()
                    self._check_status(ep, resp.status_code, resp.text, resp.headers, bool(tools))
                yield from self._consume_stream(resp, ep, used_model)
        except httpx.TransportError as e:
            raise _Retry(Attempt(ep.label, f"{type(e).__name__}: {e}")) from e

    def _check_status(self, ep: EndpointConfig, status: int, text: str, headers, with_tools: bool) -> None:
        if status < 400:
            return
        attempt = Attempt(ep.label, f"HTTP {status}: {text[:200]}", status)
        if status in RETRYABLE_STATUS:
            wait = 0.0
            try:
                wait = float(headers.get("retry-after", 0))
            except (TypeError, ValueError):
                pass
            raise _Retry(attempt, wait)
        if with_tools and status in (400, 404, 405, 415, 422, 501):
            raise _ToolsRejected(attempt)
        raise EndpointRejected(ep.label, status, text)

    @staticmethod
    def _error_in_payload(payload: Mapping[str, Any]) -> tuple[int | None, str] | None:
        err = payload.get("error")
        if not err:
            return None
        if isinstance(err, Mapping):
            code = err.get("code")
            try:
                code = int(code) if code is not None else None
            except (TypeError, ValueError):
                code = None
            return code, str(err.get("message") or err)
        return None, str(err)

    def _parse_completion(self, text: str, ep: EndpointConfig, model: str) -> ChatResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise _Retry(Attempt(ep.label, f"response is not JSON: {text[:120]!r}")) from e
        err = self._error_in_payload(data) if isinstance(data, dict) else None
        if err:
            code, msg = err
            if code is None or code in RETRYABLE_STATUS:
                raise _Retry(Attempt(ep.label, f"error in response: {msg[:200]}", code))
            raise EndpointRejected(ep.label, code, msg)
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise _Retry(Attempt(ep.label, "response has no choices")) from e
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        return ChatResult(
            content=strip_think(msg.get("content") or ""),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            endpoint=ep.label,
            model=data.get("model") or model,
            usage=data.get("usage") or {},
        )

    def _consume_stream(self, resp: httpx.Response, ep: EndpointConfig, model: str) -> Iterator[TextDelta | Final]:
        think = ThinkFilter()
        content: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        committed = False
        non_sse: list[str] = []
        used_model = model

        try:
            for line in resp.iter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    non_sse.append(line)
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                err = self._error_in_payload(chunk) if isinstance(chunk, dict) else None
                if err:
                    code, msg = err
                    if committed:
                        raise ChatError(f"{ep.label}: stream failed after output started: {msg[:200]}")
                    if code is None or code in RETRYABLE_STATUS:
                        raise _Retry(Attempt(ep.label, f"error in stream: {msg[:200]}", code))
                    raise EndpointRejected(ep.label, code, msg)
                used_model = chunk.get("model") or used_model
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        committed = True
                        content.append(piece)
                        shown = think.feed(piece)
                        if shown:
                            yield TextDelta(shown)
                    for tc in delta.get("tool_calls") or []:
                        committed = True
                        slot = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            a = fn["arguments"]
                            slot["arguments"] += a if isinstance(a, str) else json.dumps(a)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        except httpx.TransportError as e:
            if committed:
                raise ChatError(f"{ep.label}: connection lost mid-stream: {e}") from e
            raise _Retry(Attempt(ep.label, f"{type(e).__name__}: {e}")) from e

        if not committed and not content and not calls:
            # Some servers ignore stream=true and return one JSON document.
            if non_sse:
                result = self._parse_completion("\n".join(non_sse), ep, used_model)
                if result.content:
                    yield TextDelta(result.content)
                yield Final(result)
                return
            raise _Retry(Attempt(ep.label, "empty stream"))

        tail = think.flush()
        if tail:
            yield TextDelta(tail)
        tool_calls = [
            ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"]) for i, (_, c) in enumerate(sorted(calls.items()))
        ]
        yield Final(
            ChatResult(
                content=strip_think("".join(content)),
                tool_calls=tool_calls,
                finish_reason=finish,
                endpoint=ep.label,
                model=used_model,
            )
        )
