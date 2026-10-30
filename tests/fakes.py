"""A local fake OpenAI-compatible server and a deterministic embedder. No external network."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np


@dataclass
class Reply:
    status: int = 200
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # [{"name": ..., "arguments": dict | str}]
    body: str | None = None  # raw body for error statuses
    headers: dict[str, str] = field(default_factory=dict)


def _completion(reply: Reply, model: str) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"]),
                },
            }
            for i, tc in enumerate(reply.tool_calls)
        ]
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if reply.tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _sse(obj: Any) -> bytes:
    return b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"


def hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Bag-of-words hashing embedding: texts that share words have a higher cosine similarity."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeLLM:
    """`responder` is a Reply, a list of Replies (consumed in order, the last one repeats) or
    a callable(body) -> Reply. Every request is recorded in `.requests`."""

    def __init__(self, responder: Reply | list[Reply] | Callable[[dict], Reply] | None = None, *, name: str = "fake"):
        self.responder = responder if responder is not None else Reply(content="ok")
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.embed_requests: list[dict[str, Any]] = []
        self.embed_dim = 16
        self.name = name
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _read(self) -> dict:
                n = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(n) or b"{}")

            def do_POST(self):
                body = self._read()
                if self.path.endswith("/embeddings"):
                    outer.embed_requests.append(body)
                    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                    data = [
                        {"index": i, "embedding": hash_embedding(t, outer.embed_dim).tolist()}
                        for i, t in enumerate(inputs)
                    ]
                    return self._json(200, {"data": data, "model": body.get("model")})
                if not self.path.endswith("/chat/completions"):
                    return self._json(404, {"error": "not found"})
                outer.requests.append(body)
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                reply = outer._next(body)
                if reply.status != 200:
                    raw = (reply.body or json.dumps({"error": {"message": f"status {reply.status}"}})).encode()
                    self.send_response(reply.status)
                    for k, v in reply.headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if not body.get("stream"):
                    return self._json(200, _completion(reply, body.get("model", "fake")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                model = body.get("model", "fake")
                w = self.wfile
                w.write(b": keep-alive\n\n")
                if reply.content:
                    text = reply.content
                    for i in range(0, len(text), 4):
                        w.write(_sse({"model": model, "choices": [{"index": 0, "delta": {"content": text[i : i + 4]}}]}))
                for i, tc in enumerate(reply.tool_calls or []):
                    args = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
                    first = {"index": i, "id": f"call_{i}", "type": "function", "function": {"name": tc["name"], "arguments": ""}}
                    w.write(_sse({"model": model, "choices": [{"index": 0, "delta": {"tool_calls": [first]}}]}))
                    half = len(args) // 2
                    for part in (args[:half], args[half:]):
                        d = {"index": i, "function": {"arguments": part}}
                        w.write(_sse({"model": model, "choices": [{"index": 0, "delta": {"tool_calls": [d]}}]}))
                finish = "tool_calls" if reply.tool_calls else "stop"
                w.write(_sse({"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
                w.write(b"data: [DONE]\n\n")
                w.flush()

            def _json(self, status: int, obj: Any):
                raw = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def _next(self, body: dict) -> Reply:
        r = self.responder
        if callable(r):
            return r(body)
        if isinstance(r, list):
            i = len(self.requests) - 1
            return r[min(i, len(r) - 1)]
        return r

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class FakeEmbedder:
    provider = "fake"

    def __init__(self, model: str = "fake-hash", dim: int = 128):
        self.model = model
        self._dim = dim
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts):
        self.calls += 1
        if not len(texts):
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([hash_embedding(t, self._dim) for t in texts])

    def embed_query(self, text):
        return hash_embedding(text, self._dim)
