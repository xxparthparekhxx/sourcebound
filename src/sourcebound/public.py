"""The public demo app: `sourcebound serve --public`.

Meant to sit behind a gateway that does the rate limiting and the bot check. This layer assumes
nothing about the network in front of it:

* every route except ``GET /healthz`` needs the ``X-Gateway-Secret`` header to equal the secret in the
  environment variable named by ``[serve].gateway_secret_env``; otherwise 401 with an empty-ish body.
  The app refuses to start without that variable. ``/healthz`` answers ``{"status": "ok"}`` and nothing else.
* ``POST /ask`` takes one field, ``question`` (at most ``[serve].max_question_chars`` characters). Retrieval
  settings, mode and top-k are the server's; the client cannot change them.
* at most ``[serve].max_concurrent`` questions are answered at once; the next gets 503 and ``Retry-After``.
* the answer streams as server-sent events: ``retrieval``, ``token``*, ``citations`` (with each cited
  passage's text), ``faithfulness`` (per sentence, openjev-flash), ``done`` (with timings). Failures arrive
  as one ``error`` event with a generic message; details go to the server log.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator, Mapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .config import Config, ConfigError, load_config
from .faithfulness import check_answer
from .llm import AllEndpointsFailed, ChatError, NoUsableEndpoint
from .runtime import Runtime

log = logging.getLogger("sourcebound.public")

OPEN_PATHS = {"/healthz"}


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class GatewayAuth:
    """Pure ASGI middleware: constant-time check of X-Gateway-Secret on every HTTP route but /healthz."""

    def __init__(self, app: ASGIApp, secret: str):
        self.app, self.secret = app, secret.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] not in OPEN_PATHS:
            sent = dict(scope["headers"]).get(b"x-gateway-secret", b"")
            if not hmac.compare_digest(sent, self.secret):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


class Slots:
    """Non-blocking counter of requests in flight; a slot is released at most once."""

    def __init__(self, n: int):
        self._sem = threading.BoundedSemaphore(n)

    def take(self) -> "Slot | None":
        return Slot(self._sem) if self._sem.acquire(blocking=False) else None


class Slot:
    def __init__(self, sem: threading.BoundedSemaphore):
        self._sem, self._lock, self._held = sem, threading.Lock(), True

    def release(self) -> None:
        with self._lock:
            if self._held:
                self._held = False
                self._sem.release()


class PublicAsk(BaseModel):
    """The only thing a client may send. The length limit is [serve].max_question_chars, checked in the handler."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)


def friendly_error(e: Exception) -> str:
    if isinstance(e, (AllEndpointsFailed, NoUsableEndpoint)):
        return "The free language-model providers are rate limited or unavailable right now. Try again in a minute."
    if isinstance(e, ChatError):
        return "The language model could not answer this question."
    return "Something went wrong while answering."


def public_citation(c: Mapping[str, Any]) -> dict:
    return {k: c.get(k) for k in ("n", "id", "document", "file", "heading", "text")}


def create_public_app(cfg: Config | None = None, *, runtime: Runtime | None = None, env: Mapping[str, str] | None = None) -> FastAPI:
    cfg = cfg or (runtime.cfg if runtime else load_config())
    env = os.environ if env is None else env
    scfg = cfg.serve
    secret = env.get(scfg.gateway_secret_env) or ""
    if len(secret) < 16:
        raise ConfigError(f"set {scfg.gateway_secret_env} (at least 16 characters): the public mode refuses to run without a gateway secret")
    slots = Slots(scfg.max_concurrent)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rt = runtime or Runtime(cfg, env=env)
        warm_up(app.state.rt, scfg.faithfulness)
        yield
        if runtime is None:
            app.state.rt.close()

    app = FastAPI(title="sourcebound", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ask")
    def ask(req: PublicAsk, request: Request):
        if len(req.question) > scfg.max_question_chars:
            return JSONResponse({"error": f"the question is longer than {scfg.max_question_chars} characters"}, status_code=422)
        slot = slots.take()
        if slot is None:
            return JSONResponse({"error": "busy"}, status_code=503, headers={"Retry-After": "5"})
        r: Runtime = request.app.state.rt
        question = req.question.strip()

        def events() -> Iterator[str]:
            try:
                yield from answer_events(r, question, scfg.faithfulness)
            finally:
                slot.release()

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(slot.release),  # also releases when the stream never started or was cut
        )

    app.add_middleware(GatewayAuth, secret=secret)
    return app


def warm_up(rt: Runtime, faithfulness: bool) -> None:
    """Load everything lazy before the first request, so the first visitor does not pay for it."""
    try:
        rt.index.collection.count()
        if rt.index.embedder:
            rt.index.embedder.embed_query("warm up")
        if faithfulness or rt.cfg.retrieval.rerank:
            rt.reranker.scorer.predict([("warm up premise", "warm up hypothesis")])
    except Exception:  # noqa: BLE001 - the first request will report what is wrong
        log.exception("warm-up failed")


def answer_events(rt: Runtime, question: str, faithfulness: bool) -> Iterator[str]:
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    text: list[str] = []
    done: dict | None = None
    cited: dict[int, str] = {}
    try:
        for ev in rt.answerer.stream(question, mode="pipeline"):
            kind = ev["type"]
            if kind == "retrieval":
                marks["retrieval"] = time.perf_counter()
                yield sse("retrieval", {"count": ev["count"], "relevant": ev["relevant"]})
            elif kind == "token":
                text.append(ev["text"])
                yield sse("token", {"text": ev["text"]})
            elif kind == "citations":
                marks["llm"] = time.perf_counter()
                cits = [public_citation(c) for c in ev["citations"]]
                cited = {c["n"]: c["text"] for c in cits}
                yield sse("citations", {"citations": cits, "flagged": ev["flagged"]})
            elif kind == "done":
                done = ev
            # notice / retract only occur in agent mode
        if done is None:
            raise ChatError("the stream ended without a result")
        t_ret = marks.get("retrieval", t0)
        t_llm = marks.get("llm", t_ret)
        sentences: list[dict] = []
        if faithfulness and cited and not done["no_answer"]:
            sentences = check_answer(rt.reranker.scorer, "".join(text), cited, rt.cfg.rerank.entailment_threshold)
        t_end = time.perf_counter()
        yield sse("faithfulness", {"sentences": sentences, "threshold": rt.cfg.rerank.entailment_threshold})
        yield sse("done", {
            "no_answer": done["no_answer"], "uncited": done["uncited"], "endpoint": done["endpoint"], "model": done["model"],
            "timings": {"retrieval_s": round(t_ret - t0, 3), "llm_s": round(t_llm - t_ret, 3),
                        "faithfulness_s": round(t_end - t_llm, 3), "total_s": round(t_end - t0, 3)},
        })
    except Exception as e:  # noqa: BLE001 - the stream is already open: report in-band
        log.exception("ask failed")
        yield sse("error", {"message": friendly_error(e)})
