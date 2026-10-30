"""FastAPI app: POST /ask (SSE), GET /passages/{id}, GET /healthz, and the single-page UI at /."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Config, load_config
from .runtime import Runtime

log = logging.getLogger("docgraph.api")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["pipeline", "agent"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    # Per-request switches for the retrieval stages (None = use the configured value)
    vector: bool | None = None
    bm25: bool | None = None
    mmr: bool | None = None
    graph: bool | None = None
    rerank: bool | None = None


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    cfg: Config | None = None,
    *,
    runtime: Runtime | None = None,
    ask_dependencies: Sequence[Any] = (),
) -> FastAPI:
    """Build the app. `ask_dependencies` are FastAPI dependencies run before every /ask request,
    which is where per-IP limits or a bot check can be added without touching the handlers."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is None:
            app.state.rt = Runtime(cfg or load_config())
        else:
            app.state.rt = runtime
        yield
        if runtime is None:
            app.state.rt.close()

    app = FastAPI(title="docgraph", version=__version__, lifespan=lifespan)

    def rt() -> Runtime:
        return app.state.rt

    @app.get("/healthz")
    def healthz() -> dict:
        r = rt()
        stats = r.index.stats()
        return {
            "status": "ok",
            "version": __version__,
            **stats,
            "embedding": r.index.embedding_info(),
            "chat_endpoints": [
                {"name": e.name or e.base_url, "model": e.model, "usable": e.available(r.env)}
                for e in r.cfg.chat.endpoints
            ],
        }

    @app.get("/passages/{passage_id}")
    def get_passage(passage_id: int) -> dict:
        index = rt().index
        p = index.get_passage(passage_id)
        if p is None:
            raise HTTPException(404, f"no passage {passage_id}")
        sibling = index.conn.execute(
            "SELECT (SELECT id FROM chunks WHERE doc_id = ? AND ord = ?) AS prev, "
            "(SELECT id FROM chunks WHERE doc_id = ? AND ord = ?) AS next",
            (p.doc_id, p.ord - 1, p.doc_id, p.ord + 1),
        ).fetchone()
        return {**p.to_dict(), "location": p.location, "prev_id": sibling["prev"], "next_id": sibling["next"]}

    @app.post("/ask", dependencies=[Depends(d) for d in ask_dependencies])
    def ask(req: AskRequest) -> StreamingResponse:
        r = rt()
        overrides = {k: v for k, v in req.model_dump(exclude={"question", "mode"}).items() if v is not None}

        def events() -> Iterator[str]:
            if r.index.count("chunks") == 0:
                yield sse("error", {"message": "The index is empty. Run `docgraph ingest <path>` first."})
                return
            try:
                for ev in r.answerer.stream(req.question, mode=req.mode, **overrides):
                    yield sse(ev["type"], ev)
            except Exception as e:  # noqa: BLE001 - the stream is already open: report in-band
                log.exception("ask failed")
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    return app
