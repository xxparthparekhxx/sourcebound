"""Answering: `pipeline` (one LLM call over retrieved passages) and `agent` (tool calling).

Both stream, both check citations, and both say so instead of answering when retrieval found
nothing relevant. Agent mode falls back to pipeline mode when the endpoint rejects tools or the
model produces a tool call that cannot be executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .agent import AgentDidNotSearch, PassageSet, Tools, run_agent
from .citations import CitationFilter
from .config import Config
from .llm import ChatClient, Final, MalformedToolCall, PromptTooLong, TextDelta, ToolsUnsupported, estimate_tokens
from .models import Passage
from .prompts import PIPELINE_SYSTEM, pipeline_user_message
from .retrieval import Retriever
from .store import Index

Event = dict[str, Any]
FALLBACK_ERRORS = (ToolsUnsupported, MalformedToolCall, PromptTooLong, AgentDidNotSearch)


@dataclass
class Answer:
    question: str
    text: str = ""
    mode: str = "pipeline"  # the mode that produced the answer
    requested_mode: str = "pipeline"
    fallback_reason: str | None = None
    no_answer: bool = False
    citations: list[dict] = field(default_factory=list)  # passages actually cited, in order of first use
    flagged: list[int] = field(default_factory=list)  # [n] markers removed because n was not a given passage
    passages: list[dict] = field(default_factory=list)  # every passage the model was shown (no text)
    uncited: bool = False  # answered, but cited nothing
    endpoint: str = ""
    model: str = ""
    retrieval_reason: str = ""
    steps: list[dict] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    truncated: str | None = None  # set when the server seems to have cut the prompt to its context length


def citation_dict(n: int, p: Passage, *, text: bool = True) -> dict:
    return {"n": n, **p.to_dict(text=text)}


class Answerer:
    def __init__(self, index: Index, retriever: Retriever, client: ChatClient, cfg: Config):
        self.index, self.retriever, self.client, self.cfg = index, retriever, client, cfg

    # ------------------------------------------------------------------ public

    def ask(self, question: str, **kw) -> Answer:
        """Run to completion and collect the events into an Answer."""
        ans = Answer(question)
        for ev in self.stream(question, **kw):
            t = ev["type"]
            if t == "token":
                ans.text += ev["text"]
            elif t == "retract":
                ans.text = ""
            elif t == "notice":
                ans.notices.append(ev["message"])
            elif t == "retrieval":
                ans.retrieval_reason = ev["reason"]
            elif t == "citations":
                ans.citations, ans.flagged, ans.passages = ev["citations"], ev["flagged"], ev["passages"]
            elif t == "done":
                ans.mode, ans.requested_mode = ev["mode"], ev["requested_mode"]
                ans.fallback_reason, ans.no_answer, ans.uncited = ev["fallback"], ev["no_answer"], ev["uncited"]
                ans.endpoint, ans.model, ans.steps = ev["endpoint"], ev["model"], ev["steps"]
                ans.truncated = ev.get("truncated")
        return ans

    def stream(
        self,
        question: str,
        *,
        mode: str | None = None,
        doc_ids: Sequence[int] | None = None,
        **retrieval: Any,
    ) -> Iterator[Event]:
        mode = mode or self.cfg.answer.mode
        if mode not in ("pipeline", "agent"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "pipeline":
            yield from self._pipeline(question, doc_ids, retrieval, requested=mode, fallback=None)
            return

        showing = False  # has the client been shown answer text that might need retracting?
        try:
            for ev in self._agent(question, doc_ids, retrieval):
                if ev["type"] == "token":
                    showing = True
                elif ev["type"] == "retract":
                    showing = False
                yield ev
        except FALLBACK_ERRORS as e:
            reason = f"{type(e).__name__}: {e}"
            yield {"type": "notice", "message": f"Agent mode was not usable ({reason}); answering in pipeline mode instead."}
            if showing:
                yield {"type": "retract"}
            yield from self._pipeline(question, doc_ids, retrieval, requested="agent", fallback=reason)

    # ------------------------------------------------------------------ pipeline

    def _fit(self, question: str, passages: list[Passage]) -> tuple[list[Passage], int]:
        """Drop the lowest-ranked passages until the prompt fits the smallest declared context window."""
        budget = self.client.max_input_tokens()
        if budget is None:
            return passages, 0
        kept = list(passages)
        while len(kept) > 1 and self._prompt_tokens(question, kept) > budget:
            kept.pop()
        return kept, len(passages) - len(kept)

    def _messages(self, question: str, passages: Sequence[Passage]) -> list[dict]:
        return [
            {"role": "system", "content": PIPELINE_SYSTEM},
            {"role": "user", "content": pipeline_user_message(question, passages, self.cfg.answer.max_passage_chars)},
        ]

    def _prompt_tokens(self, question: str, passages: Sequence[Passage]) -> int:
        return estimate_tokens(self._messages(question, passages))

    def _no_answer(self, requested: str, fallback: str | None, reason: str, mode: str) -> Iterator[Event]:
        yield {"type": "token", "text": self.cfg.answer.no_answer_message}
        yield {"type": "citations", "citations": [], "flagged": [], "passages": []}
        yield self._done(mode, requested, fallback, no_answer=True, uncited=False, info={}, steps=[], trimmed=0)

    def _done(self, mode, requested, fallback, *, no_answer, uncited, info, steps, trimmed) -> Event:
        return {
            "type": "done", "mode": mode, "requested_mode": requested, "fallback": fallback,
            "no_answer": no_answer, "uncited": uncited, "endpoint": info.get("endpoint", ""),
            "model": info.get("model", ""), "steps": steps, "trimmed": trimmed,
            "truncated": info.get("truncated"),
        }

    def _pipeline(self, question, doc_ids, retrieval, *, requested: str, fallback: str | None) -> Iterator[Event]:
        res = self.retriever.search(question, doc_ids=doc_ids, **retrieval)
        yield {"type": "retrieval", "count": len(res.passages), "relevant": res.relevant,
               "reason": res.reason, "stages": res.stages}
        if not res.relevant:
            yield from self._no_answer(requested, fallback, res.reason, "pipeline")
            return

        passages, dropped = self._fit(question, res.passages)
        if dropped:
            yield {"type": "notice", "message": f"{dropped} lower-ranked passage(s) left out to fit the model's context window."}
        shown = PassageSet()
        for p in passages:
            shown.add(p)
        flt = CitationFilter(shown.numbers.keys())
        parts: list[str] = []
        final = None
        for ev in self.client.stream(self._messages(question, passages), params={"temperature": 0.2}):
            if isinstance(ev, TextDelta):
                out = flt.feed(ev.text)
                if out:
                    parts.append(out)
                    yield {"type": "token", "text": out}
            elif isinstance(ev, Final):
                final = ev.result
        tail = flt.flush()
        if tail:
            parts.append(tail)
            yield {"type": "token", "text": tail}
        assert final is not None
        if final.truncated:
            yield {"type": "notice", "message": f"Warning: {final.truncated}"}
        info = {"endpoint": final.endpoint, "model": final.model, "truncated": final.truncated}
        yield from self._finish("pipeline", requested, fallback, shown, flt, info, [], dropped)

    def _finish(self, mode, requested, fallback, shown: PassageSet, flt: CitationFilter, info, steps, trimmed) -> Iterator[Event]:
        cited = [citation_dict(n, shown.numbers[n]) for n in flt.used]
        yield {
            "type": "citations", "citations": cited, "flagged": list(flt.flagged),
            "passages": [citation_dict(n, p, text=False) for n, p in sorted(shown.numbers.items())],
        }
        yield self._done(mode, requested, fallback, no_answer=False, uncited=(not flt.used and bool(shown.numbers)), info=info, steps=steps, trimmed=trimmed)

    # ------------------------------------------------------------------ agent

    def _agent(self, question, doc_ids, retrieval) -> Iterator[Event]:
        passages = PassageSet()
        tools = Tools(self.index, self.retriever, passages, doc_ids, retrieval)
        steps: list[dict] = []
        info: dict = {}
        yield from run_agent(self.client, tools, passages, question, self.cfg.answer.max_steps, steps, info)
        if not info.get("text"):  # the step cap was reached without finding anything
            yield from self._no_answer("agent", None, "agent found no passages", "agent")
            return
        flt = CitationFilter(passages.numbers.keys())
        flt.used, flt.flagged = info["used"], info["flagged"]
        yield from self._finish("agent", "agent", None, passages, flt, info, steps, 0)
