"""Agent mode: the model drives retrieval through native tool calling, with a step cap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from .citations import CitationFilter
from .llm import ChatClient, Final, MalformedToolCall, TextDelta, ToolCall
from .models import Passage
from .prompts import AGENT_FORCE_ANSWER, AGENT_SYSTEM
from .retrieval import Retriever

if TYPE_CHECKING:  # pragma: no cover
    from .store import Index

SNIPPET_CHARS = 450


class AgentDidNotSearch(RuntimeError):
    """The model answered without calling any retrieval tool, so there is nothing to cite."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the indexed documents. Returns numbered passages with source, page and a text snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or a question to search for."},
                    "top_k": {"type": "integer", "description": "How many passages to return (1-8). Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": "Return the full text of a passage that search_documents or related_entities already returned.",
            "parameters": {
                "type": "object",
                "properties": {"passage": {"type": "integer", "description": "The passage number, as in [3]."}},
                "required": ["passage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "related_entities",
            "description": "Look an entity up in the knowledge graph: returns entities linked to it and passages that mention them.",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string", "description": "Name of a person, method, dataset, organisation or concept."}},
                "required": ["entity"],
            },
        },
    },
]
_REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}
_TYPES = {"query": str, "top_k": int, "passage": int, "entity": str}


@dataclass
class PassageSet:
    """Passages handed to the model, numbered in the order they were first shown."""

    numbers: dict[int, Passage] = field(default_factory=dict)
    by_chunk: dict[int, int] = field(default_factory=dict)

    def add(self, p: Passage) -> int:
        if p.id in self.by_chunk:
            return self.by_chunk[p.id]
        n = len(self.numbers) + 1
        self.numbers[n] = p
        self.by_chunk[p.id] = n
        return n

    def ordered(self) -> list[Passage]:
        return [self.numbers[n] for n in sorted(self.numbers)]


def _snippet(text: str, n: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def _line(n: int, p: Passage) -> str:
    return f"[{n}] {p.label()}\n{_snippet(p.text)}"


class Tools:
    def __init__(self, index: "Index", retriever: Retriever, passages: PassageSet, doc_ids: Sequence[int] | None, overrides: dict):
        self.index, self.retriever, self.passages, self.doc_ids, self.overrides = index, retriever, passages, doc_ids, overrides

    def run(self, name: str, args: dict[str, Any]) -> str:
        return getattr(self, name)(**args)

    def search_documents(self, query: str, top_k: int = 5) -> str:
        top_k = max(1, min(int(top_k), 8))
        res = self.retriever.search(query, doc_ids=self.doc_ids, **{**self.overrides, "top_k": top_k})
        if not res.relevant:
            return "No relevant passages found for this query. Try different keywords, or say that the documents do not contain the answer."
        lines = [_line(self.passages.add(p), p) for p in res.passages]
        return f"Found {len(lines)} passages (use read_passage for the full text):\n\n" + "\n\n".join(lines)

    def read_passage(self, passage: int) -> str:
        p = self.passages.numbers.get(int(passage))
        if p is None:
            have = ", ".join(str(n) for n in sorted(self.passages.numbers)) or "none yet"
            return f"There is no passage [{passage}]. Passages available: {have}. Call search_documents first."
        return f"[{passage}] {p.label()}\n{p.text}"

    def related_entities(self, entity: str) -> str:
        if self.index.count("entities") == 0:
            return "The knowledge graph is empty (documents were ingested without graph extraction). Use search_documents instead."
        e = self.index.find_entity(entity)
        if e is None:
            return f"No entity matching {entity!r} in the knowledge graph."
        rows = self.index.neighbours(e.id, limit=12)
        out = [f"{e.name} ({e.type or 'entity'}), mentioned in {e.mentions} passages."]
        if rows:
            out.append("Linked entities:")
            for r in rows:
                arrow = f"{e.name} {r['predicate']} {r['name']}" if r["direction"] == "out" else f"{r['name']} {r['predicate']} {e.name}"
                out.append(f"- {arrow}")
        chunk_ids = list(dict.fromkeys(
            self.index.entity_chunks(e.id, self.doc_ids)[:3] + [c for r in rows[:4] for c in self.index.entity_chunks(r["id"], self.doc_ids)[:1]]
        ))[:5]
        found = self.index.get_passages(chunk_ids)
        if found:
            out.append("\nPassages:")
            for cid in chunk_ids:
                if cid in found:
                    out.append(_line(self.passages.add(found[cid]), found[cid]))
        return "\n".join(out)


def check_tool_call(tc: ToolCall) -> dict[str, Any]:
    """Validate one tool call. Raises MalformedToolCall for anything that cannot be executed."""
    if tc.name not in _REQUIRED:
        raise MalformedToolCall(f"model called an unknown tool {tc.name!r}")
    args = tc.parsed_arguments()
    for key in _REQUIRED[tc.name]:
        if key not in args:
            raise MalformedToolCall(f"tool call {tc.name} is missing required argument {key!r}")
    clean: dict[str, Any] = {}
    allowed = set(TOOLS[[t["function"]["name"] for t in TOOLS].index(tc.name)]["function"]["parameters"]["properties"])
    for key, value in args.items():
        if key not in allowed:
            continue  # extra arguments are ignored, not fatal
        want = _TYPES[key]
        if want is int and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, want):
            raise MalformedToolCall(f"argument {key!r} of {tc.name} should be {want.__name__}, got {type(value).__name__}")
        clean[key] = value
    return clean


Event = dict[str, Any]


def run_agent(
    client: ChatClient,
    tools: Tools,
    passages: PassageSet,
    question: str,
    max_steps: int,
    steps_log: list[dict],
    endpoint_info: dict,
) -> Iterator[Event]:
    """Yield token / retract / tool events, and finish by setting endpoint_info['text'] to the final answer.

    Raises ToolsUnsupported, MalformedToolCall, PromptTooLong or AgentDidNotSearch when the caller
    should fall back to pipeline mode.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM.format(max_steps=max_steps)},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps + 1):
        forced = step == max_steps
        if forced:
            if not passages.numbers:
                endpoint_info["text"] = ""
                return
            messages.append({"role": "user", "content": AGENT_FORCE_ANSWER})
        flt = CitationFilter(passages.numbers.keys())
        parts: list[str] = []
        result = None
        for ev in client.stream(messages, tools=None if forced else TOOLS, params={"temperature": 0.2}):
            if isinstance(ev, TextDelta):
                out = flt.feed(ev.text)
                if out:
                    parts.append(out)
                    yield {"type": "token", "text": out}
            elif isinstance(ev, Final):
                result = ev.result
        tail = flt.flush()
        if tail:
            parts.append(tail)
            yield {"type": "token", "text": tail}
        assert result is not None
        endpoint_info.update(endpoint=result.endpoint, model=result.model)

        if result.tool_calls and not forced:
            checked = [(tc, check_tool_call(tc)) for tc in result.tool_calls]  # all-or-nothing: raises on a bad call
            if parts:
                yield {"type": "retract"}  # text written before a tool call is not part of the answer
            messages.append(result.assistant_message())
            for tc, args in checked:
                output = tools.run(tc.name, args)
                steps_log.append({"step": step + 1, "tool": tc.name, "arguments": args})
                yield {"type": "tool", "name": tc.name, "arguments": args, "summary": output.splitlines()[0][:120]}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            continue

        if not steps_log:
            if parts:
                yield {"type": "retract"}
            raise AgentDidNotSearch("the model answered without using any retrieval tool")
        endpoint_info["text"] = "".join(parts)
        endpoint_info["used"], endpoint_info["flagged"] = flt.used, flt.flagged
        return
    endpoint_info["text"] = ""  # pragma: no cover - loop always returns above
