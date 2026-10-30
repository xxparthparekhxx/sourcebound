"""Command line interface: ingest, ask, docs, remove, reset, serve, eval."""

from __future__ import annotations

import argparse
import json
import logging
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import Config, ConfigError, load_config, load_dotenv


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docgraph", description="Answer questions over your documents, with cited sources.")
    p.add_argument("--config", help="path to a TOML config file (default: ./docgraph.toml, then ~/.config/docgraph/config.toml)")
    p.add_argument("--version", action="version", version=f"docgraph {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="log warnings from the chat client and ingest")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index a file or a directory")
    s.add_argument("path")
    s.add_argument("--graph", action="store_true", default=None, help="extract entities and relations (one LLM call per chunk)")
    s.add_argument("--ocr", action="store_true", default=None, help="OCR scanned PDF pages with a vision-capable model")
    s.add_argument("--force", action="store_true", help="re-index files even if they are unchanged")

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question")
    s.add_argument("--mode", choices=["pipeline", "agent"])
    s.add_argument("--top-k", type=int)
    for stage in ("vector", "bm25", "mmr"):
        s.add_argument(f"--no-{stage}", dest=stage, action="store_false", default=None, help=f"switch the {stage} stage off")
    s.add_argument("--graph", action="store_true", default=None, help="add graph expansion")
    s.add_argument("--rerank", action="store_true", default=None, help="rerank with openjev-flash (needs --extra rerank)")
    s.add_argument("--json", action="store_true", help="print the full answer as JSON instead of streaming text")

    sub.add_parser("docs", help="list indexed documents")

    s = sub.add_parser("remove", help="remove one document from the index")
    s.add_argument("path")

    s = sub.add_parser("reset", help="delete the whole index")
    s.add_argument("--yes", action="store_true")

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    s = sub.add_parser("eval", help="run the evaluation scripts in eval/ (prepare | dev | retrieval | answers | report)", add_help=False)
    s.add_argument("script", choices=["prepare", "retrieval", "answers", "report", "dev"])
    s.add_argument("args", nargs=argparse.REMAINDER)
    return p


# ---------------------------------------------------------------------------- commands


def cmd_ingest(cfg: Config, a) -> int:
    from .ingest.pipeline import ingest_path
    from .runtime import Runtime

    rt = Runtime(cfg)
    try:
        def show(d) -> None:
            tail = f"  ({d.n_chunks} chunks)" if d.status in ("ingested", "replaced", "unchanged") else ""
            print(f"{d.status:<10} {d.path}{tail}")
            if d.error:
                print(f"           error: {d.error}")
            for s in d.skipped_pages:
                print(f"           skipped page {s['page']}: {s['reason']}")
            if d.ocr_pages:
                print(f"           OCR used on page(s) {', '.join(map(str, d.ocr_pages))}")

        report = ingest_path(rt.index, a.path, cfg, rt.client, graph=a.graph, ocr=a.ocr, force=a.force, progress=show)
        counts = {s: report.count(s) for s in ("ingested", "replaced", "unchanged", "failed", "unsupported")}
        summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n) or "nothing to ingest"
        print("\n" + summary)
        if report.graph is not None:
            g = report.graph
            print(f"graph: {g.extracted} chunks extracted ({g.entities} entities, {g.relations} relations), "
                  f"{g.failed} dropped as invalid, {g.errors} failed calls")
            for f in g.failures:
                _err(f"  {f}")
        st = rt.index.stats()
        print(f"index: {st['documents']} documents, {st['chunks']} chunks, {st['entities']} entities, {st['relations']} relations")
        skipped = sum(len(d.skipped_pages) for d in report.documents)
        if skipped:
            print(f"{skipped} page(s) were skipped (no text layer). Use --ocr with a vision-capable model to read them.")
        return 1 if counts["failed"] or counts["unsupported"] else 0
    finally:
        rt.close()


def cmd_ask(cfg: Config, a) -> int:
    from .runtime import Runtime

    overrides = {k: getattr(a, k) for k in ("vector", "bm25", "mmr", "graph", "rerank") if getattr(a, k, None) is not None}
    if a.top_k:
        overrides["top_k"] = a.top_k
    rt = Runtime(cfg)
    try:
        if rt.index.count("chunks") == 0:
            _err("The index is empty. Run `docgraph ingest <path>` first.")
            return 1
        if a.json:
            ans = rt.answerer.ask(a.question, mode=a.mode, **overrides)
            print(json.dumps(asdict(ans), indent=2, ensure_ascii=False))
            return 0
        citations: list[dict] = []
        flagged: list[int] = []
        done: dict = {}
        wrote = False
        for ev in rt.answerer.stream(a.question, mode=a.mode, **overrides):
            t = ev["type"]
            if t == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
                wrote = True
            elif t == "retract":
                if wrote:
                    sys.stdout.write("\n")
                    _err("(the text above was a draft before a tool call, not the answer)")
                wrote = False
            elif t == "notice":
                _err(f"note: {ev['message']}")
            elif t == "tool":
                _err(f"tool: {ev['name']}({', '.join(f'{k}={v!r}' for k, v in ev['arguments'].items())}) -> {ev['summary']}")
            elif t == "citations":
                citations, flagged = ev["citations"], ev["flagged"]
            elif t == "done":
                done = ev
        print()
        if citations:
            print("\nSources:")
            for c in citations:
                loc = f", p. {c['page']}" if c["page"] is not None else ""
                head = f", {c['heading']}" if c["heading"] else ""
                print(f"  [{c['n']}] {c['file']}{loc}{head}")
        if flagged:
            _err(f"warning: removed citation(s) {', '.join('[%d]' % n for n in flagged)}: not passages the model was given")
        if done.get("uncited") and not done.get("no_answer"):
            _err("warning: the answer cites no passage")
        sys.stdout.flush()
        if done.get("no_answer"):
            _err("[no answer: nothing relevant was retrieved, so no model was called]")
        else:
            mode = done.get("mode", "") + (" (fell back from agent)" if done.get("fallback") else "")
            _err(f"[{mode}; {done.get('endpoint', '')} / {done.get('model', '')}]")
        return 0
    finally:
        rt.close()


def cmd_docs(cfg: Config, a) -> int:
    from .store import Index

    idx = Index.open(cfg)
    try:
        docs = idx.documents()
        if not docs:
            print("(no documents)")
        for d in docs:
            pages = f", {d.n_pages} pages" if d.n_pages else ""
            skipped = f", {len(d.skipped_pages)} skipped" if d.skipped_pages else ""
            print(f"{d.title}  [{d.format}{pages}, {d.n_chunks} chunks{skipped}]\n    {d.source}")
        info = idx.embedding_info()
        if info:
            print(f"\nembedding model: {info['model']} (dimension {info['dimension']})")
        return 0
    finally:
        idx.close()


def cmd_remove(cfg: Config, a) -> int:
    from .store import Index

    idx = Index.open(cfg)
    try:
        ok = idx.delete_document(str(Path(a.path).expanduser().resolve()))
        print("removed" if ok else "not in the index")
        return 0 if ok else 1
    finally:
        idx.close()


def cmd_reset(cfg: Config, a) -> int:
    from .store import Index

    if not a.yes:
        _err(f"This deletes the whole index under {cfg.data_dir}. Re-run with --yes to confirm.")
        return 1
    idx = Index.open(cfg)
    try:
        idx.reset()
        print("index cleared")
        return 0
    finally:
        idx.close()


def cmd_serve(cfg: Config, a) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="info")
    return 0


def cmd_eval(cfg: Config, a, config_path: str | None) -> int:
    root = Path(__file__).resolve().parents[2] / "eval"
    script = root / f"{a.script}.py"
    if not script.is_file():
        _err(f"eval scripts not found at {root}. `docgraph eval` works from a source checkout of the repository.")
        return 1
    sys.path.insert(0, str(root))
    argv = [str(script), *a.args]
    if config_path:
        argv += ["--config", config_path]
    sys.argv = argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
        if args.cmd == "eval":
            return cmd_eval(cfg, args, args.config)
        handler = {"ingest": cmd_ingest, "ask": cmd_ask, "docs": cmd_docs, "remove": cmd_remove,
                   "reset": cmd_reset, "serve": cmd_serve}[args.cmd]
        return handler(cfg, args)
    except ConfigError as e:
        _err(f"configuration error: {e}")
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - a CLI should print a message, not a traceback
        from .store import EmbeddingMismatch

        if isinstance(e, (EmbeddingMismatch, FileNotFoundError)) or type(e).__module__.startswith("docgraph"):
            _err(f"error: {e}")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
