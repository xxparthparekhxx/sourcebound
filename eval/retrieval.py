"""Retrieval ablation on Qasper: recall@5, recall@10 and MRR against gold evidence paragraphs.

    docgraph eval prepare
    docgraph eval retrieval --build --graph          # index the corpus; --graph runs LLM entity extraction
    docgraph eval retrieval --tag main               # run every setting, both scopes

Scopes:
  paper   retrieval is restricted to the paper the question was written about (Qasper's own protocol)
  pooled  retrieval runs over all papers at once, which is what the assistant does with a document set

Settings switch the pipeline stages on and off: vector, vector+MMR, BM25, hybrid (RRF), +graph,
+rerank (openjev-flash). No chat model is used except for building the graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import (
    DEFAULT_DATA, DEFAULT_WORK, RESULTS, ChunkText, Evidence, bootstrap_ci, environment,
    load_questions, question_metrics, write_json,
)

from docgraph.config import ConfigError, load_config
from docgraph.ingest.pipeline import ingest_path
from docgraph.runtime import Runtime

SETTINGS: dict[str, dict] = {
    "vector": dict(vector=True, bm25=False, mmr=False, graph=False, rerank=False),
    "vector+mmr": dict(vector=True, bm25=False, mmr=True, graph=False, rerank=False),
    "bm25": dict(vector=False, bm25=True, mmr=False, graph=False, rerank=False),
    "hybrid": dict(vector=True, bm25=True, mmr=True, graph=False, rerank=False),
    "hybrid+graph": dict(vector=True, bm25=True, mmr=True, graph=True, rerank=False),
    "hybrid+rerank": dict(vector=True, bm25=True, mmr=True, graph=False, rerank=True),
    "hybrid+graph+rerank": dict(vector=True, bm25=True, mmr=True, graph=True, rerank=True),
}
METRICS = ("recall@5", "recall@10", "rr")


def doc_sources(data: Path, docs: str) -> dict[str, str]:
    return {p.stem: str(p.resolve()) for p in sorted((data / docs).glob("*.md"))}


def build_index(rt: Runtime, data: Path, docs: str, graph: bool, workers: int) -> dict:
    t0 = time.time()
    rt.cfg.ingest.graph.workers = workers
    last = [0.0]

    def prog(d):
        pass

    def gprog(stats):
        done = stats.extracted + stats.failed + stats.errors
        if time.time() - last[0] > 20 or done == stats.chunks:
            last[0] = time.time()
            print(f"  graph: {done}/{stats.chunks} chunks, {stats.entities} entities, {stats.relations} relations, "
                  f"{stats.failed} invalid, {stats.errors} errors ({time.time() - t0:.0f}s)", flush=True)

    from docgraph.graph import extract_graph

    rep = ingest_path(rt.index, data / docs, rt.cfg, rt.client, graph=False, progress=prog)
    out = {"ingested": rep.count("ingested"), "unchanged": rep.count("unchanged"), "failed": rep.count("failed"),
           "seconds": round(time.time() - t0, 1)}
    if graph:
        t1 = time.time()
        g = extract_graph(rt.index, rt.client, rt.cfg.ingest.graph, progress=gprog)
        out["graph"] = {"chunks": g.chunks, "extracted": g.extracted, "invalid_dropped": g.failed, "call_errors": g.errors,
                        "seconds": round(time.time() - t1, 1), "failures": g.failures}
    out["stats"] = rt.index.stats()
    return out


def run(args) -> None:
    data = args.data
    qdata = load_questions(data, "questions.json")
    questions = qdata["questions"][: args.limit] if args.limit else qdata["questions"]
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(f"configuration error: {e}")
    cfg.data_dir = args.work / "qasper-index"
    rt = Runtime(cfg)
    try:
        sources = doc_sources(data, "docs")
        build_info = None
        if args.build:
            print("building the index ...", flush=True)
            build_info = build_index(rt, data, "docs", args.graph, args.graph_workers)
            print(json.dumps({k: v for k, v in build_info.items() if k != "graph"}), flush=True)
            if "graph" in build_info:
                print("graph:", {k: v for k, v in build_info["graph"].items() if k != "failures"})
            info = args.work / "qasper-index" / "build-info.json"
            prev = json.loads(info.read_text()) if info.exists() else {}
            # keep an earlier graph build record when this build only re-checked the documents
            if "graph" not in build_info and "graph" in prev:
                build_info["graph"] = prev["graph"]
            if "graph" in build_info:
                build_info["graph_llm"] = [{"name": e.name, "model": e.model, "options": e.options} for e in cfg.chat.endpoints]
            info.write_text(json.dumps(build_info, indent=1))
            if not args.tag:
                return
        else:
            info = args.work / "qasper-index" / "build-info.json"
            build_info = json.loads(info.read_text()) if info.exists() else None
        idx = rt.index
        doc_id = {pid: idx.doc_ids_for_sources([src])[0] for pid, src in sources.items() if idx.doc_ids_for_sources([src])}
        missing = set(sources) - set(doc_id)
        if missing:
            sys.exit(f"{len(missing)} papers are not indexed; run with --build first")

        settings = [s for s in (args.settings.split(",") if args.settings else SETTINGS) if s in SETTINGS]
        have_graph = idx.count("entities") > 0
        if not have_graph:
            dropped = [s for s in settings if SETTINGS[s]["graph"]]
            settings = [s for s in settings if not SETTINGS[s]["graph"]]
            if dropped:
                print(f"no graph in the index: skipping {dropped}", file=sys.stderr)

        # Sanity check on the matching itself: can the gold evidence be found in the paper's chunks at all?
        chunk_cache: dict[int, ChunkText] = {}

        def ct(p) -> ChunkText:
            if p.id not in chunk_cache:
                chunk_cache[p.id] = ChunkText(p.text)
            return chunk_cache[p.id]

        prepared = []
        findable = total_ev = 0
        for q in questions:
            anns = [[Evidence(e) for e in a["evidence"]] for a in q["annotators"]]
            paper_chunks = [ct(p) for p in idx.document_passages(doc_id[q["paper_id"]])]
            for ann in anns:
                for ev in ann:
                    total_ev += 1
                    findable += any(ev.covered_by(c) for c in paper_chunks)
            prepared.append((q, anns))
        print(f"{len(questions)} questions from {len(sources)} papers; "
              f"gold evidence paragraphs covered by some chunk of their own paper: {findable}/{total_ev}", flush=True)

        records: list[dict] = []
        timing: dict[str, float] = {}
        for scope in args.scopes.split(","):
            for name in settings:
                flags = SETTINGS[name]
                t0 = time.time()
                for q, anns in prepared:
                    docs = [doc_id[q["paper_id"]]] if scope == "paper" else None
                    res = rt.retriever.search(q["question"], doc_ids=docs, top_k=args.top_k, **flags)
                    m = question_metrics([ct(p) for p in res.passages], anns)
                    records.append({"qid": q["qid"], "scope": scope, "setting": name,
                                    "chunk_ids": [p.id for p in res.passages], **m})
                timing[f"{scope}/{name}"] = round(time.time() - t0, 1)
                sub = [r for r in records if r["scope"] == scope and r["setting"] == name]
                print(f"{scope:6} {name:22} R@5 {sum(r['recall@5'] for r in sub)/len(sub):.3f}  "
                      f"R@10 {sum(r['recall@10'] for r in sub)/len(sub):.3f}  "
                      f"MRR {sum(r['rr'] for r in sub)/len(sub):.3f}   ({timing[f'{scope}/{name}']}s)", flush=True)

        summary: dict = {}
        for r in records:
            summary.setdefault(r["scope"], {}).setdefault(r["setting"], []).append(r)
        for scope, by in summary.items():
            for name, rows in by.items():
                summary[scope][name] = {"n": len(rows)} | {
                    ("mrr" if k == "rr" else k): dict(zip(("mean", "ci95_lo", "ci95_hi"), map(lambda x: round(x, 4), bootstrap_ci([r[k] for r in rows]))))
                    for k in METRICS
                }

        r_cfg = cfg.retrieval
        out = {
            "meta": {
                **environment(),
                "task": "retrieval ablation",
                "dataset": qdata["meta"] | {"questions_evaluated": len(questions)},
                "embedding": idx.embedding_info(),
                "chunking": {"chunk_size": cfg.ingest.chunk_size, "chunk_overlap": cfg.ingest.chunk_overlap,
                             "context_header": cfg.ingest.context_header},
                "retrieval_params": {k: getattr(r_cfg, k) for k in
                                     ("candidates", "vector_fetch", "mmr_lambda", "rrf_k", "graph_max_chunks", "rerank_candidates")}
                                    | {"top_k": args.top_k},
                "rerank": {"model": cfg.rerank.model, "device": cfg.rerank.device, "hypothesis": cfg.rerank.hypothesis},
                "index_stats": idx.stats(),
                "evidence_findable": {"covered": findable, "total": total_ev},
                "gold_match": "word 5-gram containment >= 0.5; recall = max over annotators",
                "seconds_per_setting": timing,
                "index_build": build_info,
            },
            "summary": summary,
            "per_question": records,
        }
        write_json(RESULTS / f"retrieval-{args.tag}.json", out)
    finally:
        rt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--build", action="store_true", help="(re)build the index from the prepared documents")
    ap.add_argument("--graph", action="store_true", help="with --build: also run LLM graph extraction")
    ap.add_argument("--graph-workers", type=int, default=4)
    ap.add_argument("--tag", help="run the ablation and write eval/results/retrieval-<tag>.json")
    ap.add_argument("--settings", help=f"comma list of: {', '.join(SETTINGS)}")
    ap.add_argument("--scopes", default="paper,pooled")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, help="only the first N questions (debugging)")
    ap.add_argument("--config")
    args = ap.parse_args()
    if not args.build and not args.tag:
        ap.error("give --build, --tag, or both")
    run(args)


if __name__ == "__main__":
    main()
