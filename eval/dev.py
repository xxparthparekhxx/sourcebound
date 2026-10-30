"""Choose two heuristics on the dev papers (disjoint from the evaluation papers).

    sourcebound eval dev templates     # which hypothesis template openjev-flash should rerank with
    sourcebound eval dev gate          # where the "nothing relevant" vector-similarity threshold should sit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DEFAULT_DATA, DEFAULT_WORK, RESULTS, ChunkText, Evidence, environment, load_questions, question_metrics, write_json,
)

from sourcebound.config import load_config
from sourcebound.ingest.pipeline import ingest_path
from sourcebound.runtime import Runtime

TEMPLATES = {
    "answers-explicitly": 'The passage answers the question "{question}": it states the answer explicitly.',
    "contains-answer": "The text contains the answer to the question: {question}",
    "relevant": 'This passage is relevant to the question "{question}".',
    "question-only": "{question}",
}

OFF_TOPIC = """What is the capital of Mongolia?
How do I bake sourdough bread at home?
Who won the football world cup in 2010?
What is the boiling point of tungsten?
How many moons does Saturn have?
What is the best way to learn the guitar?
When did the Roman Empire fall?
How does a refrigerator work?
What are the symptoms of the flu?
Who painted the Mona Lisa?
How tall is Mount Kilimanjaro?
What is the speed of light in a vacuum?
How do I change a flat tire?
What is the plot of Romeo and Juliet?
How much water should an adult drink per day?
Which planet is closest to the sun?
What is the population of Brazil?
How do vaccines work?
What is the recipe for chocolate chip cookies?
Who invented the telephone?
What causes earthquakes?
How do I file my income taxes?
What is the longest river in Africa?
Which country has the most islands?
How does compound interest work?
What is the tallest building in the world?
How do bees make honey?
What is the difference between weather and climate?
When was the printing press invented?
How do airplanes stay in the air?
What is the national anthem of Canada?
How many players are on a basketball team?
What is the chemical formula of table salt?
Who wrote Pride and Prejudice?
How long does it take to walk a marathon?
What is the currency of Japan?
How do solar panels generate electricity?
Which vitamin helps the body absorb calcium?
What is the largest ocean on Earth?
How do I train a puppy to sit?""".splitlines()


def open_dev(args, name: str, docs: str):
    cfg = load_config(args.config)
    cfg.data_dir = args.work / name
    rt = Runtime(cfg)
    ingest_path(rt.index, args.data / docs, cfg, rt.client)
    sources = {p.stem: str(p.resolve()) for p in sorted((args.data / docs).glob("*.md"))}
    doc_id = {pid: rt.index.doc_ids_for_sources([s])[0] for pid, s in sources.items()}
    return rt, doc_id


def templates(args) -> None:
    data = load_questions(args.data, "dev_questions.json")
    rt, doc_id = open_dev(args, "dev-index", "dev_docs")
    cache: dict[int, ChunkText] = {}

    def ct(p):
        return cache.setdefault(p.id, ChunkText(p.text))

    prepared = [(q, [[Evidence(e) for e in a["evidence"]] for a in q["annotators"]]) for q in data["questions"]]
    cfg = rt.cfg
    results = {}
    for name in ["hybrid (no rerank)", *TEMPLATES]:
        if name != "hybrid (no rerank)":
            cfg.rerank.hypothesis = TEMPLATES[name]
            rt.reranker.cfg = cfg.rerank
            rt.reranker._scorer = rt.reranker._scorer or None
            if rt.reranker._scorer is not None:
                rt.reranker._scorer.cfg = cfg.rerank
        rows = []
        for scope in ("paper",):  # the template choice only needs the within-paper ranking
            ms = []
            for q, anns in prepared:
                docs = [doc_id[q["paper_id"]]] if scope == "paper" else None
                res = rt.retriever.search(q["question"], doc_ids=docs, top_k=10, rerank=name != "hybrid (no rerank)")
                ms.append(question_metrics([ct(p) for p in res.passages], anns))
            n = len(ms)
            rows.append({"scope": scope, **{k: round(sum(m[k] for m in ms) / n, 4) for k in ("recall@5", "recall@10", "rr")}})
        results[name] = rows
        print(f"{name:22}", *[f"{r['scope']}: R@5 {r['recall@5']:.3f} R@10 {r['recall@10']:.3f} MRR {r['rr']:.3f}" for r in rows], flush=True)
    write_json(RESULTS / "dev-rerank-templates.json", {
        "meta": {**environment(), "task": "choose the rerank hypothesis template on dev papers",
                 "dev_questions": len(prepared), "dev_papers": len(data["papers"]), "templates": TEMPLATES},
        "results": results})
    rt.close()


def gate(args) -> None:
    """Best vector similarity for real questions vs questions the index cannot answer."""
    dev = load_questions(args.data, "dev_questions.json")["questions"]
    ev = load_questions(args.data, "questions.json")["questions"]
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    out = {}
    for label, idx_name, docs, pos, out_of_corpus in (
        ("dev", "dev-index", "dev_docs", dev, ev),
        ("eval", "qasper-index", "docs", ev, dev),
    ):
        rt, _ = open_dev(args, idx_name, docs) if idx_name == "dev-index" else _open_main(args)
        def best(qs):
            return [rt.retriever.search(q if isinstance(q, str) else q["question"], vector=True, bm25=False, mmr=False, top_k=1).best_vector for q in qs]
        sims = {"answerable": best(pos), "out_of_corpus": best(out_of_corpus), "off_topic": best(OFF_TOPIC)}
        gate_pass = {}
        for kind, qs in (("answerable", pos), ("out_of_corpus", out_of_corpus), ("off_topic", OFF_TOPIC)):
            gate_pass[kind] = sum(rt.retriever.search(q if isinstance(q, str) else q["question"], top_k=5).relevant for q in qs) / len(qs)
        table = {t: {k: round(sum(s >= t for s in v) / len(v), 3) for k, v in sims.items()} for t in thresholds}
        out[label] = {
            "n": {k: len(v) for k, v in sims.items()},
            "best_similarity_percentiles": {k: {p: round(float(__import__("numpy").percentile(v, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)} for k, v in sims.items()},
            "share_with_best_similarity_at_or_above_threshold": table,
            "configured_gate_pass_rate": gate_pass | {"min_vector_score": rt.cfg.retrieval.min_vector_score, "min_term_overlap": rt.cfg.retrieval.min_term_overlap},
        }
        print(label, json.dumps(out[label], indent=1))
        rt.close()
    write_json(RESULTS / "gate-calibration.json", {"meta": {**environment(), "task": "vector-similarity gate calibration",
                "embedding": "see retrieval results", "off_topic_questions": OFF_TOPIC}, "results": out})


def _open_main(args):
    cfg = load_config(args.config)
    cfg.data_dir = args.work / "qasper-index"
    return Runtime(cfg), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["templates", "gate"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--config")
    args = ap.parse_args()
    {"templates": templates, "gate": gate}[args.what](args)


if __name__ == "__main__":
    main()
