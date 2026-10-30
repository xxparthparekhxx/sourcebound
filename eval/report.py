"""Turn the raw JSON in eval/results/ into Markdown tables (eval/results/SUMMARY.md).

    docgraph eval report            # print and write eval/results/SUMMARY.md

Every number in the README is copied from this output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RESULTS

ORDER = ["vector", "vector+mmr", "bm25", "hybrid", "hybrid+graph", "hybrid+rerank", "hybrid+graph+rerank"]


def fmt(d: dict | None, digits: int = 3) -> str:
    if not d:
        return "n/a"
    return f"{d['mean']:.{digits}f} ({d['ci95_lo']:.{digits}f}-{d['ci95_hi']:.{digits}f})"


def retrieval_tables(path: Path) -> list[str]:
    r = json.loads(path.read_text())
    m = r["meta"]
    out = [
        f"### Retrieval ablation: `{path.name}`",
        "",
        "- runs (settings merged into this file, in order):",
        *[f"  - {x['date']}, commit `{x['git_commit']}`{' (uncommitted changes)' if x.get('git_dirty') else ''}: `{x['command']}`"
          for x in m.get("runs", [m])],
        f"- dataset: {m['dataset']['dataset']} {m['dataset']['split']} split, {m['dataset']['papers']} papers, "
        f"{m['dataset']['questions_evaluated']} questions; {m['index_stats']['chunks']} chunks "
        f"(size {m['chunking']['chunk_size']}, overlap {m['chunking']['chunk_overlap']} characters)",
        f"- embeddings: {m['embedding']['model']} (dimension {m['embedding']['dimension']}); rerank: {m['rerank']['model']} on {m['rerank']['device']}",
        f"- gold evidence reachable in the chunks: {m['evidence_findable']['covered']}/{m['evidence_findable']['total']} paragraphs",
        f"- matching: {m['gold_match']}; cells are mean (95% bootstrap interval over questions)",
    ]
    bi = m.get("index_build") or {}
    if "graph" in bi:
        g = bi["graph"]
        llm = ", ".join(f"{e['name']}={e['model']}" for e in bi.get("graph_llm", []))
        out.append(
            f"- graph: {m['index_stats']['entities']} entities, {m['index_stats']['relations']} relations, extracted by {llm}; "
            f"{g['extracted']}/{g['chunks']} chunks extracted, {g['invalid_dropped']} dropped as invalid, {g['call_errors']} failed calls, {g['seconds']:.0f}s"
        )
    for scope, by in r["summary"].items():
        label = {"paper": "within the question's paper", "pooled": "pooled over all papers"}[scope]
        out += ["", f"**Retrieval {label}**", "", "| setting | recall@5 | recall@10 | MRR@10 |", "|---|---|---|---|"]
        for name in ORDER:
            if name in by:
                s = by[name]
                out.append(f"| {name} | {fmt(s['recall@5'])} | {fmt(s['recall@10'])} | {fmt(s['mrr'])} |")
        n = next(iter(by.values()))["n"]
        out.append(f"\n(n = {n} questions)")
    return out


def answer_tables(paths: list[Path]) -> list[str]:
    out = ["### Answer quality", ""]
    rows = []
    for p in paths:
        r = json.loads(p.read_text())
        m, s = r["meta"], r["summary"]
        rows.append((p, m, s))
    out += [
        "| run | endpoint / model | mode | questions | answered | no answer | cites something | citation precision (micro) | citation precision (macro) | faithful sentences | HTTP requests |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p, m, s in rows:
        micro = s["citation_precision_micro"]
        micro_s = f"{micro['value']:.3f} ({micro['overlapping_gold']}/{micro['cited_passages']})" if micro["value"] is not None else "n/a"
        f = s["faithfulness_sentence_level"]
        faith = f"{fmt(f)} of {s['cited_sentences_scored']}" if f else "n/a"
        ep = m["chat_endpoint"]
        out.append(
            f"| {p.stem.removeprefix('answers-')} | {ep['name']} / {ep['model']} | {m['mode']} | {s['questions_asked']} | "
            f"{s['answers_returned'] - s['no_answer']} | {s['no_answer']} | {s['answered_with_citations']} | {micro_s} | "
            f"{fmt(s['citation_precision_macro'])} | {faith} | {m['http_requests']} |"
        )
    out.append("")
    for p, m, s in rows:
        out.append(
            f"- `{p.name}`: {m['date']}, commit `{m['git_commit']}`; command `{m['command']}`; sample seed {m['sample']['seed']}; "
            f"retrieval {m['retrieval_scope']}, {m['retrieval_flags']}; faithfulness model {m['faithfulness_model']} "
            f"(entailment >= {m['entailment_threshold']}); {s['answers_with_removed_citations']} answers had a citation removed; "
            f"{s['call_errors']} failed calls; {s['agent_fallbacks']} agent fallbacks; "
            f"{s.get('prompts_reported_truncated', 'n/a')} prompts reported truncated by the server; "
            f"gold evidence among the passages given to the model in {fmt(s['gold_evidence_in_provided_passages'])} of answers"
            + (f"; STOPPED EARLY: {m['stopped_early']}" if m.get("stopped_early") else "")
        )
    return out


def dev_tables() -> list[str]:
    out = []
    t = RESULTS / "dev-rerank-templates.json"
    if t.exists():
        r = json.loads(t.read_text())
        out += ["### Choosing the rerank hypothesis (dev papers, not in the evaluation set)", "",
                f"{r['meta']['dev_questions']} questions from {r['meta']['dev_papers']} papers; run {r['meta']['date']}.", "",
                "| variant | within-paper recall@5 | recall@10 | MRR@10 |", "|---|---|---|---|"]
        for name, rows in r["results"].items():
            by = {x["scope"]: x for x in rows}
            out.append(f"| {name} | {by['paper']['recall@5']:.3f} | {by['paper']['recall@10']:.3f} | {by['paper']['rr']:.3f} |")
        out.append("")
    g = RESULTS / "gate-calibration.json"
    if g.exists():
        r = json.loads(g.read_text())
        out += ["### The 'nothing relevant' gate", "", f"Run {r['meta']['date']}. Share of questions whose best vector similarity reaches each threshold; "
                "'answerable' = the paper is in the index, 'out of corpus' = real NLP questions about papers that are not, 'off topic' = 40 general-knowledge questions.", ""]
        for label, d in r["results"].items():
            out += [f"**{label} index** (n = {d['n']})", "", "| threshold | answerable | out of corpus | off topic |", "|---|---|---|---|"]
            for t_, row in d["share_with_best_similarity_at_or_above_threshold"].items():
                out.append(f"| {t_} | {row['answerable']:.3f} | {row['out_of_corpus']:.3f} | {row['off_topic']:.3f} |")
            gp = d["configured_gate_pass_rate"]
            out.append(f"\nConfigured gate (min_vector_score {gp['min_vector_score']} or lexical overlap {gp['min_term_overlap']}): "
                       f"passes {gp['answerable']:.3f} of answerable, {gp['out_of_corpus']:.3f} of out-of-corpus, {gp['off_topic']:.3f} of off-topic questions.\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--config", help=argparse.SUPPRESS)
    a = ap.parse_args()
    lines = ["# Evaluation summary", "", "Generated by `docgraph eval report` from the JSON files in this directory.", ""]
    for p in sorted(RESULTS.glob("retrieval-*.json")):
        lines += retrieval_tables(p) + [""]
    ans = sorted(RESULTS.glob("answers-*.json"))
    if ans:
        lines += answer_tables(ans) + [""]
    lines += dev_tables()
    text = "\n".join(lines).rstrip() + "\n"
    print(text)
    if not a.no_write:
        (RESULTS / "SUMMARY.md").write_text(text)
        print(f"wrote {RESULTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
