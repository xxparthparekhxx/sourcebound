"""Answer quality on a small Qasper sample: citation precision and faithfulness.

    docgraph eval answers --tag ollama-bonsai2 --endpoint ollama --n 30
    docgraph eval answers --tag openrouter-gemma --endpoint openrouter --n 30 --max-requests 80

For each sampled question, retrieval is restricted to the paper the question was written about, the
assistant answers, and then:

  citation precision  of the passages the answer cites, the share that overlap the gold evidence
                      (same 5-gram overlap rule as the retrieval eval). Reported micro (all citations
                      pooled) and macro (mean over answers that cite something).
  faithfulness        every answer sentence that carries a [n] citation is checked against the text of
                      the passages it cites with openjev-flash (P(entailment) >= 0.5 counts as
                      supported). Sentences without a citation are counted but not scored.

Only the chosen endpoint is used (no fallback to another model), so a run measures one model.
Every HTTP request, including failed attempts, counts toward --max-requests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from common import (
    DEFAULT_DATA, DEFAULT_WORK, RESULTS, ChunkText, Evidence, bootstrap_ci, environment, load_questions, sample, write_json,
)

from docgraph.config import ConfigError, load_config
from docgraph.llm import ChatError
from docgraph.rerank import FlashScorer, split_sentences
from docgraph.runtime import Runtime

_MARK = re.compile(r"\[(\d+(?:\s*[,;-]\s*\d+)*)\]")


def marker_numbers(sentence: str) -> list[int]:
    nums: list[int] = []
    for m in _MARK.finditer(sentence):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            if "-" in part:
                a, b = part.split("-", 1)
                nums.extend(range(int(a), int(b) + 1))
            else:
                nums.append(int(part))
    return nums


def strip_markers(sentence: str) -> str:
    return re.sub(r"\s+", " ", _MARK.sub("", sentence)).strip()


def run(args) -> None:
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(f"configuration error: {e}")
    chosen = [e for e in cfg.chat.endpoints if e.name == args.endpoint]
    if not chosen:
        sys.exit(f"no endpoint named {args.endpoint!r}; configured: {[e.name for e in cfg.chat.endpoints]}")
    if args.model:
        chosen[0] = chosen[0].model_copy(update={"model": args.model})
    cfg.chat.endpoints = chosen
    cfg.chat.passes = args.passes
    cfg.chat.backoff = args.backoff
    cfg.data_dir = args.work / "qasper-index"
    rt = Runtime(cfg)
    if not rt.client.usable_endpoints():
        sys.exit(f"endpoint {args.endpoint!r} is not usable (is its API key variable set?)")

    qdata = load_questions(args.data)
    questions = sample(qdata["questions"], args.n, args.sample_seed)
    sources = {p.stem: str(p.resolve()) for p in sorted((args.data / "docs").glob("*.md"))}
    doc_id = {pid: rt.index.doc_ids_for_sources([s])[0] for pid, s in sources.items()}
    flags = {k: v for k, v in (("graph", args.graph or None), ("rerank", args.rerank or None)) if v}

    cache: dict[int, ChunkText] = {}

    def ct(p) -> ChunkText:
        return cache.setdefault(p.id, ChunkText(p.text))

    records = []
    stopped = None
    t_start = time.time()
    for i, q in enumerate(questions, 1):
        if rt.client.request_count >= args.max_requests:
            stopped = f"request budget of {args.max_requests} reached after {i - 1} questions"
            print(stopped, flush=True)
            break
        anns = [[Evidence(e) for e in a["evidence"]] for a in q["annotators"]]
        t0 = time.time()
        err = None
        try:
            ans = rt.answerer.ask(q["question"], mode=args.mode, doc_ids=[doc_id[q["paper_id"]]], top_k=args.top_k, **flags)
        except ChatError as e:
            ans, err = None, f"{type(e).__name__}: {str(e)[:300]}"
        dt = time.time() - t0
        rec = {"qid": q["qid"], "question": q["question"], "paper_id": q["paper_id"], "seconds": round(dt, 2), "error": err}
        if ans is not None:
            provided = rt.index.get_passages([p["id"] for p in ans.passages])
            provided_ids = [p["id"] for p in ans.passages]
            cited = [c["id"] for c in ans.citations]
            cited_texts = {c["n"]: c["text"] for c in ans.citations}
            hit = [any(ev.covered_by(ct(provided[cid])) for a in anns for ev in a) for cid in cited]
            gold_in_provided = any(ev.covered_by(ct(provided[cid])) for cid in provided_ids for a in anns for ev in a)
            rec |= {
                "answer": ans.text, "no_answer": ans.no_answer, "mode": ans.mode, "fallback": ans.fallback_reason,
                "endpoint": ans.endpoint, "model": ans.model, "uncited": ans.uncited, "flagged": ans.flagged, "truncated_prompt": ans.truncated,
                "provided_ids": provided_ids, "cited_ids": cited, "cited_hits_gold": hit,
                "gold_in_provided": gold_in_provided, "steps": len(ans.steps),
                "_cited_texts": cited_texts,
            }
        records.append(rec)
        print(f"[{i}/{len(questions)}] {dt:5.1f}s req={rt.client.request_count} "
              f"{'ERROR ' + err if err else ('no-answer' if rec.get('no_answer') else 'cited=' + str(len(rec.get('cited_ids', []))))}",
              flush=True)

    # ---- faithfulness with openjev-flash, batched at the end
    scorer = FlashScorer(cfg.rerank)
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, int]] = []
    for ri, r in enumerate(records):
        if r.get("answer") and not r.get("no_answer"):
            r["sentences"] = []
            for sent in split_sentences(r["answer"]):
                nums = [n for n in dict.fromkeys(marker_numbers(sent)) if n in r["_cited_texts"]]
                s = {"text": sent, "cited": nums}
                r["sentences"].append(s)
                if nums:
                    hyp = strip_markers(sent)
                    if len(hyp) >= 8:
                        pairs.append(("\n\n".join(r["_cited_texts"][n] for n in nums), hyp))
                        owners.append((ri, len(r["sentences"]) - 1))
    if pairs:
        probs = scorer.predict(pairs)
        for (ri, si), p in zip(owners, probs):
            records[ri]["sentences"][si] |= {
                "p_entail": round(float(p[1]), 4), "p_contradict": round(float(p[0]), 4),
                "supported": bool(p[1] >= cfg.rerank.entailment_threshold),
            }
    for r in records:
        r.pop("_cited_texts", None)

    # ---- summary
    done = [r for r in records if r.get("answer") is not None]
    answered = [r for r in done if not r["no_answer"]]
    with_cites = [r for r in answered if r["cited_ids"]]
    total_cited = sum(len(r["cited_ids"]) for r in with_cites)
    total_hits = sum(sum(r["cited_hits_gold"]) for r in with_cites)
    macro = [sum(r["cited_hits_gold"]) / len(r["cited_ids"]) for r in with_cites]
    scored = [s for r in answered for s in r.get("sentences", []) if "supported" in s]
    uncited_sent = sum(1 for r in answered for s in r.get("sentences", []) if not s["cited"])
    per_answer = [all(s["supported"] for s in r["sentences"] if "supported" in s) for r in answered
                  if any("supported" in s for s in r.get("sentences", []))]

    def ci(vals):
        m, lo, hi = bootstrap_ci(vals)
        return {"mean": round(m, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "n": len(vals)}

    summary = {
        "questions_asked": len(records), "answers_returned": len(done), "call_errors": sum(1 for r in records if r.get("error")),
        "no_answer": sum(1 for r in done if r["no_answer"]),
        "answered_with_citations": len(with_cites), "answered_without_citations": len(answered) - len(with_cites),
        "answers_with_removed_citations": sum(1 for r in answered if r["flagged"]),
        "agent_fallbacks": sum(1 for r in done if r.get("fallback")),
        "prompts_reported_truncated": sum(1 for r in done if r.get("truncated_prompt")),
        "gold_evidence_in_provided_passages": ci([float(r["gold_in_provided"]) for r in done]) if done else None,
        "citation_precision_micro": {"cited_passages": total_cited, "overlapping_gold": total_hits,
                                     "value": round(total_hits / total_cited, 4) if total_cited else None},
        "citation_precision_macro": ci(macro) if macro else None,
        "faithfulness_sentence_level": ci([float(s["supported"]) for s in scored]) if scored else None,
        "faithfulness_answer_level_all_cited_sentences_supported": ci([float(x) for x in per_answer]) if per_answer else None,
        "cited_sentences_scored": len(scored), "uncited_sentences_not_scored": uncited_sent,
        "mean_seconds_per_question": round(sum(r["seconds"] for r in records) / max(1, len(records)), 2),
    }
    ep = chosen[0]
    out = {
        "meta": {
            **environment(), "task": "answer quality (citation precision, faithfulness)",
            "dataset": qdata["meta"], "sample": {"n_requested": args.n, "seed": args.sample_seed, "n_asked": len(records)},
            "mode": args.mode, "retrieval_scope": "restricted to the question's paper", "retrieval_flags": flags | {"top_k": args.top_k},
            "chat_endpoint": {"name": ep.name, "model": ep.model, "base_url_host": re.sub(r"^https?://([^/]+).*$", r"\1", ep.base_url),
                              "options": ep.options},
            "faithfulness_model": cfg.rerank.model, "entailment_threshold": cfg.rerank.entailment_threshold,
            "http_requests": rt.client.request_count, "max_requests": args.max_requests, "stopped_early": stopped,
            "wall_seconds": round(time.time() - t_start, 1),
        },
        "summary": summary,
        "per_question": records,
    }
    write_json(RESULTS / f"answers-{args.tag}.json", out)
    print(json.dumps(summary, indent=1))
    rt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--endpoint", required=True, help="name of one endpoint from the config; only it is used")
    ap.add_argument("--model", help="override the endpoint's model")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--mode", choices=["pipeline", "agent"], default="pipeline")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--max-requests", type=int, default=200, help="hard cap on HTTP requests, failed attempts included")
    ap.add_argument("--passes", type=int, default=1, help="passes over the (single) endpoint when it rate-limits")
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--config")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
