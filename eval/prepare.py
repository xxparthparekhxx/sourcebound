"""Build the evaluation corpus from Qasper (allenai/qasper, CC BY 4.0).

Qasper: questions written by NLP practitioners about NLP papers, with evidence paragraphs marked by
annotators. The dataset repository ships a loading script, so this reads the Hub's auto-converted
parquet files (revision refs/convert/parquet) with pyarrow.

    python eval/prepare.py --split validation --papers 40 --dev-papers 10 --seed 0

Writes eval/data/qasper/{docs,dev_docs}/<paper-id>.md (one Markdown file per paper: title, abstract,
section headings, paragraphs, figure/table captions) and questions.json / dev_questions.json.
The dev papers are disjoint from the evaluation papers and are only used to choose settings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import DEFAULT_DATA, environment, sample, write_json

FLOAT_PREFIX = "FLOAT SELECTED:"


def clean_par(text: str) -> str:
    text = " ".join(text.split())
    return ("\\" + text) if text.startswith("#") else text  # keep a paragraph from being read as a heading


def paper_markdown(row: dict) -> str:
    out = [f"# {row['title'].strip()}", "", "## Abstract", "", clean_par(row["abstract"])]
    ft = row["full_text"]
    for name, paras in zip(ft["section_name"], ft["paragraphs"]):
        heading = (name or "").split(":::")[-1].strip()
        if heading:
            out += ["", f"## {heading}"]
        for p in paras:
            if p and p.strip():
                out += ["", clean_par(p)]
    caps = [c for c in (row.get("figures_and_tables") or {}).get("caption", []) if c and c.strip()]
    if caps:
        out += ["", "## Figures and tables"]
        for c in caps:
            out += ["", clean_par(c)]
    return "\n".join(out) + "\n"


def usable_questions(row: dict) -> list[dict]:
    """Answerable questions that have evidence paragraphs. Evidence is kept per annotator."""
    qs = []
    qas = row["qas"]
    for qid, question, ans in zip(qas["question_id"], qas["question"], qas["answers"]):
        annotators = []
        for a in ans["answer"]:
            if a["unanswerable"]:
                continue
            ev = []
            for e in a["evidence"]:
                e = e.strip()
                if e.startswith(FLOAT_PREFIX):
                    e = e[len(FLOAT_PREFIX):].strip()
                e = " ".join(e.split())
                if e and e not in ev:
                    ev.append(e)
            if ev:
                kind = "yes_no" if a["yes_no"] is not None else "extractive" if a["extractive_spans"] else "free_form"
                text = a["free_form_answer"] or "; ".join(a["extractive_spans"]) or ("yes" if a["yes_no"] else "no")
                annotators.append({"evidence": ev, "answer_type": kind, "answer": text})
        if annotators:
            qs.append({"qid": qid, "paper_id": row["id"], "question": question.strip(), "annotators": annotators})
    return qs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--papers", type=int, default=40)
    ap.add_argument("--dev-papers", type=int, default=10)
    ap.add_argument("--min-questions", type=int, default=3, help="papers need at least this many usable questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--config", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.split == "train":
        raise SystemExit("do not evaluate on the train split: openjev-flash's training mix includes Qasper train")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/qasper", f"qasper/{a.split}/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    rows = pq.read_table(path).to_pylist()
    eligible = []
    for r in rows:
        qs = usable_questions(r)
        if len(qs) >= a.min_questions:
            eligible.append((r, qs))
    print(f"{len(rows)} papers in {a.split}; {len(eligible)} with >= {a.min_questions} usable questions")
    picked = sample(sorted(eligible, key=lambda x: x[0]["id"]), a.papers + a.dev_papers, a.seed)
    sets = {"": picked[: a.papers], "dev_": picked[a.papers :]}

    for prefix, chosen in sets.items():
        docdir = a.out / f"{prefix}docs"
        docdir.mkdir(parents=True, exist_ok=True)
        for old in docdir.glob("*.md"):
            old.unlink()
        questions = []
        papers = {}
        for row, qs in sorted(chosen, key=lambda x: x[0]["id"]):
            (docdir / f"{row['id']}.md").write_text(paper_markdown(row))
            papers[row["id"]] = row["title"]
            questions.extend(qs)
        write_json(a.out / f"{prefix}questions.json", {
            "meta": {
                "dataset": "allenai/qasper", "split": a.split, "revision": "refs/convert/parquet", "seed": a.seed,
                "papers": len(papers), "questions": len(questions), **environment(),
            },
            "papers": papers,
            "questions": questions,
        })
        print(f"{prefix or 'eval '}set: {len(papers)} papers, {len(questions)} questions")


if __name__ == "__main__":
    main()
