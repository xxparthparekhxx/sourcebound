import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from common import ChunkText, Evidence, bootstrap_ci, question_metrics  # noqa: E402
from answers import marker_numbers, strip_markers  # noqa: E402

PAR_A = "We train the model with the Adam optimizer for ten epochs using a batch size of thirty two and a learning rate of 0.001."
PAR_B = "The dataset contains forty thousand sentences collected from news articles published between 2010 and 2015."
OTHER = "Completely unrelated text about cooking pasta with tomatoes basil garlic and olive oil on a quiet Sunday afternoon."


def chunk(*parts):
    return ChunkText("\n\n".join(parts))


def test_a_chunk_covers_evidence_it_contains_even_with_different_whitespace_and_case():
    ev = Evidence(PAR_A)
    assert ev.covered_by(chunk("Intro text.", PAR_A.upper().replace(" ", "  "), "Trailing text."))
    assert not ev.covered_by(chunk(OTHER))


def test_half_of_the_paragraph_is_enough_but_not_less():
    ev = Evidence(PAR_A)
    words = PAR_A.split()
    assert ev.covered_by(chunk(" ".join(words[: int(len(words) * 0.65)])))      # 65% of the words -> more than half of the 5-grams
    assert not ev.covered_by(chunk(" ".join(words[: int(len(words) * 0.3)])))


def test_short_evidence_must_appear_whole():
    ev = Evidence("Table 2 results")
    assert ev.covered_by(chunk("see table 2 results below"))
    assert not ev.covered_by(chunk("table 3 results")) and not Evidence("").covered_by(chunk("x"))


def test_recall_takes_the_best_annotator_and_mrr_the_first_hit_of_any():
    anns = [[Evidence(PAR_A), Evidence(PAR_B)], [Evidence(PAR_B)]]
    ranked = [chunk(OTHER), chunk(PAR_B), chunk(OTHER), chunk(PAR_A)]
    m = question_metrics(ranked, anns, ks=(2, 5))
    assert m["recall@2"] == 1.0           # annotator 2 needs only PAR_B, found at rank 2
    assert m["recall@5"] == 1.0           # annotator 1 is complete at rank 4
    assert m["rr"] == 0.5 and m["first_hit_rank"] == 2
    at1 = question_metrics(ranked, anns, ks=(1,))
    assert at1["recall@1"] == 0.0


def test_partial_recall_and_no_hit():
    anns = [[Evidence(PAR_A), Evidence(PAR_B)]]
    m = question_metrics([chunk(PAR_A), chunk(OTHER)], anns, ks=(5,))
    assert m["recall@5"] == 0.5 and m["rr"] == 1.0
    miss = question_metrics([chunk(OTHER)] * 3, anns, ks=(5,))
    assert miss["recall@5"] == 0.0 and miss["rr"] == 0.0 and miss["first_hit_rank"] is None


def test_one_chunk_may_cover_several_evidence_paragraphs():
    m = question_metrics([chunk(PAR_A, PAR_B)], [[Evidence(PAR_A), Evidence(PAR_B)]], ks=(5,))
    assert m["recall@5"] == 1.0


def test_bootstrap_interval_contains_the_mean_and_is_deterministic():
    vals = [0.0, 1.0] * 50
    mean, lo, hi = bootstrap_ci(vals)
    assert mean == 0.5 and lo < 0.5 < hi and bootstrap_ci(vals) == (mean, lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_marker_helpers():
    s = "Cats sleep a lot [1][3], and eat meat [2, 4]. Range [5-6]."
    assert marker_numbers(s) == [1, 3, 2, 4, 5, 6]
    assert strip_markers(s) == "Cats sleep a lot , and eat meat . Range ."
