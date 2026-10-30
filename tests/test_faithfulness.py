import numpy as np

from sourcebound.faithfulness import check_answer, marker_numbers, sentence_pairs, strip_markers


class Scorer:
    def predict(self, pairs):
        return np.array([[0.0, 0.8 if "yes" in h.lower() else 0.2, 0.0] for _, h in pairs], dtype=np.float32)


def test_marker_numbers_handle_lists_and_ranges():
    assert marker_numbers("A [1]. B [2, 3] and [4-6]; C [7;8].") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert strip_markers("Yes [1] it   is [2, 3].") == "Yes it is ."


def test_only_markers_that_name_a_shown_passage_count():
    items = sentence_pairs("Yes it is [1][2]. Also this [7]. And no marker at all here.", {1: "one", 2: "two"})
    assert [i["cited"] for i in items] == [[1, 2], [], []]
    assert items[0]["pair"] == ("one\n\ntwo", "Yes it is .") and items[1]["pair"] is None and items[2]["pair"] is None


def test_short_sentences_are_not_scored_and_threshold_decides():
    out = check_answer(Scorer(), "Yes. Yes it is true [1]. No it is not [1].", {1: "p"}, threshold=0.5)
    assert [o["supported"] for o in out] == [None, True, False]
    assert out[1]["p_entail"] == 0.8 and out[0]["p_entail"] is None
    assert check_answer(Scorer(), "Yes it is true [1].", {1: "p"}, threshold=0.9)[0]["supported"] is False
    assert check_answer(Scorer(), "", {1: "p"}) == []
