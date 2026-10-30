import pytest

from docgraph.citations import CitationFilter, validate_citations


def test_valid_markers_pass_through_unchanged():
    r = validate_citations("Cats sleep a lot [1]. They eat meat [2][3].", {1, 2, 3})
    assert r.text == "Cats sleep a lot [1]. They eat meat [2][3]." and r.used == [1, 2, 3] and r.flagged == []


def test_marker_for_a_passage_that_was_not_given_is_flagged_and_removed():
    r = validate_citations("The moon is large [7]. Jupiter has moons [1].", {1, 2, 3})
    assert r.text == "The moon is large. Jupiter has moons [1]."    # no stray space before the full stop
    assert r.flagged == [7] and r.used == [1]


def test_partially_valid_lists_keep_the_valid_numbers():
    r = validate_citations("Both agree [1, 9, 2].", {1, 2})
    assert r.text == "Both agree [1, 2]." and r.flagged == [9] and r.used == [1, 2]


def test_zero_and_ranges():
    r = validate_citations("a [0] b [1-3] c [2-5]", {1, 2, 3})
    assert r.flagged == [0, 4, 5]
    assert r.text == "a b [1-3] c [2, 3]"


def test_non_citation_brackets_are_left_alone():
    text = "See array[i] and [link](http://x) and [ ] and [a1] and [1234] and the year [2023]."
    r = validate_citations(text, {1})
    assert r.text == text and r.flagged == [] and r.used == []


def test_unterminated_marker_at_the_end_is_kept_as_text():
    r = validate_citations("incomplete [1", {1})
    assert r.text == "incomplete [1" and r.used == []


@pytest.mark.parametrize("size", [1, 2, 3, 5, 100])
def test_streaming_in_pieces_gives_the_same_result_as_the_whole_text(size):
    text = "Answer one [1]. Bad one [9] here. Mixed [2, 8], and brackets [x] and [3]!\nNext line [10]."
    whole = validate_citations(text, {1, 2, 3})
    f = CitationFilter({1, 2, 3})
    out = "".join(f.feed(text[i : i + size]) for i in range(0, len(text), size)) + f.flush()
    assert out == whole.text
    assert (f.used, f.flagged) == (whole.used, whole.flagged) == ([1, 2, 3], [9, 8, 10])


def test_a_marker_is_held_back_until_it_closes():
    f = CitationFilter({1})
    assert f.feed("fact [") == "fact"        # the open marker (and the space before it) are not released yet
    assert f.feed("1") == ""
    assert f.feed("]") == " [1]" and f.used == [1]


def test_allowed_set_can_grow_while_streaming():
    allowed = {1}
    f = CitationFilter(allowed)
    first = f.feed("A [2] ")                   # 2 is not allowed yet: removed
    allowed.add(2)
    assert first + f.feed("B [2]") == "A B [2]" and f.used == [2] and f.flagged == [2]
