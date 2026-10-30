from docgraph.ingest.chunking import Block, chunk_blocks, overlap_tail, split_long
from docgraph.ingest.loaders import parse_markdown


def para(n: int, word: str = "alpha") -> str:
    return " ".join([word] * n) + "."


def test_small_section_is_one_chunk_with_heading_and_page():
    blocks = [Block("First paragraph here.", page=2, heading="Intro", section=1),
              Block("Second paragraph here.", page=3, heading="Intro", section=1)]
    chunks = chunk_blocks(blocks, size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].heading == "Intro"
    assert (chunks[0].page, chunks[0].page_end) == (2, 3)
    assert "First paragraph" in chunks[0].text and "Second paragraph" in chunks[0].text


def test_chunks_never_cross_a_heading():
    blocks = [Block("Text under A that is long enough to count.", 1, "A", 1),
              Block("Text under B that is long enough to count.", 1, "B", 2)]
    chunks = chunk_blocks(blocks, size=1000, overlap=100)
    assert [c.heading for c in chunks] == ["A", "B"]
    assert "under B" not in chunks[0].text


def test_same_heading_text_in_different_sections_stays_separate():
    blocks = [Block("Notes from the first part, long enough.", 1, "Notes", 1),
              Block("Notes from the second part, long enough.", 1, "Notes", 2)]
    assert len(chunk_blocks(blocks, size=1000, overlap=100)) == 2


def test_paragraphs_are_packed_up_to_size_and_overlap_is_applied():
    blocks = [Block(f"Sentence number {i} is here. " + para(30, f"w{i}"), 1, "H", 1) for i in range(6)]
    size, overlap = 400, 80
    chunks = chunk_blocks(blocks, size=size, overlap=overlap)
    assert len(chunks) >= 3
    assert all(len(c.text) <= size for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = overlap_tail(prev.text.split("\n\n")[-1], overlap)
        assert tail and nxt.text.startswith(tail)  # modest overlap carried into the next chunk


def test_no_overlap_across_sections():
    blocks = [Block("Alpha section body text, quite long enough to chunk.", 1, "A", 1),
              Block("Beta section body text, quite long enough to chunk.", 1, "B", 2)]
    chunks = chunk_blocks(blocks, size=1000, overlap=200)
    assert not chunks[1].text.startswith("Alpha")


def test_long_paragraph_is_split_at_sentences():
    text = " ".join(f"This is sentence {i} of a very long paragraph." for i in range(60))
    pieces = split_long(text, 300)
    assert len(pieces) > 5 and all(len(p) <= 300 for p in pieces)
    assert all(p.endswith(".") for p in pieces)
    assert " ".join(pieces) == text


def test_unbreakable_text_is_hard_split():
    pieces = split_long("x" * 1000, 300)
    assert all(len(p) <= 300 for p in pieces) and "".join(pieces) == "x" * 1000


def test_overlap_tail_starts_at_a_boundary():
    text = "One two three. Four five six seven eight nine ten eleven twelve."
    tail = overlap_tail(text, 30)
    assert text.endswith(tail) and len(tail) <= 30
    assert not tail.startswith(("our", "ive"))  # not mid-word


def test_markdown_structure_end_to_end():
    md = "# Title\n\nIntro paragraph that is long enough.\n\n## Methods\n\nWe did a thing.\nSecond line.\n\n```py\n# not a heading\nx = 1\n```\n"
    doc = parse_markdown(md, default_title="f")
    assert doc.title == "Title"
    chunks = chunk_blocks(doc.blocks, size=200, overlap=20)
    heads = [c.heading for c in chunks]
    assert heads == ["Title", "Methods"]
    assert "# not a heading" in chunks[1].text
