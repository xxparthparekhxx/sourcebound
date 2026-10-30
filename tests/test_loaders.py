import pymupdf
import pytest

from docgraph.ingest.loaders import (
    UnsupportedFormat, load_file, parse_html, parse_markdown, parse_text,
)


def test_markdown_front_matter_title_and_headings():
    doc = parse_markdown("---\ntitle: My Doc\n---\n# Heading One\n\nBody one.\n\n## Sub\n\nBody two.\n")
    assert doc.title == "My Doc"
    assert [(b.heading, b.text) for b in doc.blocks] == [("Heading One", "Body one."), ("Sub", "Body two.")]


def test_plain_text_paragraphs():
    doc = parse_text("line one\nline two\n\n\nsecond paragraph\n", default_title="notes")
    assert [b.text for b in doc.blocks] == ["line one line two", "second paragraph"]
    assert doc.title == "notes"


def test_html_headings_paragraphs_and_boilerplate():
    html = """<html><head><title>Page Title</title><style>p{}</style></head><body>
    <nav>Menu Home About</nav><h1>Top</h1><p>Hello <b>bold</b> world.</p>
    <script>var x = 1;</script><h2>Part &amp; Two</h2><p>Second.</p><footer>copyright</footer></body></html>"""
    doc = parse_html(html)
    assert doc.title == "Page Title"
    assert [(b.heading, b.text) for b in doc.blocks] == [("Top", "Hello bold world."), ("Part & Two", "Second.")]


def make_pdf(path, pages):
    """pages: list of lists of (text, fontsize, fontname) drawn top to bottom; empty list = blank page."""
    doc = pymupdf.open()
    for items in pages:
        page = doc.new_page()
        y = 80
        for text, size, font in items:
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + 120), text, fontsize=size, fontname=font)
            y += 130 if size < 12 else 40
    doc.save(path)
    doc.close()


BODY = "This is an ordinary body paragraph with enough words in it to be treated as prose text."


def test_pdf_pages_headings_and_body(tmp_path):
    p = tmp_path / "a.pdf"
    make_pdf(p, [
        [("Introduction", 20, "hebo"), (BODY, 10, "helv")],
        [("Methods", 20, "hebo"), (BODY + " Second page.", 10, "helv")],
    ])
    doc = load_file(p)
    assert doc.n_pages == 2 and doc.format == "pdf"
    got = [(b.page, b.heading) for b in doc.blocks]
    assert got == [(1, "Introduction"), (2, "Methods")]
    assert not doc.skipped_pages


def test_scanned_page_is_reported_as_skipped_when_ocr_is_off(tmp_path):
    p = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 500, 200), BODY, fontsize=10)
    page2 = doc.new_page()  # a page that is only an image: no text layer
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page2.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    doc.save(p)
    doc.close()

    parsed = load_file(p)
    assert [s.page for s in parsed.skipped_pages] == [2]
    assert "OCR is off" in parsed.skipped_pages[0].reason
    assert {b.page for b in parsed.blocks} == {1}


def test_scanned_page_goes_through_ocr_when_enabled(tmp_path):
    p = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    doc.save(p)
    doc.close()

    seen = []

    def fake_ocr(png: bytes) -> str:
        seen.append(png[:4])
        return "# Scanned Heading\n\nText recognised from the image of the page."

    parsed = load_file(p, ocr=fake_ocr)
    assert seen == [b"\x89PNG"]
    assert parsed.ocr_pages == [1] and not parsed.skipped_pages
    assert [(b.page, b.heading) for b in parsed.blocks] == [(1, "Scanned Heading")]


def test_ocr_failure_skips_the_page_instead_of_aborting(tmp_path):
    p = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    doc.save(p)
    doc.close()

    def broken(_png):
        raise RuntimeError("endpoint down")

    parsed = load_file(p, ocr=broken)
    assert parsed.skipped_pages and "OCR failed" in parsed.skipped_pages[0].reason


def test_unsupported_extension(tmp_path):
    f = tmp_path / "x.docx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFormat):
        load_file(f)
