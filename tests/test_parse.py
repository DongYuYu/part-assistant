"""Parser tests.

Unit tests use hand-written HTML and always run.
Integration tests use the real datasheets and are skipped if data/raw is empty
(run `uv run python -m part_assistant.download` first).
"""

from pathlib import Path

import pytest

from part_assistant.parse import apply_footnote_markers, clean_text, parse_html_table, parse_pdf

# ---------------------------------------------------------------- unit tests


def test_rowspan_copies_value_into_every_covered_row():
    html = ("<table><tr><th>Param</th><th>Cond</th><th>Val</th></tr>"
            "<tr><td rowspan='2'>Efficiency</td><td>10%</td><td>90</td></tr>"
            "<tr><td>20%</td><td>94</td></tr></table>")
    rows, has_header = parse_html_table(html)
    assert has_header
    assert rows[1] == ["Efficiency", "10%", "90"]
    assert rows[2] == ["Efficiency", "20%", "94"]  # the merged cell reaches row 2


def test_colspan_keeps_columns_aligned():
    html = ("<table><tr><td>CSE</td><td colspan='3'>±4%</td><td>-</td></tr>"
            "<tr><td>Ripple</td><td>a</td><td>b</td><td>c</td><td>1%</td></tr></table>")
    rows, _ = parse_html_table(html)
    assert rows[0] == ["CSE", "±4%", "±4%", "±4%", "-"]
    assert len(rows[0]) == len(rows[1])  # still a rectangle


def test_br_and_sup_inside_cells():
    rows, _ = parse_html_table("<table><tr><td>via the I<sup>2</sup> C<br/>bus</td></tr></table>")
    assert rows[0][0] == "via the I2C bus"


def test_footnote_marker_only_changes_the_marked_value():
    markers = [("0 A", "1")]
    assert apply_footnote_markers("0 A 1", markers) == "0 A[^1]"
    assert apply_footnote_markers("0 A", markers) == "0 A"  # standby output: no mark
    assert apply_footnote_markers("10 A 15", markers) == "10 A 15"  # \b: '15' is not mark '1'


def test_clean_text_collapses_whitespace():
    assert clean_text("  a \n  b\t c ") == "a b c"


# --------------------------------------------------------- integration tests

RAW = Path("data/raw")
needs_pdfs = pytest.mark.skipif(not any(RAW.glob("*.pdf")), reason="datasheets not downloaded")


def _rows(blocks, page):
    for b in blocks:
        if b.kind == "table" and b.table.page <= page <= b.table.page_end:
            yield from b.table.rows


def _row(blocks, page, needle):
    return next(r for r in _rows(blocks, page) if any(needle == c for c in r))


@pytest.fixture(scope="module")
def tec3200():
    return parse_pdf(RAW / "bel-tec3200-54.pdf")


@pytest.fixture(scope="module")
def csu2400():
    return parse_pdf(RAW / "ae-csu2400at.pdf")


@pytest.fixture(scope="module")
def tec1600():
    return parse_pdf(RAW / "bel-tec1600.pdf")


@needs_pdfs
def test_borderless_first_row_of_group_gets_its_own_name(tec3200):
    """Name is vertically centered: forward-fill would wrongly say 'iTHD' here."""
    assert _row(tec3200, 2, "230 VAC / 60 Hz, 10% load")[0] == "Efficiency"
    assert _row(tec3200, 2, "230 VAC / 60 Hz, 100% load")[0] == "Efficiency"


@needs_pdfs
def test_borderless_wrapped_name_is_joined(tec3200):
    assert _row(tec3200, 2, "230 VAC / 50 & 60 Hz, 10% load")[0] == (
        "iTHD (Total Harmonic Distortion)")


@needs_pdfs
def test_borderless_last_row_of_group(tec3200):
    assert _row(tec3200, 2, "@ 2700 W")[0] == "Hold-up Time"


@needs_pdfs
def test_footnote_marks_restored(csu2400):
    row = _row(csu2400, 2, "Output Current")
    assert row[1] == "0 A[^1]"  # main output min current has footnote 1
    assert row[4] == "0 A"  # standby min current has none
    text = " ".join(b.text for b in csu2400 if b.kind == "text")
    assert "Minimum loading is 5% of rated current" in text


@needs_pdfs
def test_table_continued_over_three_pages(tec1600):
    table = next(b.table for b in tec1600
                 if b.kind == "table" and any("MFR_IOUT_MAX" in r for r in b.table.rows))
    assert (table.page, table.page_end) == (12, 14)
    assert table.rows[0][:2] == ["COMMAND CODE", "COMMAND NAME"]


@needs_pdfs
def test_running_title_removed(tec1600):
    assert not any(b.text.strip() == "# TEC1600-12-074xA" for b in tec1600 if b.kind == "text")
