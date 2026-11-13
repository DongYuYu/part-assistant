from dataclasses import replace
from pathlib import Path

import pytest

from part_assistant.chunk import (
    OVERLAP_TOKENS,
    TABLE_TOKENS,
    TEXT_TOKENS,
    build,
    chunk_document,
    dedup_row,
    find_footnotes,
    n_tokens,
    table_bodies,
)
from part_assistant.golden import check_answer_text, load_cases
from part_assistant.parse import Block, Table

META = {"file": "x.pdf", "vendor": "V", "model": "M"}


def text_block(page: int, text: str) -> Block:
    return Block("text", page, text=text)


def table_block(page: int, rows, has_header=True) -> Block:
    return Block("table", page, table=Table(rows, has_header, page, page))


# --- tables

def test_dedup_row_merges_colspan_copies():
    assert dedup_row(["Input", "Input", "Input"]) == ["Input"]
    assert dedup_row(["Vout", "12 V", "12 V", "54 V"]) == ["Vout", "12 V", "54 V"]


def test_big_table_is_split_by_rows_and_repeats_header():
    rows = [["Param", "Value"]] + [[f"row{i} " + "x" * 40, str(i)] for i in range(100)]
    bodies = table_bodies(Table(rows, True, 1, 1))
    assert len(bodies) > 1
    for b in bodies:
        assert b.startswith("| Param | Value |\n| --- | --- |")
        assert n_tokens(b) <= TABLE_TOKENS
    all_rows = "\n".join(bodies)
    assert all(f"row{i} " in all_rows for i in range(100)), "no row may be lost"


def test_footnotes_found_in_any_list_style():
    t = Table([["Output Current", "0 A[^1]"], ["CSE[^2]", "±4%"]], False, 1, 1)
    after = ["Note:\n\n1. Minimum loading is 5%.\n\n- 2 From 5% to 100%.\n\n3. unrelated"]
    assert find_footnotes(t, after) == [
        "[^1]: Minimum loading is 5%.", "[^2]: From 5% to 100%."]


def test_table_without_markers_gets_no_footnotes():
    t = Table([["Weight", "740 g"]], False, 1, 1)
    assert find_footnotes(t, ["1. Some numbered list item"]) == []


# --- text

def test_headings_set_section_and_prefix():
    blocks = [text_block(3, "## **INPUT**\n\nAC input 90-264 V.\n\n## **OUTPUT**\n\n12 V main.")]
    chunks = chunk_document(blocks, META)
    assert [c.section for c in chunks] == ["INPUT", "OUTPUT"]
    assert chunks[0].text == "[V | M | x.pdf | p.3 | INPUT]\nAC input 90-264 V."


def test_long_section_is_split_with_overlap():
    pars = [f"Paragraph {i} " + "word " * 30 for i in range(40)]
    chunks = chunk_document([text_block(1, "\n\n".join(pars))], META)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert n_tokens(a.text) <= TEXT_TOKENS + 40  # + the context line
        last_par = a.text.split("\n\n")[-1]
        assert n_tokens(last_par) <= OVERLAP_TOKENS
        assert last_par in b.text, "next chunk must start with the tail of this one"


def test_overlap_does_not_cross_headings():
    a = "Alpha " * 50
    chunks = chunk_document([text_block(1, f"# A\n\n{a}\n\n# B\n\nBeta.")], META)
    assert "Alpha" not in chunks[-1].text


def test_table_chunk_keeps_section_page_and_notes():
    blocks = [
        text_block(2, "## ELECTRICAL"),
        table_block(2, [["Param", "Value"], ["Output Current", "0 A[^1]"]]),
        text_block(2, "1. Minimum loading is 5%."),
    ]
    table = [c for c in chunk_document(blocks, META) if c.kind == "table"][0]
    assert table.section == "ELECTRICAL"
    assert table.text.startswith("[V | M | x.pdf | p.2 | ELECTRICAL]\n| Param | Value |")
    assert table.text.endswith("[^1]: Minimum loading is 5%.")


def test_chunk_ids_are_stable_and_content_based():
    blocks = [text_block(1, "# T\n\nSame text.")]
    a, b = chunk_document(blocks, META), chunk_document(blocks, META)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    c = chunk_document([text_block(1, "# T\n\nOther text.")], META)
    assert a[0].chunk_id != c[0].chunk_id


def test_reserved_fields_have_safe_defaults():
    c = chunk_document([text_block(1, "Hello.")], META)[0]
    assert c.access_level == "public" and c.revision == ""


# --- real datasheets: is every golden answer present in the chunks of its page?

HAS_PDFS = any(Path("data/raw").glob("*.pdf"))
ANSWER_CASES = [c for c in load_cases() if c.expect == "answer" and c.sources]


@pytest.fixture(scope="module")
def real_chunks(tmp_path_factory):
    return build(out=tmp_path_factory.mktemp("chunks") / "chunks.jsonl")


@pytest.mark.skipif(not HAS_PDFS, reason="datasheets not downloaded")
@pytest.mark.parametrize("case", ANSWER_CASES, ids=lambda c: c.id)
def test_golden_answer_is_in_source_page_chunks(case, real_chunks):
    """Upper bound for RAG: if the answer is not in any chunk of the cited page,
    no retriever can find it. Table cells are joined ("| 740 | g |" -> "740g").
    must_not_include is about the ANSWER, not the page: the page may well list
    other values (e.g. 94% at another load), so only must_include is checked."""
    if case.known_issue:
        pytest.xfail(case.known_issue)
    text = "\n".join(
        c.text for s in case.sources for c in real_chunks
        if c.file == s["file"] and c.page <= s["page"] <= c.page_end
    )
    assert check_answer_text(text.replace("|", ""), replace(case, must_not_include=[])) == []
