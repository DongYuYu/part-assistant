"""The golden set is data, and data has typos. Validate it like code."""

from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from part_assistant.download import load_sources
from part_assistant.golden import CATEGORIES, EXPECTS, TOOLS, Case, check_answer_text, load_cases
from part_assistant.seed import PARTS

CASES = load_cases()
SOURCE_FILES = {s["file"] for s in load_sources()}
PART_NOS = {p[0] for p in PARTS}


def test_ids_unique():
    dupes = [i for i, n in Counter(c.id for c in CASES).items() if n > 1]
    assert not dupes


def test_size_and_coverage():
    assert 35 <= len(CASES) <= 60
    cats = Counter(c.category for c in CASES)
    assert set(cats) == CATEGORIES, "every category must have at least one case"
    assert cats["refuse"] + cats["clarify"] >= 6, "need enough 'should not answer' cases"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_case_is_well_formed(case: Case):
    assert case.category in CATEGORIES
    assert case.expect in EXPECTS
    assert case.why, "every case must say what it tests"
    assert set(case.tools) <= TOOLS
    if case.expect == "answer":
        assert case.must_include, "an 'answer' case needs something to check"
    for s in case.sources:
        assert s["file"] in SOURCE_FILES, f"unknown file {s['file']}"
    if "part_no" in case.tool_args:
        assert case.tool_args["part_no"] in PART_NOS


@pytest.mark.skipif(not any(Path("data/raw").glob("*.pdf")), reason="datasheets not downloaded")
@pytest.mark.parametrize("case", [c for c in CASES if c.sources], ids=lambda c: c.id)
def test_source_pages_exist(case: Case):
    for s in case.sources:
        n_pages = pymupdf.open(Path("data/raw") / s["file"]).page_count
        assert 1 <= s["page"] <= n_pages


# --- the matcher itself

def _case(**kw) -> Case:
    return Case(id="x", category="spec", question="q", expect="answer", why="w", **kw)


def test_match_ignores_case_and_spaces():
    c = _case(must_include=[["9 ms", "9 毫秒"]])
    assert check_answer_text("Hold-up time 為 9ms。", c) == []
    assert check_answer_text("約 9 毫秒", c) == []


def test_every_group_must_match():
    c = _case(must_include=[["73.5"], ["185"]])
    assert check_answer_text("寬 73.5 mm", c) == ["missing one of ['185']"]


def test_forbidden_value_fails():
    c = _case(must_include=[["92"]], must_not_include=["94%"])
    assert check_answer_text("效率 92%，TEC3200 則是 94%", c) == ["contains forbidden '94%'"]
