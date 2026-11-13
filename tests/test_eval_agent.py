"""The grader itself must be right, or the score means nothing. Scripted fake client: no cost."""

import pytest
from test_agent import FakeClient, text, tool_use

from part_assistant.agent import Agent
from part_assistant.db import connect, init_schema
from part_assistant.eval_agent import cites, run_eval, summarize
from part_assistant.golden import load_cases
from part_assistant.seed import seed
from part_assistant.tools import Tools

CASES = {c.id: c for c in load_cases()}
SRC = {"file": "bel-tec3200-54.pdf", "page": 2}


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


def run(tools, case_id, *responses, max_usd=5.0):
    agent = Agent(tools, FakeClient(*responses))
    return run_eval(agent, [CASES[case_id]], max_usd, log=lambda *_: None)[0]


# --- citation

@pytest.mark.parametrize("answer,ok", [
    ("9 ms（bel-tec3200-54.pdf p.2）", True),
    ("9 ms (BEL-TEC3200-54.PDF P. 2)", True),
    ("9 ms（bel-tec3200-54.pdf p.1-3）", True),   # a range that covers page 2
    ("9 ms（bel-tec3200-54.pdf p.12）", False),   # p.12 is not p.2
    ("9 ms（p.2）", False),                        # no file name
])
def test_cites(answer, ok):
    assert cites(answer, SRC) is ok


# --- grading each kind of case

LOOKUP = ("lookup_part", {"part_no": "PSU-3200-54V"})
SEARCH = ("search_docs", {"query": "TEC3200 hold-up"})


def test_answer_case_passes(tools):
    g = run(tools, "multi-01", tool_use(LOOKUP), tool_use(SEARCH),
            text("9 ms（bel-tec3200-54.pdf p.2）"))
    assert g.passed, g.problems


def test_answer_without_citation_fails(tools):
    g = run(tools, "multi-01", tool_use(LOOKUP), tool_use(SEARCH), text("9 ms"))
    assert not g.passed and any("citation" in p for p in g.problems)


def test_answer_that_skipped_a_tool_fails(tools):
    g = run(tools, "multi-01", tool_use(SEARCH), text("9 ms（bel-tec3200-54.pdf p.2）"))
    assert "did not call lookup_part" in g.problems


def test_refuse_case(tools):
    assert run(tools, "refuse-01", text("規格書沒有寫保固年限。")).passed
    g = run(tools, "refuse-01", text("保固 5 年。"))  # borrowed from another vendor
    assert not g.passed


def test_clarify_must_ask_and_must_not_add_up(tools):
    stock = ("check_stock", {"part_no": "PSU-1600-12V"})
    assert run(tools, "clarify-02", tool_use(stock),
               text("PSU-1600 對應兩個料號，你是指哪一個？")).passed
    g = run(tools, "clarify-02", tool_use(stock), text("總共 741 顆。"))
    assert not g.passed


def test_confirm_case_passes_only_without_a_written_ticket(tools):
    high = ("create_ticket", {"part_no": "PSU-3200-54V", "title": "smoke",
                              "description": "line 2 burn-in", "severity": "high"})
    g = run(tools, "ticket-01", tool_use(high), text("需要你確認後才會建立。"))
    assert g.passed and g.stop == "needs_confirmation", g.problems


def test_wrong_tool_args_fail(tools):
    low_wrong = ("create_ticket", {"part_no": "FAN-4028-HS", "title": "noise",
                                   "description": "line 3", "severity": "medium"})
    g = run(tools, "ticket-02", tool_use(low_wrong), text("已建立 TK-0003"))
    assert any("args" in p for p in g.problems)


# --- run control

def test_budget_stops_the_run(tools):
    agent = Agent(tools, FakeClient(text("hi"), repeat_last=True))
    cases = [CASES["refuse-02"]] * 3
    assert len(run_eval(agent, cases, max_usd=0.0, log=lambda *_: None)) == 0


def test_api_error_fails_one_case_and_the_run_goes_on(tools):
    class Boom(FakeClient):
        def _create(self, **kw):
            raise TimeoutError("API timeout")

    grades = run_eval(Agent(tools, Boom()), [CASES["refuse-02"], CASES["refuse-04"]], 5.0,
                      log=lambda *_: None)
    assert len(grades) == 2 and all(g.stop == "error" and not g.passed for g in grades)


def test_known_issues_are_not_scored(tools):
    known = next(c for c in CASES.values() if c.known_issue)
    agent = Agent(tools, FakeClient(text("不知道"), text("規格書沒有寫保固。")))
    grades = run_eval(agent, [known, CASES["refuse-01"]], 5.0, log=lambda *_: None)
    s = summarize(grades)
    assert s["scored"] == 1 and s["pass_rate"] == 1.0 and known.id in s["known_issues"]


def test_regrade_uses_the_current_golden_set_without_calling_the_api(tools):
    from dataclasses import asdict, replace

    from part_assistant.eval_agent import regrade

    g = run(tools, "multi-01", tool_use(LOOKUP), tool_use(SEARCH),
            text("9 ms（bel-tec3200-54.pdf p.3）"))
    assert not g.passed  # golden set only lists p.2
    fixed = replace(CASES["multi-01"], sources=[SRC, {**SRC, "page": 3}])
    again = regrade([asdict(g)], [fixed])[0]
    assert again.passed, again.problems


def test_markdown_report_lists_failures():
    from part_assistant.eval_agent import Grade, markdown_report

    bad = Grade(id="x-01", category="spec", expect="answer", passed=False,
                problems=["missing one of ['9 ms']", "no tool call with args {'a': 'b|c'}"])
    ok = Grade(id="x-02", category="spec", expect="answer", passed=True)
    md = markdown_report(summarize([bad, ok]), [bad, ok])
    assert "1/2 passed (50.0%)" in md and "| spec | 1/2 |" in md
    row = next(line for line in md.splitlines() if line.startswith("| x-01 "))
    assert row.count("|") == 3, "a '|' inside a problem must not add a table column"
    assert "x-02" not in md  # passing cases are not listed


def test_each_case_is_reported_as_soon_as_it_finishes(tools):
    seen = []
    agent = Agent(tools, FakeClient(text("規格書沒有寫保固。"), text("規格書沒有寫保固。")))
    run_eval(agent, [CASES["refuse-01"], CASES["refuse-01"]], 5.0, log=lambda *_: None,
             on_grade=lambda g: seen.append(g.id))
    assert seen == ["refuse-01", "refuse-01"]
