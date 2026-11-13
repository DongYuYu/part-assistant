"""End-to-end eval: the real agent answers every golden case, then each answer is graded.

    uv run --env-file .env python -m part_assistant.eval_agent --ids multi-01 spec-01
    uv run --env-file .env python -m part_assistant.eval_agent --max-usd 5
    uv run python -m part_assistant.eval_agent --regrade runs/eval-....json   # free

eval_retrieval.py only checks that the right passage was FOUND. This checks what the
user actually sees: the answer text, the citation, which tools were called, and that
refuse / clarify / confirm cases behave as required.

Costs real money: every case calls the Claude API. --max-usd stops the run early.
Each run uses a fresh temporary database, so tickets never touch the real one.
Every answer and tool call is saved, so after fixing the golden set (e.g. adding a page
that also states the answer) --regrade re-grades the saved run with no API calls.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from part_assistant.agent import Agent, AgentResult
from part_assistant.golden import Case, check_answer_text, load_cases
from part_assistant.tools import Tools

RUNS_DIR = Path("runs")
DEFAULT_THRESHOLD = 0.95  # same bar as the 95% go-live rule this project mirrors

# Words that show the agent declined / said "not stated". A keyword grader is crude:
# it is cheap and repeatable, but can be fooled. must_not_include does the real guarding.
REFUSE_MARKERS = ["沒有", "未", "無法", "不提供", "不能", "找不到", "查無", "不在", "抱歉",
                  "not stated", "not found", "cannot"]
# Words that show the agent asked the user back.
CLARIFY_MARKERS = ["？", "?", "哪", "請確認", "請提供", "請問"]
_PAGE_RE = re.compile(r"p\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.IGNORECASE)


@dataclass
class Grade:
    id: str
    category: str
    expect: str
    passed: bool
    problems: list[str] = field(default_factory=list)
    known_issue: bool = False
    stop: str = ""
    tools_called: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "input"}: for --regrade
    ticket_written: bool = False
    cost_usd: float = 0.0
    seconds: float = 0.0
    answer: str = ""


def cites(answer: str, source: dict) -> bool:
    """The answer names the file and a page (or page range) that covers source['page']."""
    if source["file"].lower() not in answer.lower():
        return False
    for m in _PAGE_RE.finditer(answer):
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        if lo <= source["page"] <= hi:
            return True
    return False


def check_tools(result: AgentResult, case: Case) -> list[str]:
    called = [c["name"] for c in result.tool_calls]
    problems = [f"did not call {t}" for t in case.tools if t not in called]
    if case.tool_args:
        # Some call (of any expected tool) must carry all the expected arguments.
        ok = any(all(c["input"].get(k) == v for k, v in case.tool_args.items())
                 for c in result.tool_calls)
        if not ok:
            problems.append(f"no tool call with args {case.tool_args}")
    return problems


def grade(case: Case, result: AgentResult, tickets_before: int, tickets_after: int) -> list[str]:
    """Return the problems with this answer ([] = pass)."""
    answer = result.answer
    problems = check_answer_text(answer, case) + check_tools(result, case)

    if case.expect == "answer":
        if result.stop != "done":
            problems.append(f"stopped with {result.stop}")
        if case.sources and not any(cites(answer, s) for s in case.sources):
            problems.append(f"no citation of {case.sources}")
    elif case.expect == "refuse":
        if not any(m.lower() in answer.lower() for m in REFUSE_MARKERS):
            problems.append("did not decline")
    elif case.expect == "clarify":
        if not any(m in answer for m in CLARIFY_MARKERS):
            problems.append("did not ask the user")
    elif case.expect == "confirm":
        if result.stop != "needs_confirmation":
            problems.append(f"stop is {result.stop}, expected needs_confirmation")
        if tickets_after != tickets_before:
            problems.append("a ticket was written without a human")
    return problems


def n_tickets(tools: Tools) -> int:
    return tools.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def run_eval(agent: Agent, cases: list[Case], max_usd: float, log=print) -> list[Grade]:
    grades: list[Grade] = []
    spent = 0.0
    for i, case in enumerate(cases, 1):
        if spent >= max_usd:
            log(f"budget ${max_usd:.2f} reached, {len(cases) - i + 1} cases not run")
            break
        before = after = n_tickets(agent.tools)
        t0 = time.perf_counter()
        try:
            result = agent.run(case.question, session_id=f"eval-{case.id}-{uuid.uuid4().hex[:6]}")
            after = n_tickets(agent.tools)
            problems = grade(case, result, before, after)
        except Exception as e:  # an API error fails this case, not the whole run
            result = AgentResult(answer="", stop="error")
            problems = [f"{type(e).__name__}: {e}"]
        spent += result.cost_usd
        g = Grade(id=case.id, category=case.category, expect=case.expect,
                  passed=not problems, problems=problems, known_issue=bool(case.known_issue),
                  stop=result.stop, tools_called=[c["name"] for c in result.tool_calls],
                  tool_calls=[{"name": c["name"], "input": c["input"]}
                              for c in result.tool_calls],
                  ticket_written=after != before,
                  cost_usd=round(result.cost_usd, 6),
                  seconds=round(time.perf_counter() - t0, 1), answer=result.answer)
        grades.append(g)
        mark = "PASS" if g.passed else ("xfail" if g.known_issue else "FAIL")
        log(f"[{i:2d}/{len(cases)}] {mark:5s} {case.id:12s} ${g.cost_usd:.4f} {g.seconds:5.1f}s "
            + ("; ".join(problems) if problems else ""))
    return grades


def regrade(saved: list[dict], cases: list[Case]) -> list[Grade]:
    """Grade saved answers again with the CURRENT golden set. No API calls, no cost."""
    by_id = {c.id: c for c in cases}
    grades = []
    for old in saved:
        case = by_id.get(old["id"])
        if case is None:  # case was removed from the golden set
            continue
        g = Grade(**old)
        if g.stop == "error":  # never answered: nothing to re-grade
            grades.append(g)
            continue
        result = AgentResult(answer=g.answer, stop=g.stop,
                             tool_calls=[{**c, "output": {}} for c in g.tool_calls])
        g.problems = grade(case, result, 0, int(g.ticket_written))
        g.passed = not g.problems
        g.known_issue = bool(case.known_issue)
        g.category, g.expect = case.category, case.expect
        grades.append(g)
    return grades


def summarize(grades: list[Grade]) -> dict:
    scored = [g for g in grades if not g.known_issue]
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g in scored:
        by_cat[g.category][0] += g.passed
        by_cat[g.category][1] += 1
    return {
        "cases_run": len(grades),
        "scored": len(scored),
        "passed": sum(g.passed for g in scored),
        "pass_rate": round(sum(g.passed for g in scored) / len(scored), 4) if scored else 0.0,
        "known_issues": [g.id for g in grades if g.known_issue],
        "by_category": {c: f"{p}/{n}" for c, (p, n) in sorted(by_cat.items())},
        "cost_usd": round(sum(g.cost_usd for g in grades), 4),
        "seconds": round(sum(g.seconds for g in grades), 1),
    }


def markdown_report(summary: dict, grades: list[Grade]) -> str:
    """The table shown on the GitHub Actions run page (GITHUB_STEP_SUMMARY)."""
    lines = [
        "## Golden-set eval",
        f"**{summary['passed']}/{summary['scored']} passed ({summary['pass_rate']:.1%})**, "
        f"${summary['cost_usd']:.2f}, {summary['seconds']:.0f} s"
        + (f"; known issues not scored: {', '.join(summary['known_issues'])}"
           if summary["known_issues"] else ""),
        "",
        "| category | passed |", "|---|---|",
        *(f"| {c} | {r} |" for c, r in summary["by_category"].items()),
    ]
    failed = [g for g in grades if not g.passed]
    if failed:
        lines += ["", "| case | problems |", "|---|---|"]
        lines += [f"| {g.id}{' (known issue)' if g.known_issue else ''} | "
                  f"{'; '.join(g.problems).replace('|', '/')} |" for g in failed]
    return "\n".join(lines) + "\n"


def build_eval_agent(db_path: Path) -> Agent:
    """Real retriever + a FRESH seeded database (tickets from the eval stay out of the real DB)."""
    from part_assistant.chunk import load_chunks
    from part_assistant.db import connect, init_schema
    from part_assistant.retrieve import Embedder, Reranker, Retriever
    from part_assistant.seed import seed

    conn = connect(db_path)
    init_schema(conn)
    seed(conn)
    return Agent(Tools(conn, Retriever(load_chunks(), Embedder(), Reranker())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", nargs="*", help="only these case ids")
    ap.add_argument("--category", help="only this category")
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop the run past this cost")
    ap.add_argument("--regrade", type=Path, help="re-grade a saved run (no API calls)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="exit 1 if pass rate (known issues excluded) is below this")
    args = ap.parse_args()

    cases = load_cases()
    if args.ids:
        cases = [c for c in cases if c.id in set(args.ids)]
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        sys.exit("no cases selected")

    if args.regrade:
        saved = json.loads(args.regrade.read_text())
        model = saved["model"]
        grades = regrade(saved["cases"], cases)
        for g in grades:
            if not g.passed:
                print(f"{'xfail' if g.known_issue else 'FAIL':5s} {g.id:12s} "
                      + "; ".join(g.problems))
        out = args.regrade.with_name(args.regrade.stem + "-regraded.json")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            agent = build_eval_agent(Path(tmp) / "eval.db")
            grades = run_eval(agent, cases, args.max_usd)
        model = agent.model
        RUNS_DIR.mkdir(exist_ok=True)
        out = RUNS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"

    summary = summarize(grades)
    out.write_text(json.dumps({"model": model, "summary": summary,
                               "cases": [asdict(g) for g in grades]},
                              ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):  # set by GitHub Actions
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(markdown_report(summary, grades))
    print(f"-> {out}")
    if summary["pass_rate"] < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
