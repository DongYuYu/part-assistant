"""Load the golden set (eval/golden.toml) and check answers against it."""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_PATH = Path("eval/golden.toml")
CATEGORIES = {
    "spec", "table", "footnote", "similar", "tool", "multi", "refuse", "clarify", "ticket",
}
EXPECTS = {"answer", "refuse", "clarify", "confirm"}
TOOLS = {"search_docs", "lookup_part", "check_stock", "create_ticket"}


@dataclass
class Case:
    id: str
    category: str
    question: str
    expect: str
    why: str
    must_include: list[list[str]] = field(default_factory=list)  # groups of alternatives
    must_not_include: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tool_args: dict = field(default_factory=dict)
    known_issue: str = ""


def load_cases(path: Path = GOLDEN_PATH) -> list[Case]:
    with path.open("rb") as f:
        raw = tomllib.load(f)["case"]
    cases = []
    for c in raw:
        # Normalise: a group may be written as "x" or ["x", "y"]; store as list.
        groups = [[g] if isinstance(g, str) else list(g) for g in c.pop("must_include", [])]
        cases.append(Case(must_include=groups, **c))
    return cases


def _norm(s: str) -> str:
    """Lowercase and drop all whitespace: '9 ms' == '9ms', 'Platinum' == 'platinum'."""
    return re.sub(r"\s+", "", s).lower()


def check_answer_text(answer: str, case: Case) -> list[str]:
    """Return a list of problems with the answer text ([] means it passes)."""
    a = _norm(answer)
    problems = []
    for group in case.must_include:
        if not any(_norm(alt) in a for alt in group):
            problems.append(f"missing one of {group}")
    for bad in case.must_not_include:
        if _norm(bad) in a:
            problems.append(f"contains forbidden {bad!r}")
    return problems
