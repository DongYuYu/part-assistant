"""Agent loop tests with a scripted fake client: no API key, no cost, same result every run."""

from types import SimpleNamespace as NS

import pytest

from part_assistant.agent import Agent, idempotency_key
from part_assistant.db import connect, init_schema
from part_assistant.seed import seed
from part_assistant.tools import Tools


def usage(i=100, o=50):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=0,
              cache_creation_input_tokens=0)


def text(t, **kw):
    return NS(content=[NS(type="text", text=t)], stop_reason="end_turn",
              usage=kw.get("usage", usage()), model="claude-opus-5-5")


def tool_use(*calls, u=None):
    blocks = [NS(type="tool_use", id=f"tu_{i}", name=n, input=a)
              for i, (n, a) in enumerate(calls)]
    return NS(content=blocks, stop_reason="tool_use", usage=u or usage(), model="claude-opus-5-5")


class FakeClient:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, *responses, repeat_last=False):
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        # Snapshot: the agent keeps appending to the same list after this call.
        self.requests.append({**kw, "messages": list(kw["messages"])})
        if len(self.responses) > 1 or not self.repeat_last:
            r = self.responses.pop(0)
        else:
            r = self.responses[0]
        if isinstance(r, Exception):  # scripted failure: the API is down at this step
            raise r
        return r


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


def n_tickets(tools):
    return tools.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


LOW = {"part_no": "FAN-4028-HS", "title": "noise", "description": "line 3", "severity": "low"}
HIGH = {"part_no": "PSU-3200-54V", "title": "smoke", "description": "burn-in", "severity": "high"}


def test_runs_tool_then_answers(tools):
    client = FakeClient(tool_use(("check_stock", {"part_no": "PSU-1600-54V"})),
                        text("總共可用 513 顆。"))
    r = Agent(tools, client).run("PSU-1600-54V 還有多少？")
    assert r.stop == "done" and r.answer == "總共可用 513 顆。"
    assert r.tool_calls[0]["output"]["total_available"] == 513
    # The second request carries the tool result, linked by tool_use_id.
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user" and last["content"][0]["tool_use_id"] == "tu_0"
    assert [e["type"] for e in r.trace] == ["llm", "tool", "llm"]


def test_parallel_tool_calls_return_in_one_message(tools):
    client = FakeClient(tool_use(("check_stock", {"part_no": "FAN-4028-HS"}),
                                 ("lookup_part", {"part_no": "FAN-4028-HS"})), text("ok"))
    Agent(tools, client).run("q")
    results = client.requests[1]["messages"][-1]["content"]
    assert [x["tool_use_id"] for x in results] == ["tu_0", "tu_1"]


def test_stops_at_max_steps(tools):
    client = FakeClient(tool_use(("lookup_part", {"part_no": "FAN-4028-HS"})), repeat_last=True)
    r = Agent(tools, client, max_steps=3).run("loop forever")
    assert r.stop == "max_steps" and len(client.requests) == 3


def test_stops_at_cost_budget(tools):
    big = usage(i=200_000, o=0)  # 200k input tokens x $4/M = $0.80 > $0.50
    client = FakeClient(tool_use(("lookup_part", {"part_no": "FAN-4028-HS"}), u=big),
                        repeat_last=True)
    r = Agent(tools, client).run("expensive")
    assert r.stop == "budget" and len(client.requests) == 1


def test_tool_errors_are_reported_not_raised(tools):
    client = FakeClient(tool_use(("no_such_tool", {}), ("lookup_part", {"part_no": "X"})),
                        text("找不到"))
    r = Agent(tools, client).run("q")
    results = client.requests[1]["messages"][-1]["content"]
    assert all(x["is_error"] for x in results) and r.stop == "done"


def test_high_severity_stops_for_a_human_then_approve_once(tools):
    before = n_tickets(tools)
    client = FakeClient(tool_use(("create_ticket", HIGH)), text("需要你確認後才會建立。"))
    agent = Agent(tools, client)
    r = agent.run("冒煙了，開高嚴重度工單")
    assert r.stop == "needs_confirmation" and r.pending_ticket["severity"] == "high"
    assert n_tickets(tools) == before, "model alone must not create it"

    first = agent.approve(r.pending_ticket, approver="QA lead")
    second = agent.approve(r.pending_ticket, approver="QA lead")  # double click / retry
    assert first["ticket_id"] == second["ticket_id"] and second["duplicate"]
    assert n_tickets(tools) == before + 1
    assert first["approved_by"] == "QA lead"  # who approved is stored, not just printed


def test_model_asking_twice_in_one_session_opens_one_ticket(tools):
    """E.g. the model saw a timeout and retried: the harness gives both calls the same key."""
    before = n_tickets(tools)
    client = FakeClient(tool_use(("create_ticket", LOW)), tool_use(("create_ticket", LOW)),
                        text("已建立 TK-0003"))
    r = Agent(tools, client).run("開低嚴重度工單", session_id="s1")
    assert n_tickets(tools) == before + 1
    assert r.tool_calls[1]["output"]["duplicate"] is True


def test_idempotency_key_depends_on_session():
    assert idempotency_key("s1", "P", "low") == idempotency_key("s1", "P", "low")
    assert idempotency_key("s1", "P", "low") != idempotency_key("s2", "P", "low")


def test_request_uses_strict_tools_and_no_key_in_code(tools):
    client = FakeClient(text("hi"))
    Agent(tools, client).run("hi")
    req = client.requests[0]
    assert {t["name"] for t in req["tools"]} == {
        "search_docs", "lookup_part", "check_stock", "create_ticket"}
    assert "api_key" not in req


@pytest.mark.parametrize("reason,stop", [
    ("end_turn", "done"), ("max_tokens", "truncated"),
    ("model_context_window_exceeded", "truncated"), ("refusal", "refusal"),
    ("something_new", "something_new"),  # unknown: never silently "done"
])
def test_stop_reason_mapping(tools, reason, stop):
    resp = NS(content=[NS(type="text", text="x")], stop_reason=reason, usage=usage(),
              model="claude-opus-5-5")
    assert Agent(tools, FakeClient(resp)).run("q").stop == stop


def test_show_prints_each_exchange(tools, capsys):
    client = FakeClient(tool_use(("check_stock", {"part_no": "PSU-1600-54V"})), text("513"))
    Agent(tools, client, show=True).run("庫存？")
    out = capsys.readouterr().out
    assert "第 1 次呼叫 API" in out and "第 2 次呼叫 API" in out
    assert "check_stock" in out and "tool_result" in out and "stop_reason=end_turn" in out
