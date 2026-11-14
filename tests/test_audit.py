"""Audit log: who asked what, which tools ran, what changed. No network, no cost."""

import json
import sqlite3

import anthropic
import httpx2
import pytest
from mcp.client.client import Client
from test_agent import HIGH, LOW, FakeClient, text, tool_use
from test_mcp_server import human

from part_assistant import audit
from part_assistant.agent import Agent, save_trace
from part_assistant.db import connect, init_schema
from part_assistant.local_agent import LocalAgent
from part_assistant.mcp_server import build_server
from part_assistant.seed import seed
from part_assistant.tools import Tools

REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def log(tools, **kw):
    return audit.entries(tools.conn, limit=100, **kw)


def write(tools, **over):
    row = {"user_id": "alice", "session_id": "s", "channel": "cli", "action": "ask",
           "detail": {"question": "q"}, **over}
    return audit.record(tools.conn, **row)


# --- the table

def test_rows_can_not_be_changed_or_deleted(tools):
    write(tools)
    for sql in ("UPDATE audit_log SET user_id = 'mallory'", "DELETE FROM audit_log"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            tools.conn.execute(sql)
    assert log(tools)[0]["user_id"] == "alice"


@pytest.mark.parametrize("bad", [{"user_id": "  "}, {"channel": "web"}, {"action": "edit"}])
def test_the_table_refuses_bad_rows(tools, bad):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        write(tools, **bad)


def test_detail_must_be_json(tools):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        tools.conn.execute(
            "INSERT INTO audit_log (at, user_id, session_id, channel, action, detail) "
            "VALUES ('t', 'alice', 's', 'cli', 'ask', 'not json')")


def test_filter_by_user_and_action(tools):
    write(tools, user_id="alice")
    write(tools, user_id="bob")
    write(tools, user_id="bob", action="approve", detail={"ticket_id": "TK-1"})
    assert [e["user_id"] for e in log(tools, user="bob")] == ["bob", "bob"]
    assert [e["action"] for e in log(tools, user="bob", action="approve")] == ["approve"]


# --- who is the user

def test_current_user_order(monkeypatch):
    monkeypatch.setenv(audit.USER_ENV, "env-user")
    assert audit.current_user(" alice ") == "alice"
    assert audit.current_user("  ") == "env-user"  # blank is not a name
    monkeypatch.delenv(audit.USER_ENV)
    assert audit.current_user() not in ("", None)  # the operating-system account


# --- the agent

def test_a_question_is_one_ask_row_with_tools_and_ticket(tools):
    client = FakeClient(tool_use(("search_docs", {"query": "x"}), ("create_ticket", LOW)),
                        text("已開單"))
    r = Agent(tools, client).run("風扇有噪音，開低嚴重度工單", user_id="alice")
    [e] = log(tools)
    assert (e["user_id"], e["action"], e["channel"]) == ("alice", "ask", "cli")
    assert e["session_id"] == r.session_id
    d = e["detail"]
    assert d["question"] == "風扇有噪音，開低嚴重度工單" and d["answer"] == "已開單"
    assert [t["name"] for t in d["tools"]] == ["search_docs", "create_ticket"]
    ticket = d["tools"][1]["result"]
    assert ticket["created_by"] == "alice" and ticket["ticket_id"].startswith("TK-")
    row = tools.conn.execute("SELECT created_by FROM tickets WHERE ticket_id = ?",
                             (ticket["ticket_id"],)).fetchone()
    assert row["created_by"] == "alice"  # stored with the ticket, same transaction


def test_search_results_are_summarized_not_copied(tools):
    out = {"results": [{"source": "a.pdf p.2", "text": "long datasheet text " * 100}]}
    assert audit.summarize("search_docs", out) == {"sources": ["a.pdf p.2"]}


def test_a_failed_question_is_still_audited(tools):
    bad_key = anthropic.AuthenticationError(
        "HTTP 401", response=httpx2.Response(401, request=REQ), body=None)
    client = FakeClient(tool_use(("create_ticket", LOW)), bad_key)
    with pytest.raises(anthropic.AuthenticationError):
        Agent(tools, client).run("q", user_id="alice")
    [e] = log(tools)
    assert "AuthenticationError" in e["detail"]["error"]
    assert e["detail"]["tools"][0]["name"] == "create_ticket"  # the write before the error


def test_degraded_answer_keeps_the_user(tools):
    from test_fallback import FakeOllama, ol_text

    down = anthropic.APIConnectionError(request=REQ)
    agent = Agent(tools, FakeClient(down),
                  fallback=LocalAgent(tools, chat=FakeOllama(ol_text("120 台"))))
    r = agent.run("q", user_id="alice")
    [e] = log(tools)
    assert r.user_id == "alice" and e["user_id"] == "alice" and e["detail"]["degraded"]


def test_approval_is_its_own_row_with_requester_and_approver(tools):
    agent = Agent(tools, FakeClient(tool_use(("create_ticket", HIGH)), text("請確認")))
    r = agent.run("冒煙了", user_id="alice")
    out = agent.approve(r.pending_ticket, approver="QA lead", user_id="bob",
                        session_id=r.session_id)
    ask, approve = log(tools)
    assert ask["detail"]["tools"][0]["result"]["status"] == "needs_human_confirmation"
    assert approve["action"] == "approve" and approve["user_id"] == "bob"
    assert approve["detail"]["requested_by"] == "alice"
    assert approve["detail"]["approver"] == "QA lead"
    assert approve["detail"]["ticket_id"] == out["ticket_id"]
    assert (out["created_by"], out["approved_by"]) == ("alice", "QA lead")


def test_trace_file_has_the_user(tools, tmp_path):
    r = Agent(tools, FakeClient(text("ok"))).run("q", user_id="alice")
    saved = json.loads(save_trace("q", r, tmp_path).read_text())
    assert (saved["user_id"], saved["session_id"]) == ("alice", r.session_id)


# --- MCP

@pytest.mark.anyio
async def test_every_mcp_tool_call_is_audited(tools):
    server = build_server(tools, session_id="s1", user_id="carol")
    async with Client(server, elicitation_callback=human(name="QA lead")) as client:
        await client.call_tool("check_stock", {"part_no": "FAN-4028-HS"})
        await client.call_tool("create_ticket", HIGH)
    stock, ticket = log(tools)
    assert (stock["channel"], stock["action"], stock["user_id"]) == ("mcp", "tool", "carol")
    assert stock["detail"]["result"]["total_available"] is not None
    t = ticket["detail"]["result"]
    assert (t["created_by"], t["approved_by"]) == ("carol", "QA lead")


@pytest.mark.anyio
async def test_declined_mcp_ticket_is_audited_too(tools):
    async with Client(build_server(tools, user_id="carol"),
                      elicitation_callback=human("decline")) as client:
        await client.call_tool("create_ticket", HIGH)
    [e] = log(tools)
    assert e["detail"]["result"]["status"] == "not_created"
    assert "decline" in e["detail"]["result"]["reason"]
    assert e["detail"]["input"]["severity"] == "high"
