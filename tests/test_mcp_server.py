"""MCP server tests: a real MCP client talks to the server in memory (no subprocess, no LLM)."""

import json

import mcp_types as types
import pytest
from mcp.client.client import Client
from mcp.shared.exceptions import MCPError

from part_assistant.db import connect, init_schema
from part_assistant.mcp_server import build_server
from part_assistant.seed import seed
from part_assistant.tools import Tools

HIGH = {"part_no": "PSU-3200-54V", "title": "smoke", "description": "line 2 burn-in",
        "severity": "high"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


def n_tickets(tools):
    return tools.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def data(result) -> dict:
    """A tool's dict comes back as structured content (and as JSON text for old hosts)."""
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


def human(action="accept", name="QA lead", calls=None):
    """An elicitation callback standing in for the person in front of the host."""
    async def answer(context, params):
        if calls is not None:
            calls.append(params.message)
        content = {"approver": name} if action == "accept" else None
        return types.ElicitResult(action=action, content=content)
    return answer


@pytest.mark.anyio
async def test_lists_the_four_tools_with_schemas(tools):
    async with Client(build_server(tools)) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
    assert set(listed) == {"search_docs", "lookup_part", "check_stock", "create_ticket"}
    props = listed["create_ticket"].input_schema["properties"]
    assert props["severity"]["enum"] == ["low", "medium", "high"]
    assert "idempotency_key" not in props and "approved_by" not in props
    assert "approval" not in props, "filled by asking the user, never by the model"


@pytest.mark.anyio
async def test_stock_over_mcp(tools):
    async with Client(build_server(tools)) as client:
        r = await client.call_tool("check_stock", {"part_no": "PSU-1600-54V"})
    assert data(r)["total_available"] == 513


@pytest.mark.anyio
async def test_low_ticket_twice_in_one_session_is_one_ticket(tools):
    before = n_tickets(tools)
    low = {**HIGH, "part_no": "FAN-4028-HS", "severity": "low"}
    async with Client(build_server(tools)) as client:
        first = data(await client.call_tool("create_ticket", low))
        again = data(await client.call_tool("create_ticket", low))
    assert first["ticket_id"] == again["ticket_id"] and again["duplicate"]
    assert n_tickets(tools) == before + 1


@pytest.mark.anyio
async def test_high_ticket_is_written_only_after_a_person_accepts(tools):
    asked = []
    before = n_tickets(tools)
    async with Client(build_server(tools), elicitation_callback=human(calls=asked)) as client:
        r = data(await client.call_tool("create_ticket", HIGH))
    assert asked and "高嚴重度" in asked[0]
    assert r["ticket_id"].startswith("TK-") and n_tickets(tools) == before + 1
    assert r["approved_by"] == "QA lead"  # the name typed in the form, stored for audit


@pytest.mark.anyio
@pytest.mark.parametrize("action,name", [("decline", ""), ("cancel", ""), ("accept", "  ")])
async def test_high_ticket_declined_or_unnamed_writes_nothing(tools, action, name):
    before = n_tickets(tools)
    async with Client(build_server(tools), elicitation_callback=human(action, name)) as client:
        r = data(await client.call_tool("create_ticket", HIGH))
    assert r["status"] == "not_created" and n_tickets(tools) == before


@pytest.mark.anyio
async def test_host_without_elicitation_gets_an_error_and_nothing_is_written(tools):
    before = n_tickets(tools)
    with pytest.raises(ExceptionGroup) as err:
        async with Client(build_server(tools)) as client:  # no elicitation_callback
            await client.call_tool("create_ticket", HIGH)
    assert err.group_contains(MCPError, match="elicitation capability", depth=None)
    assert n_tickets(tools) == before


@pytest.mark.anyio
async def test_approved_high_ticket_retried_is_not_asked_again(tools):
    asked = []
    server = build_server(tools, session_id="s1")
    async with Client(server, elicitation_callback=human(calls=asked)) as client:
        first = data(await client.call_tool("create_ticket", HIGH))
        again = data(await client.call_tool("create_ticket", HIGH))  # e.g. a timeout retry
    assert len(asked) == 1 and again["ticket_id"] == first["ticket_id"] and again["duplicate"]


@pytest.mark.anyio
async def test_works_on_the_older_protocol_too(tools):
    """Hosts may still speak the 2025 protocol (server-to-client request mid-call)."""
    async with Client(build_server(tools), mode="legacy", elicitation_callback=human()) as c:
        r = data(await c.call_tool("create_ticket", HIGH))
    assert r["ticket_id"].startswith("TK-")


@pytest.mark.anyio
async def test_retriever_loads_lazily_on_first_search(tools):
    loads = []

    class FakeRetriever:
        def model_filter(self, q):
            return None

        def search(self, q, k, where):
            return []

    def load():
        loads.append(1)
        return FakeRetriever()

    async with Client(build_server(tools, load)) as client:
        assert loads == []  # starting the server did not load the models
        await client.call_tool("search_docs", {"query": "x"})
        await client.call_tool("search_docs", {"query": "y"})
    assert loads == [1]


def test_setup_check_names_the_command_to_run(tmp_path, tools):
    from part_assistant.mcp_server import check_setup

    problems = check_setup(tmp_path / "none.db", tmp_path / "none.jsonl")
    assert any("part_assistant.chunk" in p for p in problems)
    assert any("part_assistant.seed" in p for p in problems)
    chunks = tmp_path / "c.jsonl"
    chunks.write_text("")
    assert check_setup(tmp_path / "t.db", chunks) == []  # the seeded fixture DB
