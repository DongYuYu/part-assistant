"""MCP server: the same four tools, for any MCP host (Claude Code, Claude Desktop, ...).

    uv run python -m part_assistant.mcp_server        # speaks MCP over stdin/stdout

With MCP the HOST (e.g. Claude Code) runs the agent loop and the model; this process only
runs tools. So the rules that agent.py enforced as the "harness" move in here:
  - the idempotency key is chosen here, never by the model;
  - a high-severity ticket asks the USER through MCP elicitation (a form in the host).
    The question is asked by a resolver that runs BEFORE the tool body, so the body only
    runs once there is an answer; it writes only if the person accepted and typed a name.
    A host without elicitation gets an error and nothing is written.

All tools are `async def`: they run on the server's event loop thread, which is the
thread that opened the SQLite connection (sqlite3 refuses cross-thread use). Trade-off:
a search blocks the loop for ~0.5 s, fine for one user over stdio.

stdout carries the protocol: never print() here (logs go to stderr).
"""

import uuid
from collections.abc import Callable
from typing import Annotated, Literal

from mcp.server.mcpserver import Elicit, ElicitationResult, MCPServer, Resolve
from pydantic import BaseModel, Field

from part_assistant.agent import idempotency_key
from part_assistant.retrieve import Retriever
from part_assistant.tools import TOOL_SCHEMAS, Tools

DESCRIPTIONS = {t["name"]: t["description"] for t in TOOL_SCHEMAS}  # one source of truth

INSTRUCTIONS = """\
Parts assistant for a server power-supply factory (demo data, public datasheets).
Facts must come from these tools; cite datasheet facts as (file p.N). For an internal part
number, call lookup_part first and search_docs with its vendor_mpn. Never replace an
unknown part number with a similar one: ask the user."""

Severity = Literal["low", "medium", "high"]


class Approval(BaseModel):
    """The form the host shows the user for a high-severity ticket."""

    approver: str = Field(description="你的名字（記錄為核准人）")


def build_server(tools: Tools, load_retriever: Callable[[], Retriever] | None = None,
                 session_id: str | None = None) -> MCPServer:
    """load_retriever is called on the first search, so the server starts in a second
    instead of waiting for two GPU models to load."""
    # stdio: one server process per host session, so one id per process.
    session_id = session_id or uuid.uuid4().hex
    mcp = MCPServer(name="part-assistant", instructions=INSTRUCTIONS)

    def key(part_no: str, severity: str) -> str:
        return idempotency_key(session_id, part_no, severity)

    async def ask_a_person(part_no: str, title: str, description: str,
                           severity: Severity) -> Elicit[Approval] | None:
        """Runs before create_ticket. Must have no side effects: it may run more than once."""
        if severity != "high" or tools._part(part_no) is None:
            return None  # nothing to approve (an unknown part fails in the tool)
        exists = tools.conn.execute("SELECT 1 FROM tickets WHERE idempotency_key = ?",
                                    (key(part_no, severity),)).fetchone()
        if exists:
            return None  # already approved and written: a retry must not ask again
        return Elicit(f"要開立【高嚴重度】工單嗎？\n料號：{part_no}\n標題：{title}\n"
                      f"說明：{description}\n同意請填你的名字；不同意請按拒絕。", Approval)

    @mcp.tool(description=DESCRIPTIONS["search_docs"])
    async def search_docs(query: str) -> dict:
        if tools.retriever is None and load_retriever is not None:
            tools.retriever = load_retriever()
        return tools.search_docs(query)

    @mcp.tool(description=DESCRIPTIONS["lookup_part"])
    async def lookup_part(part_no: str) -> dict:
        return tools.lookup_part(part_no)

    @mcp.tool(description=DESCRIPTIONS["check_stock"])
    async def check_stock(part_no: str) -> dict:
        return tools.check_stock(part_no)

    @mcp.tool(description=DESCRIPTIONS["create_ticket"]
              + " For high severity the user is asked to confirm in a form.")
    async def create_ticket(
        part_no: str, title: str, description: str, severity: Severity,
        approval: Annotated[ElicitationResult[Approval], Resolve(ask_a_person)],
    ) -> dict:
        approver = None
        if severity == "high" and approval.action == "accept" and approval.data is not None:
            approver = approval.data.approver.strip() or None
        result = tools.create_ticket(part_no, title, description, severity,
                                     key(part_no, severity), approved_by=approver)
        if result.get("status") == "needs_human_confirmation":  # declined, cancelled, no name
            why = "no approver name given" if approval.action == "accept" else approval.action
            return {"status": "not_created", "reason": f"the user did not approve: {why}"}
        return result

    return mcp


def main() -> None:
    from part_assistant.chunk import load_chunks
    from part_assistant.db import connect
    from part_assistant.retrieve import Embedder, Reranker

    def load_retriever() -> Retriever:
        return Retriever(load_chunks(), Embedder(), Reranker())

    build_server(Tools(connect()), load_retriever).run("stdio")


if __name__ == "__main__":
    main()
