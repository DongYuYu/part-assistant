"""Audit log: who asked what, which tools ran, what changed.

Traces (traces/*.json) are for debugging: anyone can delete or edit them. The audit
log is a table in the same database as the tickets, and the database refuses to
UPDATE or DELETE its rows (see db.py).

There is no login in this project, so the user id is what the caller says it is:
--user, then PART_ASSISTANT_USER, then the operating-system account.

  uv run python -m part_assistant.audit                 # last 20 entries
  uv run python -m part_assistant.audit --user alice --action approve
"""

import argparse
import getpass
import json
import os
import sqlite3
from datetime import datetime

from part_assistant.tools import TZ

USER_ENV = "PART_ASSISTANT_USER"


def current_user(explicit: str | None = None) -> str:
    for name in (explicit, os.environ.get(USER_ENV)):
        if name and name.strip():
            return name.strip()
    try:
        return getpass.getuser()
    except Exception:  # no account name (some containers): still record something
        return "unknown"


def summarize(name: str, output: dict) -> dict:
    """What an auditor needs from a tool result, without copying whole datasheet pages."""
    if "error" in output:
        return {"error": str(output["error"])[:200]}
    if name == "create_ticket":
        keys = ("ticket_id", "status", "severity", "duplicate", "created_by", "approved_by",
                "reason")
        return {k: output[k] for k in keys if k in output}
    if name == "search_docs":
        return {"sources": [r["source"] for r in output.get("results", [])]}
    if name == "check_stock":
        return {"total_available": output.get("total_available")}
    if name == "lookup_part":
        return {"part_no": output.get("part_no"), "vendor_mpn": output.get("vendor_mpn")}
    return {}


def record(conn: sqlite3.Connection, user_id: str, session_id: str, channel: str,
           action: str, detail: dict) -> int:
    with conn:  # its own transaction: committed even if the caller fails later
        cur = conn.execute(
            "INSERT INTO audit_log (at, user_id, session_id, channel, action, detail) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.now(TZ).isoformat(timespec="seconds"), user_id, session_id, channel,
             action, json.dumps(detail, ensure_ascii=False, default=str)))
    return cur.lastrowid


def entries(conn: sqlite3.Connection, user: str | None = None, action: str | None = None,
            limit: int = 20) -> list[dict]:
    where, args = [], []
    if user:
        where.append("user_id = ?")
        args.append(user)
    if action:
        where.append("action = ?")
        args.append(action)
    sql = ("SELECT * FROM audit_log" + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY id DESC LIMIT ?")
    rows = conn.execute(sql, [*args, limit]).fetchall()
    return [{**dict(r), "detail": json.loads(r["detail"])} for r in reversed(rows)]


def one_line(e: dict) -> str:
    d = e["detail"]
    if e["action"] == "ask":
        tools = ",".join(t["name"] for t in d.get("tools", [])) or "-"
        what = f"{d.get('question', '')[:40]!r} tools={tools} stop={d.get('stop')}"
        tickets = [t["result"].get("ticket_id") for t in d.get("tools", [])
                   if t["name"] == "create_ticket" and t["result"].get("ticket_id")]
        if tickets:
            what += f" tickets={','.join(tickets)}"
    elif e["action"] == "approve":
        what = f"approved {d.get('ticket_id')} as {d.get('approver')!r}"
    else:
        what = f"{d.get('name')}({json.dumps(d.get('input'), ensure_ascii=False)[:60]})"
    return f"#{e['id']:<5} {e['at']}  {e['user_id']:<12} {e['channel']:<4} {e['action']:<7} {what}"


def main() -> None:
    from part_assistant.db import connect, init_schema

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--user")
    ap.add_argument("--action", choices=["ask", "tool", "approve"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", action="store_true", help="full entries as JSON")
    args = ap.parse_args()
    conn = connect()
    init_schema(conn)  # an old database gets the table (and the triggers)
    found = entries(conn, args.user, args.action, args.limit)
    if args.json:
        print(json.dumps(found, ensure_ascii=False, indent=2))
    else:
        for e in found:
            print(one_line(e))


if __name__ == "__main__":
    main()
