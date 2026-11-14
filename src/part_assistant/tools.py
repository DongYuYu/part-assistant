"""The four tools the agent can call, plus their JSON schemas.

Every tool returns a plain dict. Errors are returned as {"error": ...} (the model
reads them and can recover), never raised.

Safety rules live HERE, in code, not in the prompt:
  - unknown part numbers are never replaced by a "close" one;
  - a high-severity ticket is never created without a human approver, and the
    model has no parameter to approve it with;
  - the idempotency key is chosen by the caller (the agent harness), not the model.
"""

import difflib
import sqlite3
from datetime import datetime, timedelta, timezone

from part_assistant import access
from part_assistant.retrieve import Retriever

TZ = timezone(timedelta(hours=8))  # factory local time (UTC+8)
SEARCH_K = 5


class Tools:
    def __init__(self, conn: sqlite3.Connection, retriever: Retriever | None = None):
        self.conn = conn
        self.retriever = retriever

    # --- helpers

    def _part(self, part_no: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM parts WHERE part_no = ?", (part_no,)).fetchone()

    def _not_found(self, part_no: str) -> dict:
        all_parts = [r[0] for r in self.conn.execute("SELECT part_no FROM parts")]
        similar = difflib.get_close_matches(part_no, all_parts, n=3, cutoff=0.6)
        return {
            "error": f"part_no {part_no!r} not found",
            "similar_part_nos": similar,
            "note": "Do not assume the user meant a similar part number; ask them.",
        }

    # --- tools

    def search_docs(self, query: str, user_id: str | None = None) -> dict:
        """user_id comes from the caller (agent, MCP server), never from the model:
        the model can not ask for a higher clearance."""
        if self.retriever is None:
            return {"error": "document search is not available"}
        chunks = self.retriever.chunks
        clearance = access.clearance_of(self.conn, user_id)
        allowed = access.visible(chunks, clearance, datetime.now(TZ).date().isoformat())
        model = self.retriever.model_filter(query)
        where = (lambda c: allowed(c) and model(c)) if model else allowed
        hits = self.retriever.search(query, k=SEARCH_K, where=where)
        return {"results": [_result(h.chunk) for h in hits]}

    def lookup_part(self, part_no: str) -> dict:
        row = self._part(part_no)
        if row is None:
            return self._not_found(part_no)
        out = dict(row)
        if not out["vendor_mpn"]:
            out["note"] = "No vendor part number on file, so no datasheet is linked."
        return out

    def check_stock(self, part_no: str) -> dict:
        if self._part(part_no) is None:
            return self._not_found(part_no)
        rows = self.conn.execute(
            "SELECT warehouse, qty_on_hand, qty_reserved FROM inventory "
            "WHERE part_no = ? ORDER BY warehouse", (part_no,)).fetchall()
        warehouses = [
            {"warehouse": r["warehouse"], "on_hand": r["qty_on_hand"],
             "reserved": r["qty_reserved"], "available": r["qty_on_hand"] - r["qty_reserved"]}
            for r in rows
        ]
        return {"part_no": part_no, "warehouses": warehouses,
                "total_available": sum(w["available"] for w in warehouses)}

    def create_ticket(self, part_no: str, title: str, description: str, severity: str,
                      idempotency_key: str, approved_by: str | None = None,
                      created_by: str | None = None) -> dict:
        """Create a ticket once per idempotency_key.

        Same key again (e.g. a retry after a timeout) -> the SAME ticket is returned,
        never a second one. High severity without approved_by -> nothing is written.
        """
        if self._part(part_no) is None:
            return self._not_found(part_no)
        if severity not in ("low", "medium", "high"):
            return {"error": f"invalid severity {severity!r}"}
        approved_by = (approved_by or "").strip() or None  # "  " is not a name
        existing = self.conn.execute(
            "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
        if existing:
            return {**dict(existing), "duplicate": True}
        if severity == "high" and not approved_by:
            return {
                "status": "needs_human_confirmation",
                "message": "High severity tickets need a person to confirm. "
                           "Nothing was created. Ask the user to confirm.",
                "draft": {"part_no": part_no, "title": title,
                          "description": description, "severity": severity},
            }
        with self.conn:  # one transaction: the id and the insert cannot interleave
            n = self.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            ticket_id = f"TK-{n + 1:04d}"
            now = datetime.now(TZ).isoformat(timespec="seconds")
            try:
                self.conn.execute(
                    "INSERT INTO tickets (ticket_id, idempotency_key, part_no, title, "
                    "description, severity, status, created_at, created_by, approved_by, "
                    "approved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (ticket_id, idempotency_key, part_no, title, description, severity,
                     "open", now, created_by, approved_by, now if approved_by else None))
            except sqlite3.IntegrityError:
                # A concurrent request with the same key won the race: return its ticket.
                row = self.conn.execute(
                    "SELECT * FROM tickets WHERE idempotency_key = ?",
                    (idempotency_key,)).fetchone()
                if row is None:
                    raise
                return {**dict(row), "duplicate": True}
        row = self.conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        return {**dict(row.fetchone()), "duplicate": False}


# ---------------------------------------------------------------------------
# Schemas shown to the model. strict: the model's input always matches them.
# create_ticket has NO idempotency_key and NO approval field: those are not
# the model's decisions.
# ---------------------------------------------------------------------------

def _result(c) -> dict:
    out = {
        "source": f"{c.file} p.{c.page}" + (f"-{c.page_end}" if c.page_end != c.page else ""),
        "model": c.model,
        "section": c.section,
        "text": c.text.split("\n", 1)[1],  # drop the context line: fields above
    }
    # Only on internal documents, so a public datasheet result is exactly as before.
    if c.revision:
        out["revision"], out["effective"] = c.revision, c.effective
    if c.access_level != "public":
        out["access_level"] = c.access_level
    return out


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


PART_NO = {"type": "string", "description": "Internal part number, e.g. PSU-1600-12V"}

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Search the vendor datasheets (English PDFs). Returns the 5 most relevant "
            "passages with file and page. Put the datasheet model name in the query "
            "(e.g. 'TEC3200 hold-up time'): it restricts the search to that model. "
            "For an internal part number, call lookup_part first to get vendor_mpn."
        ),
        "strict": True,
        "input_schema": _schema({"query": {"type": "string"}}, ["query"]),
    },
    {
        "name": "lookup_part",
        "description": (
            "Look up an internal part number: name, category, spec, lifecycle "
            "(active / nrnd = not recommended for new designs / eol = end of life) and "
            "vendor_mpn (the vendor's model number, used to search datasheets)."
        ),
        "strict": True,
        "input_schema": _schema({"part_no": PART_NO}, ["part_no"]),
    },
    {
        "name": "check_stock",
        "description": (
            "Stock of one internal part number per warehouse. available = on_hand - "
            "reserved; total_available is the sum over warehouses."
        ),
        "strict": True,
        "input_schema": _schema({"part_no": PART_NO}, ["part_no"]),
    },
    {
        "name": "create_ticket",
        "description": (
            "Open a quality/maintenance ticket for an internal part number. "
            "High severity is not created directly: it returns needs_human_confirmation."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "part_no": PART_NO,
                "title": {"type": "string", "description": "Short summary"},
                "description": {"type": "string", "description": "What happened, where"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            ["part_no", "title", "description", "severity"],
        ),
    },
]
