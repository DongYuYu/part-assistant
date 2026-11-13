import pytest

from part_assistant.db import connect, init_schema
from part_assistant.seed import seed
from part_assistant.tools import TOOL_SCHEMAS, Tools


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)
    return Tools(conn)


def n_tickets(tools: Tools) -> int:
    return tools.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def ticket(tools: Tools, severity="low", key="k1", approved_by=None, part_no="FAN-4028-HS"):
    return tools.create_ticket(part_no, "noise", "line 3 fan noise", severity, key,
                               approved_by=approved_by)


# --- lookup / stock

def test_lookup_known_part(tools):
    p = tools.lookup_part("PSU-3200-54V")
    assert p["vendor_mpn"] == "TEC3200-54-074NA" and p["lifecycle"] == "active"


def test_unknown_part_lists_similar_but_never_picks_one(tools):
    r = tools.lookup_part("PSU-1600-12")
    assert "error" in r and "PSU-1600-12V" in r["similar_part_nos"]
    assert "part_no" not in r, "must not silently answer for PSU-1600-12V"


def test_stock_totals_match_seed(tools):
    assert tools.check_stock("PSU-1600-54V")["total_available"] == 513
    wh = {w["warehouse"]: w["available"] for w in tools.check_stock("PSU-1600-12V")["warehouses"]}
    assert wh["WH-TPE"] == 7
    assert tools.check_stock("CAP-1000UF-63V")["total_available"] == 0


def test_stock_of_unknown_part_is_an_error(tools):
    assert "error" in tools.check_stock("PSU-9999-XX")


# --- tickets

def test_low_ticket_is_created(tools):
    r = ticket(tools)
    assert r["ticket_id"] == "TK-0003" and r["status"] == "open" and not r["duplicate"]


def test_same_idempotency_key_never_creates_a_second_ticket(tools):
    """A timeout retry re-sends the same request with the same key."""
    first = ticket(tools, key="retry-me")
    before = n_tickets(tools)
    again = ticket(tools, key="retry-me")
    assert again["ticket_id"] == first["ticket_id"] and again["duplicate"]
    assert n_tickets(tools) == before


def test_high_severity_needs_a_human(tools):
    before = n_tickets(tools)
    r = ticket(tools, severity="high", key="h1")
    assert r["status"] == "needs_human_confirmation"
    assert n_tickets(tools) == before, "nothing may be written before approval"
    ok = ticket(tools, severity="high", key="h1", approved_by="QA lead")
    assert ok["ticket_id"].startswith("TK-") and n_tickets(tools) == before + 1


def test_ticket_for_unknown_part_is_refused(tools):
    before = n_tickets(tools)
    assert "error" in ticket(tools, part_no="PSU-9999-XX")
    assert n_tickets(tools) == before


def test_search_without_retriever_is_an_error_not_a_crash(tools):
    assert "error" in tools.search_docs("TEC3200 efficiency")


# --- schemas

def test_model_cannot_approve_or_choose_the_key():
    props = next(t for t in TOOL_SCHEMAS if t["name"] == "create_ticket")["input_schema"]
    assert "approved_by" not in props["properties"]
    assert "idempotency_key" not in props["properties"]


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda t: t["name"])
def test_schemas_are_strict(schema):
    s = schema["input_schema"]
    assert schema["strict"] and s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])
