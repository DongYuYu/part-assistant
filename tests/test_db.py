import sqlite3

import pytest

from part_assistant.db import connect
from part_assistant.seed import PARTS, SEED_TIME, WAREHOUSES, seed

# Name the columns: a bare "INSERT ... VALUES" breaks when a column is added.
COLS = "(ticket_id, idempotency_key, part_no, title, description, severity, status, created_at)"


@pytest.fixture
def conn():
    """A fresh in-memory database for each test (fast, no files left behind)."""
    c = connect(":memory:")
    seed(c)
    yield c
    c.close()


def test_seed_counts(conn):
    assert conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == len(PARTS)
    inv = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
    assert inv == len(PARTS) * len(WAREHOUSES)


def test_seed_is_deterministic():
    """Same seed -> identical inventory. Evals depend on this."""
    rows = []
    for _ in range(2):
        c = connect(":memory:")
        seed(c)
        rows.append(c.execute("SELECT * FROM inventory ORDER BY part_no, warehouse").fetchall())
        c.close()
    assert [tuple(r) for r in rows[0]] == [tuple(r) for r in rows[1]]


def test_out_of_stock_part_has_zero_everywhere(conn):
    total = conn.execute(
        "SELECT SUM(qty_on_hand) FROM inventory WHERE part_no = 'PSU-0800-12V'"
    ).fetchone()[0]
    assert total == 0


def test_duplicate_idempotency_key_rejected(conn):
    """The UNIQUE constraint must block a second ticket with the same key."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO tickets {COLS} VALUES ('TK-9999', 'seed-0001', NULL, 't', 'd',"
            " 'low', 'open', ?)",
            (SEED_TIME,),
        )


def test_invalid_severity_rejected(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO tickets {COLS} VALUES ('TK-9999', 'k', NULL, 't', 'd', 'urgent',"
            " 'open', ?)",
            (SEED_TIME,),
        )


def test_reserved_cannot_exceed_on_hand(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE inventory SET qty_reserved = qty_on_hand + 1 WHERE part_no = 'PSU-1600-12V'"
        )


def test_foreign_key_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO inventory VALUES ('NO-SUCH-PART', 'WH-TPE', 1, 0, ?)", (SEED_TIME,)
        )
