"""Generate deterministic fake factory data (parts, inventory, tickets).

Run:  uv run python -m part_assistant.seed
All values are synthetic. No real company data is used.
"""

import random
import sqlite3
from pathlib import Path

from part_assistant.db import DEFAULT_DB_PATH, connect, init_schema

# Fixed timestamp so every run produces byte-identical data (reproducible evals).
SEED_TIME = "2026-11-01T09:00:00+08:00"
WAREHOUSES = ["WH-TPE", "WH-TYN", "WH-SZX"]

# (part_no, name, category, spec, lifecycle)
# Part numbers are deliberately similar (e.g. PSU-1600-12V vs PSU-1600-54V)
# so retrieval must match them exactly, not just "roughly".
PARTS = [
    ("PSU-0800-12V", "800W server PSU, 12V output", "psu", "800W, 12V main, 80 PLUS Platinum", "eol"),
    ("PSU-1200-12V", "1200W server PSU, 12V output", "psu", "1200W, 12V main, 80 PLUS Platinum", "nrnd"),
    ("PSU-1600-12V", "1600W server PSU, 12V output", "psu", "1600W, 12V main, 80 PLUS Titanium", "active"),
    ("PSU-1600-54V", "1600W server PSU, 54V output", "psu", "1600W, 54V main, 80 PLUS Titanium", "active"),
    ("PSU-2000-54V", "2000W server PSU, 54V output", "psu", "2000W, 54V main, 80 PLUS Titanium", "active"),
    ("PSU-2700-54V", "2700W server PSU, 54V output", "psu", "2700W, 54V main, 80 PLUS Titanium", "active"),
    ("PSU-3200-54V", "3200W server PSU, 54V output", "psu", "3200W, 54V main, 80 PLUS Titanium", "active"),
    ("FAN-4028-HS", "40x28mm high-speed fan", "fan", "40x28mm, 12V, 25000 RPM", "active"),
    ("FAN-4056-HS", "40x56mm high-speed fan", "fan", "40x56mm, 12V, 23000 RPM", "active"),
    ("FAN-6038-STD", "60x38mm standard fan", "fan", "60x38mm, 12V, 12000 RPM", "nrnd"),
    ("CON-ATX-24P", "24-pin ATX power connector", "connector", "24-pin, 2 rows, 4.2mm pitch", "active"),
    ("CON-PCIE-8P", "8-pin PCIe power connector", "connector", "8-pin, 2 rows, 4.2mm pitch", "active"),
    ("CON-12VHPWR", "16-pin 12VHPWR connector", "connector", "16-pin (12+4), 3.0mm pitch", "active"),
    ("CON-BUSBAR-2P", "2-pole busbar connector", "connector", "2-pole, 100A per pole", "active"),
    ("CAP-470UF-35V", "470uF 35V electrolytic capacitor", "capacitor", "470uF, 35V, 105C", "active"),
    ("CAP-1000UF-16V", "1000uF 16V electrolytic capacitor", "capacitor", "1000uF, 16V, 105C", "active"),
    ("CAP-1000UF-63V", "1000uF 63V electrolytic capacitor", "capacitor", "1000uF, 63V, 105C", "nrnd"),
    ("CAP-22UF-450V", "22uF 450V bulk capacitor", "capacitor", "22uF, 450V, 105C", "active"),
]

# Internal part -> public datasheet model, only where the specs really match
# (power, output voltage, 80 PLUS grade). Others stay NULL on purpose: the agent
# must say "no datasheet on file" instead of guessing.
VENDOR_MPN = {
    "PSU-0800-12V": "LMS800-P12B",  # Mornsun, 800 W 12 V Platinum
    "PSU-1600-12V": "TEC1600-12-074NA",  # Bel, 1600 W 12 V Titanium
    "PSU-3200-54V": "TEC3200-54-074NA",  # Bel, 3200 W 54 V Titanium
}

# A few parts are intentionally out of stock everywhere -> good eval cases.
OUT_OF_STOCK = {"PSU-0800-12V", "CAP-1000UF-63V"}

TICKETS = [
    # (ticket_id, idempotency_key, part_no, title, description, severity, status)
    ("TK-0001", "seed-0001", "FAN-6038-STD", "Fan noise on line 3",
     "Abnormal bearing noise found during burn-in.", "medium", "open"),
    ("TK-0002", "seed-0002", "PSU-1200-12V", "Replace NRND part",
     "Plan migration from PSU-1200-12V to PSU-1600-12V.", "low", "closed"),
]


def seed(conn: sqlite3.Connection, rng_seed: int = 42) -> None:
    """Fill an empty database. Same rng_seed -> same data, every time."""
    rng = random.Random(rng_seed)  # local RNG: does not touch global random state
    init_schema(conn)
    with conn:  # one transaction: all rows commit together, or none do
        conn.executemany(
            "INSERT INTO parts (part_no, name, category, spec, vendor_mpn, lifecycle)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(pn, name, cat, spec, VENDOR_MPN.get(pn), lc) for pn, name, cat, spec, lc in PARTS],
        )
        for part_no, *_ in PARTS:
            for wh in WAREHOUSES:
                on_hand = 0 if part_no in OUT_OF_STOCK else rng.randint(0, 500)
                reserved = rng.randint(0, on_hand)
                conn.execute(
                    "INSERT INTO inventory (part_no, warehouse, qty_on_hand, qty_reserved, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (part_no, wh, on_hand, reserved, SEED_TIME),
                )
        conn.executemany(
            "INSERT INTO tickets (ticket_id, idempotency_key, part_no, title, description,"
            " severity, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(*t, SEED_TIME) for t in TICKETS],
        )


def main(path: Path = DEFAULT_DB_PATH) -> None:
    path.unlink(missing_ok=True)  # rebuild from scratch every time
    conn = connect(path)
    seed(conn)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("parts", "inventory", "tickets")}
    conn.close()
    print(f"Seeded {path}: {counts}")


if __name__ == "__main__":
    main()
