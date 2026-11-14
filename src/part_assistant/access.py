"""Who may read which document, and which revision of a document is current.

Two rules, applied to every search BEFORE ranking (Retriever.search `where`), so a
chunk the user may not read never even becomes a candidate:

  1. access   a chunk is readable if its level is at or below the user's clearance:
              public < internal < confidential. A user not in the `users` table is
              public. An unknown level (a typo in a document header) is readable by
              nobody: fail closed.
  2. version  documents with the same doc_id are revisions of one document. Only the
              current one is searched: the latest revision whose effective date has
              arrived. A revision dated in the future waits for its date.

There is no login in this project: the user id is what the caller says it is
(see audit.current_user), so this filter is only as strong as that id.
"""

import sqlite3
from collections.abc import Callable, Iterable

from part_assistant.chunk import Chunk

LEVELS = ("public", "internal", "confidential")  # ordered: each one sees the ones before it
DEFAULT_CLEARANCE = "public"


def can_read(clearance: str, level: str) -> bool:
    if clearance not in LEVELS or level not in LEVELS:
        return False
    return LEVELS.index(level) <= LEVELS.index(clearance)


def clearance_of(conn: sqlite3.Connection, user_id: str | None) -> str:
    if not user_id:
        return DEFAULT_CLEARANCE
    row = conn.execute("SELECT clearance FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return row["clearance"] if row else DEFAULT_CLEARANCE


def doc_key(c: Chunk) -> str:
    return c.doc_id or c.file  # chunks made before doc_id existed: one file = one document


def current_revisions(chunks: Iterable[Chunk], on: str) -> set[tuple[str, str]]:
    """{(doc_id, revision)} of the revision in force on date `on` (YYYY-MM-DD).
    A chunk with no effective date is always in force (the public datasheets)."""
    best: dict[str, tuple[str, str]] = {}  # doc_id -> (effective, revision)
    for c in chunks:
        if c.effective > on:
            continue  # not in force yet
        key = doc_key(c)
        if key not in best or c.effective > best[key][0]:
            best[key] = (c.effective, c.revision)
    return {(key, rev) for key, (_, rev) in best.items()}


def visible(chunks: list[Chunk], clearance: str, on: str) -> Callable[[Chunk], bool]:
    """The `where` for Retriever.search: readable by this clearance AND current on `on`."""
    current = current_revisions(chunks, on)
    return lambda c: can_read(clearance, c.access_level) and (doc_key(c), c.revision) in current
