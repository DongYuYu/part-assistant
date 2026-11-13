"""Parsed blocks -> retrieval chunks (data/processed/chunks.jsonl).

Rules:
  - Text: headings start a new section; paragraphs are packed up to ~TEXT_TOKENS,
    and the next chunk repeats the last ~15% (overlap) so a fact on the boundary
    is never cut in half. Overlap never crosses a heading.
  - Tables: never cut mid-row. Big tables are split by rows and every piece repeats
    the header. Footnote text found right after the table is copied into it.
  - Every chunk starts with a context line "[vendor | model | file | pages | section]",
    so a chunk still makes sense alone (and BM25 can match the model name).
  - chunk_id is a hash of the content: same input -> same id, so the embedding stage
    can skip chunks it has already embedded.
"""

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from part_assistant.download import load_sources, sha256_of
from part_assistant.parse import Block, Table, parse_pdf_cached

TEXT_TOKENS = 400
OVERLAP_TOKENS = 60  # ~15% of TEXT_TOKENS
TABLE_TOKENS = 600  # tables may be bigger than text: splitting them loses context
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")


@dataclass
class Chunk:
    chunk_id: str
    file: str
    vendor: str
    model: str
    kind: str  # "text" or "table"
    page: int
    page_end: int
    section: str
    text: str  # what gets indexed and shown to the model: context line + body
    labels_inferred: bool = False  # table parameter names were guessed
    # Reserved now so the production-gap work does not force a full re-index later.
    doc_sha256: str = ""  # which exact PDF this came from
    revision: str = ""  # datasheet revision, when known
    access_level: str = "public"


def n_tokens(s: str) -> int:
    """Rough count: English is ~4 characters per token. The exact number depends on
    the embedding model (chosen later), so the budgets above are soft limits."""
    return len(s) // 4 + 1


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


def heading_text(par: str) -> str | None:
    m = _HEADING_RE.match(par.strip())
    if not m:
        return None
    return re.sub(r"\*+", "", m.group(1)).strip() or None


def paragraphs(text: str) -> list[str]:
    text = re.sub(r"<!-- (Start|End) of picture text -->", "", text)  # keep the words
    pars = (p.replace("**", "").strip() for p in re.split(r"\n\s*\n", text))
    return [p for p in pars if p]


def split_long(par: str, budget: int = TEXT_TOKENS) -> list[str]:
    """Split a paragraph that is over budget at sentence ends; cut by characters
    only when a single sentence is still too long."""
    if n_tokens(par) <= budget:
        return [par]
    out: list[str] = []
    cur = ""
    for sent in re.split(r"(?<=[.!?;])\s+", par):
        while n_tokens(sent) > budget:
            cut = budget * 4
            out.append(sent[:cut])
            sent = sent[cut:]
        if cur and n_tokens(cur + " " + sent) > budget:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def overlap_tail(units: list[tuple[int, str]], budget: int = OVERLAP_TOKENS):
    """The last units that fit in the overlap budget (may be empty)."""
    tail: list[tuple[int, str]] = []
    used = 0
    for unit in reversed(units):
        used += n_tokens(unit[1])
        if used > budget:
            break
        tail.insert(0, unit)
    return tail


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def dedup_row(row: list[str]) -> list[str]:
    """colspan was expanded into copies ("Input", "Input", ...). Keep one."""
    out: list[str] = []
    for cell in row:
        if not out or cell != out[-1]:
            out.append(cell)
    return out


def render_row(row: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in dedup_row(row)) + " |"


def table_bodies(table: Table, budget: int = TABLE_TOKENS) -> list[str]:
    lines = [render_row(r) for r in table.rows]
    head: list[str] = []
    if table.has_header:
        n_cols = len(dedup_row(table.rows[0]))
        head = [lines[0], "|" + " --- |" * n_cols]
        lines = lines[1:]
    pieces: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if cur and n_tokens("\n".join(head + cur + [ln])) > budget:
            pieces.append(cur)
            cur = []
        cur.append(ln)
    if cur or not pieces:
        pieces.append(cur)
    return ["\n".join(head + p) for p in pieces]


_MARK_RE = re.compile(r"\[\^(\d+)\]")
# "1. text", "- 2 text" (list item), "> 3 text" (quote): the prefix depends on the layout.
_NOTE_RE = re.compile(r"^\s*(?:[-*>]\s*)?(\d{1,2})[.)]?\s+(\S.*)$")


def find_footnotes(table: Table, following_texts: list[str]) -> list[str]:
    """Footnote text sits in the text block after the table ("1. Minimum loading...").
    Copy the lines whose number matches a [^n] marker used in this table."""
    marks = {m for row in table.rows for cell in row for m in _MARK_RE.findall(cell)}
    if not marks:
        return []
    notes: dict[str, str] = {}
    for text in following_texts:
        for line in text.splitlines():
            m = _NOTE_RE.match(line)
            if m and m.group(1) in marks and m.group(1) not in notes:
                notes[m.group(1)] = m.group(2).strip()
    return [f"[^{k}]: {notes[k]}" for k in sorted(notes, key=int)]


# ---------------------------------------------------------------------------
# Document -> chunks
# ---------------------------------------------------------------------------

def make_chunk(meta: dict, kind: str, page: int, page_end: int, section: str, body: str,
               labels_inferred: bool = False) -> Chunk:
    pages = f"p.{page}" if page == page_end else f"p.{page}-{page_end}"
    context = f"[{meta['vendor']} | {meta['model']} | {meta['file']} | {pages} | {section or '-'}]"
    text = f"{context}\n{body}"
    chunk_id = hashlib.sha256(f"{meta['file']}\n{text}".encode()).hexdigest()[:16]
    return Chunk(
        chunk_id=chunk_id, file=meta["file"], vendor=meta["vendor"], model=meta["model"],
        kind=kind, page=page, page_end=page_end, section=section, text=text,
        labels_inferred=labels_inferred, doc_sha256=meta.get("doc_sha256", ""),
        revision=meta.get("revision", ""), access_level=meta.get("access_level", "public"),
    )


def chunk_document(blocks: list[Block], meta: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    buf: list[tuple[int, str]] = []  # (page, paragraph) waiting to become a text chunk

    def flush(keep_overlap: bool) -> None:
        nonlocal buf
        if buf:
            pages = [p for p, _ in buf]
            body = "\n\n".join(u for _, u in buf)
            chunks.append(make_chunk(meta, "text", min(pages), max(pages), section, body))
        buf = overlap_tail(buf) if keep_overlap else []

    for i, b in enumerate(blocks):
        if b.kind == "table":
            t = b.table
            after = [x.text for x in blocks[i + 1:i + 4]
                     if x.kind == "text" and x.page <= t.page_end + 1]
            notes = find_footnotes(t, after)
            for body in table_bodies(t):
                if notes:
                    body += "\n\n" + "\n".join(notes)
                chunks.append(make_chunk(meta, "table", t.page, t.page_end, section, body,
                                         t.labels_inferred))
            continue
        for par in paragraphs(b.text):
            h = heading_text(par)
            if h is not None:
                flush(keep_overlap=False)  # overlap must not leak into another section
                section = h
                continue
            for unit in split_long(par):
                if buf and n_tokens("\n\n".join([u for _, u in buf] + [unit])) > TEXT_TOKENS:
                    flush(keep_overlap=True)
                buf.append((b.page, unit))
    flush(keep_overlap=False)

    seen: set[str] = set()  # identical text on the same page -> keep one copy
    return [c for c in chunks if not (c.chunk_id in seen or seen.add(c.chunk_id))]


def build(raw_dir: Path = RAW_DIR, out: Path = OUT_PATH) -> list[Chunk]:
    chunks: list[Chunk] = []
    for src in load_sources():
        path = raw_dir / src["file"]
        if not path.exists():
            print(f"missing {path} (run: python -m part_assistant.download)", file=sys.stderr)
            continue
        meta = {**src, "doc_sha256": sha256_of(path)}
        doc_chunks = chunk_document(parse_pdf_cached(path), meta)
        n_tables = sum(c.kind == "table" for c in doc_chunks)
        print(f"{src['file']:28} {len(doc_chunks):4} chunks ({n_tables} table)")
        chunks += doc_chunks
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with tmp.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"wrote {len(chunks)} chunks -> {out}")
    return chunks


def load_chunks(path: Path = OUT_PATH) -> list[Chunk]:
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f]


if __name__ == "__main__":
    build()
