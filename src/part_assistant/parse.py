"""PDF -> list of text/table blocks with page numbers, plus datasheet-specific cleanup.

Pipeline (each step is a small function, tested in tests/test_parse.py):
  1. pymupdf4llm renders each page to markdown, tables as HTML.
  2. HTML tables are parsed into a rectangular grid; rowspan/colspan are expanded.
  3. Footnote markers (small raised digits) become [^n] so "0 A 1" is not read as a value.
  4. Borderless tables: empty parameter-name cells are filled using row shading/lines.
  5. Tables continued on the next page are merged.

parse_pdf_cached() stores the result keyed by (PDF sha256, PARSER_VERSION).
"""

import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

import pymupdf
import pymupdf4llm

from part_assistant.download import sha256_of

# Bump this whenever a code change can change parse output. It is part of the cache
# key: the PDF did not change, but the code did, so old cached results are stale.
PARSER_VERSION = "1"
CACHE_DIR = Path("data/processed/parsed")


@dataclass
class Table:
    rows: list[list[str]]
    has_header: bool  # rows[0] is a header row
    page: int  # 1-based page where the table starts
    page_end: int
    labels_inferred: bool = False  # parameter names were guessed (no visual evidence)


@dataclass
class Block:
    kind: str  # "text" or "table"
    page: int
    text: str = ""
    table: Table | None = None


# ---------------------------------------------------------------------------
# Step 1: render pages
# ---------------------------------------------------------------------------

def render_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, markdown_with_html_tables), ...]."""
    pages = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,  # one dict per page, so we keep page numbers for citations
        header=False,  # drop repeated page headers (layout model decides)
        footer=False,  # drop repeated footers (contact info, copyright)
        use_ocr=False,  # every page has a text layer; OCR would only add noise
        table_output="html",  # HTML keeps rowspan/colspan; markdown tables cannot
        show_progress=False,
    )
    return [(p["metadata"]["page_number"], p["text"]) for p in pages]


# ---------------------------------------------------------------------------
# Step 2: HTML table -> grid
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Parse one <table> into rows of (text, rowspan, colspan, is_header) cells."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._is_th = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []
            self._span = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
            self._is_th = tag == "th"
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")  # in-cell line wrap, not a real break
        elif tag == "sup" and self._cell is not None:
            self._cell.append("^")  # I<sup>2</sup>C -> I^2C, 2<sup>N</sup> -> 2^N

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = clean_text("".join(self._cell))
            self.rows[-1].append((text, *self._span, self._is_th))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("I^2 C", "I2C").replace("I^2C", "I2C")  # domain spelling people search for
    return s


def parse_html_table(html: str) -> tuple[list[list[str]], bool]:
    """Expand rowspan/colspan so every row has the same number of columns.

    Rule: a merged cell's value applies to every grid position it covers.
    That is what fills parameter names in bordered tables, e.g. a 'Model Name'
    cell spanning 12 rows is copied into all 12 rows.
    """
    p = _TableParser()
    p.feed(html)
    grid: list[list[str]] = []
    carry: dict[int, tuple[int, str]] = {}  # col -> (rows still to fill, text) from rowspan above
    header_flags = []
    for raw in p.rows:
        row: list[str] = []
        cells = iter(raw)
        col = 0
        all_th = bool(raw)
        while True:
            if col in carry:  # position occupied by a rowspan from an earlier row
                left, text = carry[col]
                row.append(text)
                carry[col] = (left - 1, text) if left > 1 else None
                if carry[col] is None:
                    del carry[col]
                col += 1
                continue
            cell = next(cells, None)
            if cell is None:
                if any(c >= col for c in carry):  # trailing rowspans still to place
                    row.append("")
                    col += 1
                    continue
                break
            text, rs, cs, is_th = cell
            all_th &= is_th
            for k in range(cs):
                row.append(text)
                if rs > 1:
                    carry[col + k] = (rs - 1, text)
            col += cs
        grid.append(row)
        header_flags.append(all_th)
    width = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (width - len(r)) for r in grid]
    return grid, bool(header_flags) and header_flags[0]


# ---------------------------------------------------------------------------
# Step 3: footnote markers
# ---------------------------------------------------------------------------

def footnote_markers(page: pymupdf.Page) -> list[tuple[str, str]]:
    """Find (text_before, digit) pairs where digit is a small raised footnote mark.

    A mark is digits-only, under 75% of its left neighbour's font size, starts
    right where the neighbour ends, and its baseline sits higher. We compare by
    position, not by "same line": PDFs often store the mark as its own line.
    """
    spans = [s for b in page.get_text("dict")["blocks"] for line in b.get("lines", [])
             for s in line["spans"] if s["text"].strip()]
    found = []
    for s in spans:
        mark = s["text"].strip()
        if not mark.isdigit():
            continue
        for n in spans:
            gap = s["bbox"][0] - n["bbox"][2]  # mark.x0 - neighbour.x1
            raised = n["origin"][1] - n["size"] < s["origin"][1] < n["origin"][1] - 0.5
            if n is not s and 0 <= gap < 3 and s["size"] < 0.75 * n["size"] and raised:
                found.append((clean_text(n["text"]), mark))
                break
    return found


def apply_footnote_markers(text: str, markers: list[tuple[str, str]]) -> str:
    for before, mark in markers:
        text = re.sub(rf"({re.escape(before)}) {mark}\b", rf"\1[^{mark}]", text)
    return text


# ---------------------------------------------------------------------------
# Step 4: parameter names in borderless tables
# ---------------------------------------------------------------------------

def _fill_color_at(fills, x: float, y: float):
    hits = [color for rect, color in fills if rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1]
    return hits[-1] if hits else None  # last drawn shape is on top


def fill_parameter_names(rows: list[list[str]], start: int, page: pymupdf.Page) -> bool:
    """Fill empty first-column cells using visual group boundaries.

    The parameter name of a merged cell is vertically centered, so it lands on a
    middle row; simple forward-fill would give the first rows the PREVIOUS name.
    We find each row's y on the page, then cut groups wherever the background
    color of the name column changes or a horizontal line crosses it.
    Returns True if we had to fall back to forward-fill (no visual evidence).
    """
    body = range(start, len(rows))
    if not any(rows[i][0] == "" for i in body) or len(rows[0]) < 2:
        return False

    # Locate each row on the page, top to bottom (search hits must keep increasing).
    # Probe with a short prefix: a cell wrapped onto two PDF lines is not found whole.
    ys: dict[int, float] = {}
    last_y = -1.0
    for i in body:
        probe = next((c[:20] for c in rows[i][1:] if len(c) >= 2), "")
        hits = [r for r in page.search_for(probe) if r.y0 > last_y] if probe else []
        if hits:
            ys[i] = (hits[0].y0 + hits[0].y1) / 2
            last_y = hits[0].y0 + 0.5
    names = [page.search_for(rows[i][0][:12]) for i in body if rows[i][0]]
    name_hits = [h[0] for h in names if h]

    if len(ys) < len(body) or not name_hits:
        _forward_fill(rows, start)
        return True

    x = min(h.x0 for h in name_hits) + 2  # a point inside the name column
    drawings = page.get_drawings()
    fills = [(d["rect"], d["fill"]) for d in drawings if d.get("fill")]
    hlines = [d["rect"] for d in drawings if d["rect"].height < 1.5 and d["rect"].width > 20]

    # Cut groups ONLY at visual boundaries. Two adjacent rows that both have a
    # name can still be one group: a long name wrapped onto two lines
    # ("iTHD" / "(Total Harmonic Distortion)").
    groups: list[list[int]] = [[start]]
    for prev_i, i in zip(body, body[1:], strict=False):
        y0, y1 = ys[prev_i], ys[i]
        color_changed = _fill_color_at(fills, x, y0) != _fill_color_at(fills, x, y1)
        line_between = any(y0 < r.y0 < y1 and r.x0 <= x <= r.x1 for r in hlines)
        if color_changed or line_between:
            groups.append([])
        groups[-1].append(i)

    if len(groups) == 1 and len(body) > 1:  # no visual evidence at all
        _forward_fill(rows, start)
        return True

    for g in groups:
        if all(rows[i][0] for i in g):
            continue  # nothing to fill; keep distinct names as they are
        name = " ".join(rows[i][0] for i in g if rows[i][0])  # wrapped names are joined
        for i in g:
            rows[i][0] = name
    return False


def _forward_fill(rows: list[list[str]], start: int) -> None:
    for i in range(start + 1, len(rows)):
        if not rows[i][0]:
            rows[i][0] = rows[i - 1][0]


# ---------------------------------------------------------------------------
# Steps 1-5 together
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"(<table>.*?</table>)", re.S)


def parse_pdf(path: str | Path) -> list[Block]:
    doc = pymupdf.open(str(path))
    # Pass 1: parse each page on its own (geometry-based fixes need that page).
    per_page = [_parse_page(doc[n - 1], n, md) for n, md in render_pages(path)]
    # Pass 2: drop lines repeated on most pages (running titles the layout model missed).
    repeated = _repeated_lines(per_page)
    # Pass 3: join tables continued across pages.
    blocks: list[Block] = []
    for page_blocks in per_page:
        page_blocks = _drop_lines(page_blocks, repeated)
        _merge_continued_table(blocks, page_blocks)
        blocks.extend(page_blocks)
    return blocks


def _parse_page(page: pymupdf.Page, page_no: int, md: str) -> list[Block]:
    markers = footnote_markers(page)
    blocks: list[Block] = []
    for part in _TABLE_RE.split(md):
        if not part.strip():
            continue
        if part.startswith("<table>"):
            rows, has_header = parse_html_table(part)
            if not rows:
                continue
            # Fill names BEFORE marking footnotes: filling searches the page for the
            # cell text, and "0 A[^1]" does not appear on the page.
            inferred = fill_parameter_names(rows, 1 if has_header else 0, page)
            rows = [[apply_footnote_markers(c, markers) for c in r] for r in rows]
            table = Table(rows, has_header, page_no, page_no, inferred)
            blocks.append(Block("table", page_no, table=table))
        else:
            text = apply_footnote_markers(clean_markdown(part), markers)
            if text:
                blocks.append(Block("text", page_no, text=text))
    return blocks


def _repeated_lines(per_page: list[list[Block]]) -> set[str]:
    """Short lines found on >= 25% of pages (min 3) are running headers/footers.

    25%, not 50%: the layout model already removes the title on many pages, so
    the survivors are undercounted (TEC1600: title left on 21 of 56 pages).
    The length cap keeps real sentences that happen to repeat.
    """
    counts: dict[str, int] = {}
    for page_blocks in per_page:
        seen = {ln.strip() for b in page_blocks if b.kind == "text" for ln in b.text.splitlines()}
        for ln in seen - {""}:
            if len(ln) <= 80:
                counts[ln] = counts.get(ln, 0) + 1
    threshold = max(3, len(per_page) * 0.25)
    return {ln for ln, c in counts.items() if c >= threshold}


def _drop_lines(page_blocks: list[Block], repeated: set[str]) -> list[Block]:
    out = []
    for b in page_blocks:
        if b.kind == "text":
            kept = [ln for ln in b.text.splitlines() if ln.strip() not in repeated]
            b.text = "\n".join(kept).strip()
            if not b.text:
                continue
        out.append(b)
    return out


def clean_markdown(s: str) -> str:
    s = s.replace("<br>", " ").replace("<br/>", " ")
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
    s = re.sub(r"</?(mark|u)>", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.replace("I^2 C", "I2C").replace("I^2C", "I2C").strip()


# ---------------------------------------------------------------------------
# Step 5: tables split across pages
# ---------------------------------------------------------------------------

def _merge_continued_table(done: list[Block], page_blocks: list[Block]) -> None:
    """If a page starts with a table and the previous page ended with one of the
    same width, it is the same table: append rows and reuse the earlier header."""
    if not done or not page_blocks:
        return
    prev, first = done[-1], page_blocks[0]
    if prev.kind != "table" or first.kind != "table":
        return
    a, b = prev.table, first.table
    if len(a.rows[0]) != len(b.rows[0]):
        return
    new_rows = b.rows
    if b.has_header and a.has_header and b.rows[0] == a.rows[0]:
        new_rows = b.rows[1:]  # header repeated on the new page: drop the duplicate
    a.rows.extend(new_rows)  # otherwise the "header" row was really a data row
    a.page_end = b.page_end
    a.labels_inferred |= b.labels_inferred
    page_blocks.pop(0)


# ---------------------------------------------------------------------------
# Cache: parsing all 9 PDFs takes ~2 minutes, chunking takes < 1 second
# ---------------------------------------------------------------------------

def parse_pdf_cached(path: str | Path, cache_dir: Path = CACHE_DIR) -> list[Block]:
    path = Path(path)
    key = f"{sha256_of(path)[:16]}.v{PARSER_VERSION}"
    cache = cache_dir / f"{path.stem}.{key}.json"
    if cache.exists():
        return [_block_from_dict(d) for d in json.loads(cache.read_text())]
    blocks = parse_pdf(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{path.stem}.*.json"):  # stale versions of this PDF
        old.unlink()
    tmp = cache.with_suffix(".part")
    tmp.write_text(json.dumps([asdict(b) for b in blocks], ensure_ascii=False))
    tmp.replace(cache)  # atomic: a crash never leaves a half-written cache file
    return blocks


def _block_from_dict(d: dict) -> Block:
    table = Table(**d["table"]) if d["table"] else None
    return Block(d["kind"], d["page"], d["text"], table)
