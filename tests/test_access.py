"""Access levels and document revisions in search. No network, no cost."""

from pathlib import Path

import pytest
from mcp.client.client import Client
from test_agent import FakeClient, text, tool_use
from test_fallback import FakeOllama, ol_text, ol_tool
from test_retrieve import FakeModel

from part_assistant import access
from part_assistant.agent import Agent
from part_assistant.chunk import (
    INTERNAL_DIR,
    Chunk,
    check_revisions,
    chunk_document,
    load_markdown,
)
from part_assistant.db import connect, init_schema
from part_assistant.local_agent import LocalAgent
from part_assistant.mcp_server import build_server
from part_assistant.retrieve import Embedder, Retriever
from part_assistant.seed import seed
from part_assistant.tools import TOOL_SCHEMAS, Tools

Q = "TEC3200 burn-in hours"


def doc(cid, file, level="public", doc_id="", revision="", effective="", words=Q):
    return Chunk(chunk_id=cid, file=file, vendor="V", model="TEC3200-54-074xA", kind="text",
                 page=1, page_end=1, section="", text=f"[ctx]\nTEC3200 {words}",
                 access_level=level, doc_id=doc_id, revision=revision, effective=effective)


CHUNKS = [
    doc("ds", "bel-tec3200-54.pdf", words="TEC3200 efficiency"),  # public datasheet
    doc("wa", "wi-revA.md", "internal", "WI-1", "A", "2020-03-01"),
    doc("wb", "wi-revB.md", "internal", "WI-1", "B", "2020-10-01"),
    doc("wc", "wi-revC.md", "internal", "WI-1", "C", "2999-01-01"),  # not in force yet
    doc("sq", "sqe.md", "confidential", "SQE-1", "1", "2020-09-15"),
]


@pytest.fixture
def tools(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_schema(conn)
    seed(conn)  # alice = internal, bob = confidential
    emb = Embedder("fake", cache_dir=tmp_path, model=FakeModel())
    return Tools(conn, Retriever(CHUNKS, emb))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def files(out: dict) -> set[str]:
    return {r["source"].split(" ")[0] for r in out["results"]}


# --- the rules

@pytest.mark.parametrize("clearance,level,ok", [
    ("public", "public", True), ("public", "internal", False),
    ("internal", "internal", True), ("internal", "confidential", False),
    ("confidential", "public", True), ("confidential", "confidential", True),
    ("confidential", "secret", False),  # typo in a document header: nobody reads it
    ("admin", "public", False),  # unknown clearance: nothing
])
def test_can_read(clearance, level, ok):
    assert access.can_read(clearance, level) is ok


def test_unknown_user_is_public(tools):
    assert access.clearance_of(tools.conn, "mallory") == "public"
    assert access.clearance_of(tools.conn, None) == "public"
    assert access.clearance_of(tools.conn, "bob") == "confidential"


@pytest.mark.parametrize("on,rev", [("2020-06-01", "A"), ("2021-01-01", "B"),
                                    ("2999-01-01", "C")])
def test_the_revision_in_force_on_a_date(on, rev):
    current = access.current_revisions(CHUNKS, on)
    assert ("WI-1", rev) in current and len({r for d, r in current if d == "WI-1"}) == 1
    assert ("bel-tec3200-54.pdf", "") in current  # no date: always in force


def test_before_the_first_revision_the_document_does_not_exist():
    assert not any(d == "WI-1" for d, _ in access.current_revisions(CHUNKS, "2019-01-01"))


# --- search_docs

@pytest.mark.parametrize("user,expected", [
    ("guest", {"bel-tec3200-54.pdf"}),
    ("alice", {"bel-tec3200-54.pdf", "wi-revB.md"}),
    ("bob", {"bel-tec3200-54.pdf", "wi-revB.md", "sqe.md"}),
])
def test_search_returns_only_what_the_user_may_read(tools, user, expected):
    assert files(tools.search_docs(Q, user)) == expected  # never rev A or the future rev C


def test_no_user_means_public(tools):
    assert files(tools.search_docs(Q)) == {"bel-tec3200-54.pdf"}


def test_internal_results_say_revision_and_level_public_ones_unchanged(tools):
    out = {r["source"].split(" ")[0]: r for r in tools.search_docs(Q, "bob")["results"]}
    assert (out["wi-revB.md"]["revision"], out["wi-revB.md"]["effective"]) == ("B", "2020-10-01")
    assert out["sqe.md"]["access_level"] == "confidential"
    assert set(out["bel-tec3200-54.pdf"]) == {"source", "model", "section", "text"}


def test_the_model_can_not_choose_the_user():
    schema = next(t for t in TOOL_SCHEMAS if t["name"] == "search_docs")["input_schema"]
    assert set(schema["properties"]) == {"query"}


# --- every caller passes the user

def test_agent_searches_as_the_asking_user(tools):
    def ask(user):
        client = FakeClient(tool_use(("search_docs", {"query": Q})), text("ok"))
        r = Agent(tools, client).run(Q, user_id=user)
        return files(r.tool_calls[0]["output"])

    assert "sqe.md" in ask("bob")
    assert "sqe.md" not in ask("alice")


def test_local_fallback_searches_as_the_same_user(tools):
    local = LocalAgent(tools, chat=FakeOllama(ol_tool("search_docs", {"query": Q}),
                                              ol_text("ok")))
    r = local.run(Q, user_id="alice")
    assert files(r.tool_calls[0]["output"]) == {"bel-tec3200-54.pdf", "wi-revB.md"}


@pytest.mark.anyio
async def test_mcp_searches_as_its_user(tools):
    async with Client(build_server(tools, user_id="alice")) as client:
        out = await client.call_tool("search_docs", {"query": Q})
    assert "wi-revB.md" in str(out) and "sqe.md" not in str(out)


# --- internal documents on disk

def test_the_internal_documents_load_and_carry_their_header():
    metas = [load_markdown(p)[0] for p in sorted(INTERNAL_DIR.glob("*.md"))]
    assert {m["access_level"] for m in metas} == {"internal", "confidential"}
    check_revisions(metas)  # no two revisions take effect on the same day
    rev_b = next(m for m in metas if m["file"] == "wi-iqc-0032-revB.md")
    chunks = chunk_document(load_markdown(INTERNAL_DIR / rev_b["file"])[1], rev_b)
    assert all(c.doc_id == "WI-IQC-0032" and c.access_level == "internal" for c in chunks)
    assert "wi-iqc-0032-revB.md rev B" in chunks[0].text  # context line names the revision
    assert any("4 小時" in c.text for c in chunks)


def write(tmp_path: Path, header: str) -> Path:
    p = tmp_path / "d.md"
    p.write_text(f"+++\n{header}\n+++\n# T\nbody\n", encoding="utf-8")
    return p


GOOD = ('doc_id = "D"\ntitle = "t"\nrevision = "A"\neffective = "2026-01-01"\n'
        'access_level = "internal"\nvendor = "v"\nmodel = "m"')


@pytest.mark.parametrize("header,error", [
    (GOOD.replace('"internal"', '"intrenal"'), "access_level"),  # typo must not go public
    (GOOD.replace('access_level = "internal"\n', ""), "missing"),
    (GOOD.replace('"2026-01-01"', '"next week"'), "effective"),
])
def test_a_bad_header_stops_the_build(tmp_path, header, error):
    with pytest.raises(ValueError, match=error):
        load_markdown(write(tmp_path, header))


def test_no_header_stops_the_build(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# just markdown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_markdown(p)


def test_two_revisions_on_the_same_day_are_refused():
    same = [{"file": "a.md", "doc_id": "D", "effective": "2026-01-01"},
            {"file": "b.md", "doc_id": "D", "effective": "2026-01-01"}]
    with pytest.raises(ValueError, match="same doc_id"):
        check_revisions(same)
