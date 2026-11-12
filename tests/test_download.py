"""Download logic tests. No network: file:// URLs stand in for vendor sites."""

import hashlib

import pytest

from part_assistant.download import fetch, load_sources, sync

FAKE_PDF = b"%PDF-1.7\nfake datasheet body\n"


@pytest.fixture
def remote(tmp_path):
    """A local 'remote' PDF plus its manifest entry."""
    src = tmp_path / "remote" / "ds.pdf"
    src.parent.mkdir()
    src.write_bytes(FAKE_PDF)
    entry = {"file": "ds.pdf", "url": src.as_uri(),
             "sha256": hashlib.sha256(FAKE_PDF).hexdigest()}
    return src, entry


def test_download_and_verify(tmp_path, remote):
    _, entry = remote
    raw = tmp_path / "raw"
    assert sync([entry], raw) is True
    assert (raw / "ds.pdf").read_bytes() == FAKE_PDF


def test_hash_mismatch_fails(tmp_path, remote, capsys):
    """Vendor silently updated the file -> we must notice."""
    src, entry = remote
    src.write_bytes(b"%PDF-1.7\nrevised by vendor\n")
    assert sync([entry], tmp_path / "raw") is False
    assert "MISMATCH" in capsys.readouterr().out


def test_unpinned_hash_is_not_success(tmp_path, remote):
    _, entry = remote
    assert sync([{**entry, "sha256": ""}], tmp_path / "raw") is False


def test_html_instead_of_pdf_rejected(tmp_path):
    """Sites that block scripts often return an HTML page with status 200."""
    page = tmp_path / "blocked.pdf"
    page.write_bytes(b"<!DOCTYPE html><html>Access denied</html>")
    dest = tmp_path / "out.pdf"
    with pytest.raises(ValueError, match="not a PDF"):
        fetch(page.as_uri(), dest)
    assert not dest.exists()  # nothing half-written left behind
    assert not dest.with_suffix(".pdf.part").exists()


def test_manifest_is_fully_pinned():
    """Guard: every committed manifest entry must have a 64-hex-char sha256."""
    for src in load_sources():
        assert len(src["sha256"]) == 64, src["file"]
