"""Download the public datasheets listed in data/sources.toml and verify SHA-256.

Run:  uv run python -m part_assistant.download
Exit code is non-zero if any file fails to download or its hash does not match.
"""

import hashlib
import sys
import tomllib
import urllib.request
from pathlib import Path

MANIFEST = Path("data/sources.toml")
RAW_DIR = Path("data/raw")
# Some vendor sites reject the default Python user agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) part-assistant-datasheet-fetcher"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):  # 64 KB at a time: constant memory
            h.update(chunk)
    return h.hexdigest()


def load_sources(manifest: Path = MANIFEST) -> list[dict]:
    with manifest.open("rb") as f:
        return tomllib.load(f)["source"]


def fetch(url: str, dest: Path, timeout: float = 60) -> None:
    """Download to a temp file, then rename: a crash never leaves a half-written PDF."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    if not tmp.read_bytes()[:5] == b"%PDF-":
        tmp.unlink()
        raise ValueError(f"not a PDF (site may be blocking scripts): {url}")
    tmp.replace(dest)  # atomic on the same filesystem


def sync(sources: list[dict], raw_dir: Path = RAW_DIR) -> bool:
    """Make raw_dir match the manifest. Returns True if everything verified."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for src in sources:
        dest = raw_dir / src["file"]
        expected = src.get("sha256", "")
        if dest.exists() and expected and sha256_of(dest) == expected:
            print(f"  ok      {src['file']} (cached)")
            continue
        try:
            fetch(src["url"], dest)
        except Exception as e:  # report every failure, keep going with the rest
            print(f"  FAILED  {src['file']}: {e}")
            ok = False
            continue
        actual = sha256_of(dest)
        if not expected:
            print(f"  NEW     {src['file']}  sha256 = \"{actual}\"  <- pin this in the manifest")
            ok = False
        elif actual != expected:
            print(f"  MISMATCH {src['file']}: expected {expected[:12]}…, got {actual[:12]}…")
            ok = False
        else:
            print(f"  ok      {src['file']}")
    return ok


def main() -> None:
    ok = sync(load_sources())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
