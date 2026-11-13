import zlib
from pathlib import Path

import numpy as np
import pytest

from part_assistant.chunk import Chunk
from part_assistant.retrieve import Embedder, Reranker, Retriever, model_alias, rrf, tokenize


def chunk(cid: str, model: str, text: str, page: int = 1) -> Chunk:
    return Chunk(chunk_id=cid, file=f"{model.lower()}.pdf", vendor="V", model=model,
                 kind="text", page=page, page_end=page, section="", text=text)


CHUNKS = [
    chunk("a", "TEC1600-12-074xA", "TEC1600-12-074NA efficiency 92% at 100% load"),
    chunk("b", "TEC3200-54-074xA", "TEC3200-54-074NA efficiency 94% at 100% load"),
    chunk("c", "TEC3200-54-074xA", "TEC3200 hold-up time 9 ms at 3200 W"),
    chunk("d", "CSU2400AT", "CSU2400AT MTBF 400k hours"),
]


class FakeModel:
    """Bag of words hashed into 64 dims. Same words -> similar vectors."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += len(texts)
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the text."""

    def predict(self, pairs, **_):
        return [len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs]


# --- pieces

def test_tokenize_keeps_and_splits_part_numbers():
    assert tokenize("TEC3200-54-074NA") == ["tec3200-54-074na", "tec3200", "54", "074na"]
    assert "tec3200" in tokenize("TEC3200 的效率是多少？")


def test_rrf_rewards_agreement_and_ignores_raw_scores():
    fused = [i for i, _, _ in rrf({"bm25": [1, 2], "vector": [2, 3]})]
    assert fused[0] == 2, "an item ranked by both lists beats one ranked #1 by only one"
    assert [i for i, _, _ in rrf({"bm25": [1], "vector": [2]}, weights={"bm25": 0})][0] == 2


def test_model_alias():
    assert model_alias("TEC1600-12-074xA") == "tec1600"
    assert model_alias("CSU2400AT") == "csu2400at"


# --- search

def test_model_filter_keeps_only_named_model():
    r = Retriever(CHUNKS)
    query = "92% 94%"  # words that only some chunks have (in a 4-chunk corpus, a word in
    # half the chunks gets BM25 IDF 0 and matches nothing)
    unfiltered = r.search(query, k=10, mode="bm25")
    assert {h.chunk.model for h in unfiltered} == {"TEC1600-12-074xA", "TEC3200-54-074xA"}
    where = r.model_filter("TEC3200 在 100% 負載的效率？")
    hits = r.search(query, k=10, mode="bm25", where=where)
    assert {h.chunk.model for h in hits} == {"TEC3200-54-074xA"}


def test_no_model_named_means_no_filter():
    assert Retriever(CHUNKS).model_filter("哪一台電源效率最高？") is None


def test_bm25_matches_exact_part_number():
    hits = Retriever(CHUNKS).search("CSU2400AT", k=1, mode="bm25")
    assert hits[0].chunk.chunk_id == "d"


def test_vector_mode_needs_embedder():
    with pytest.raises(ValueError):
        Retriever(CHUNKS).search("x", mode="vector")


def test_reranker_reorders_candidates(tmp_path: Path):
    emb = Embedder("fake", cache_dir=tmp_path, model=FakeModel())
    r = Retriever(CHUNKS, emb, Reranker("fake", model=FakeCrossEncoder()))
    hits = r.search("TEC3200 hold-up time", k=3)
    assert hits[0].chunk.chunk_id == "c"
    assert hits[0].ranks["rerank"] == 1 and "vector" in hits[0].ranks


# --- embedding cache

def test_embedding_cache_only_embeds_new_chunks(tmp_path: Path):
    model = FakeModel()
    Embedder("fake", cache_dir=tmp_path, model=model).embed_chunks(CHUNKS)
    assert model.calls == len(CHUNKS)

    again = Embedder("fake", cache_dir=tmp_path, model=model)
    again.embed_chunks(CHUNKS)
    assert model.calls == len(CHUNKS) and again.n_embedded == 0, "all from cache"

    changed = CHUNKS[:3] + [chunk("e", "CSU2400AT", "CSU2400AT MTBF > 400k hours")]
    vecs = Embedder("fake", cache_dir=tmp_path, model=model).embed_chunks(changed)
    assert model.calls == len(CHUNKS) + 1 and vecs.shape == (4, 64)
    assert "d" not in np.load(tmp_path / "fake.npz")["ids"].tolist(), "stale vector dropped"


def test_each_model_has_its_own_cache_file(tmp_path: Path):
    a = Embedder("org/model-a", cache_dir=tmp_path, model=FakeModel())
    b = Embedder("org/model-b", cache_dir=tmp_path, model=FakeModel())
    assert a.cache != b.cache, "vectors from different models must never be mixed"


# --- real data: the retrieval quality gate

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


@pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="run python -m part_assistant.chunk first")
def test_retrieval_quality_gate():
    """Full default pipeline (model filter + bge-m3 + reranker) on the golden set.
    Measured 2026-11-13 on RTX 5070 Laptop: r@5 1.00, MRR 0.88."""
    from part_assistant.chunk import load_chunks
    from part_assistant.eval_retrieval import evaluate
    from part_assistant.golden import load_cases

    retriever = Retriever(load_chunks(), Embedder(), Reranker())
    summary, _ = evaluate(retriever, [c for c in load_cases() if c.sources])
    assert summary["r@5"] >= 0.90
    assert summary["mrr"] >= 0.80
