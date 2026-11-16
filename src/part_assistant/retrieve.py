"""Retrieval over chunks: vectors (+ optional BM25) -> model filter -> reranker.

Default pipeline (chosen by measurement, see design/retrieval.md):
  1. filter   if the question names a model ("TEC3200"), search only that model's docs
  2. vector   bge-m3 (multilingual) finds the top 20 chunks by meaning
  3. rerank   a cross-encoder reads (question, chunk) pairs and re-orders them

BM25 and hybrid (BM25 + vector via Reciprocal Rank Fusion) are kept as options.
They lost here because the questions are Chinese and the datasheets are English:
after tokenizing, a Chinese question leaves BM25 with little more than the model
name, so its ranking is mostly noise. Exact part-number matching is done by the
model filter instead.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from part_assistant.chunk import Chunk

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
EMB_DIR = Path("data/processed/emb")
RRF_K = 60  # standard constant from the RRF paper; damps the weight of top ranks
RERANK_DEPTH = 20  # how many first-stage candidates the reranker re-orders

# Some models were trained with a prefix telling them which side of the pair a text is.
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
}


def _load_on_gpu(model):
    """fp16 on the GPU: half the VRAM and faster, with the same ranking in practice."""
    if _device() == "cuda":
        model.half()
    return model


def _device() -> str:
    """PART_ASSISTANT_RETRIEVAL_DEVICE=cpu leaves the GPU to a local LLM: on an 8 GB card
    the two retrieval models need about 4.3 GB, too much to share with vLLM."""
    import torch

    forced = os.environ.get("PART_ASSISTANT_RETRIEVAL_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9µ°%±]+(?:[-./][a-z0-9µ]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase words; compound words are kept AND split.

    "TEC3200-54-074NA" -> ["tec3200-54-074na", "tec3200", "54", "074na"], so a
    question that only says "TEC3200" still matches. Chinese characters are not
    words for BM25 here: only the part numbers and numbers in a Chinese question match.
    """
    tokens: list[str] = []
    for word in _WORD_RE.findall(text.lower()):
        tokens.append(word)
        parts = re.split(r"[-./]", word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

class Embedder:
    """Text -> unit vectors. Chunk vectors are cached on disk per model, keyed by
    chunk_id (a hash of the content), so only new or changed chunks are embedded.
    Vectors from different models live in different files and are never mixed."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path = EMB_DIR, model=None):
        if model is None:  # tests pass a fake model
            from sentence_transformers import SentenceTransformer  # slow import

            model = _load_on_gpu(SentenceTransformer(model_name, device=_device()))
            # Our chunks are < ~700 tokens; never exceed what the model supports
            # (e5 has 512 position embeddings: longer input crashes on the GPU).
            model.max_seq_length = min(1024, model.max_seq_length)
        self.model = model
        self.model_name = model_name
        self.query_prefix, self.doc_prefix = PREFIXES.get(model_name, ("", ""))
        self.cache = cache_dir / (model_name.replace("/", "__") + ".npz")
        self.n_embedded = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=16, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_prefix + query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        cached: dict[str, np.ndarray] = {}
        if self.cache.exists():
            data = np.load(self.cache)
            cached = dict(zip(data["ids"].tolist(), data["vecs"], strict=True))
        todo = [c for c in chunks if c.chunk_id not in cached]
        if todo:
            new = self._encode([self.doc_prefix + c.text for c in todo])
            cached.update(zip([c.chunk_id for c in todo], new, strict=True))
        vecs = np.stack([cached[c.chunk_id] for c in chunks])
        if todo or len(cached) != len(chunks):  # save only current chunks: drops stale ones
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache.with_suffix(".part.npz")
            np.savez(tmp, ids=np.array([c.chunk_id for c in chunks]), vecs=vecs)
            tmp.replace(self.cache)  # atomic, like every other cache in this project
        self.n_embedded = len(todo)
        return vecs


class Reranker:
    """Cross-encoder: reads question and chunk TOGETHER and scores relevance.
    More accurate than comparing two separately made vectors, but it must run for
    every (question, chunk) pair at query time, so it only re-orders a short list."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = _load_on_gpu(CrossEncoder(model_name, device=_device(), max_length=1024))
        self.model = model
        self.model_name = model_name

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16)]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # "bm25"/"vector"/"rerank" -> rank


def rrf(rankings: dict[str, list[int]], k: int = RRF_K,
        weights: dict[str, float] | None = None) -> list[tuple[int, float, dict]]:
    """Reciprocal Rank Fusion: score = sum over lists of weight / (k + rank).

    Uses only ranks, never raw scores: BM25 scores (0..30) and cosine similarity
    (0..1) are on different scales and cannot be added directly.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, order in rankings.items():
        for rank, idx in enumerate(order, start=1):
            w = (weights or {}).get(name, 1.0)
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
            ranks.setdefault(idx, {})[name] = rank
    return sorted(((i, s, ranks[i]) for i, s in scores.items()), key=lambda t: -t[1])


def model_alias(model: str) -> str:
    """'TEC1600-12-074xA' -> 'tec1600': the part people actually write in questions."""
    return re.split(r"[-\s]", model)[0].lower()


class Retriever:
    MODES = ("bm25", "vector", "hybrid")

    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None,
                 reranker: Reranker | None = None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.embedder = embedder
        self.vecs = embedder.embed_chunks(chunks) if embedder else None
        self.reranker = reranker
        self.aliases = {model_alias(c.model): c.model for c in chunks}

    def detect_models(self, query: str) -> set[str]:
        """Models named in the query ('TEC3200 的效率' -> {'TEC3200-54-074xA'})."""
        words = set(tokenize(query))
        return {model for alias, model in self.aliases.items() if alias in words}

    def model_filter(self, query: str) -> Callable[[Chunk], bool] | None:
        """None when no model is named: then nothing is filtered out."""
        models = self.detect_models(query)
        return (lambda c: c.model in models) if models else None

    def _bm25_order(self, query: str, depth: int, allowed: np.ndarray) -> list[int]:
        scores = np.where(allowed, self.bm25.get_scores(tokenize(query)), 0.0)
        order = np.argsort(-scores)[:depth]
        return [int(i) for i in order if scores[i] > 0]  # no shared word = no match

    def _vector_order(self, query: str, depth: int, allowed: np.ndarray) -> list[int]:
        sims = self.vecs @ self.embedder.embed_query(query)  # unit vectors: dot = cosine
        sims = np.where(allowed, sims, -np.inf)
        return [int(i) for i in np.argsort(-sims)[:min(depth, int(allowed.sum()))]]

    def search(self, query: str, k: int = 5, mode: str = "vector", depth: int = 50,
               where: Callable[[Chunk], bool] | None = None,
               rerank: bool = True) -> list[Hit]:
        """where: keep only chunks for which it returns True (model, access level...).
        Filtering happens BEFORE ranking, so all k results are allowed chunks.
        rerank: re-order the first RERANK_DEPTH candidates (if a Reranker was given)."""
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode != "bm25" and self.embedder is None:
            raise ValueError(f"mode {mode!r} needs an Embedder")
        allowed = np.array([where is None or where(c) for c in self.chunks])
        rankings: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self._bm25_order(query, depth, allowed)
        if mode in ("vector", "hybrid"):
            rankings["vector"] = self._vector_order(query, depth, allowed)
        hits = [Hit(self.chunks[i], s, r) for i, s, r in rrf(rankings)]

        if rerank and self.reranker and hits:
            head = hits[:RERANK_DEPTH]
            scores = self.reranker.scores(query, [h.chunk.text for h in head])
            head = [Hit(h.chunk, s, h.ranks) for s, h in sorted(
                zip(scores, head, strict=True), key=lambda t: -t[0])]
            for rank, h in enumerate(head, start=1):
                h.ranks["rerank"] = rank
            hits = head + hits[RERANK_DEPTH:]  # beyond the reranked head: original order
        return hits[:k]
