"""Retrieval eval: for golden cases with a source page, is that page in the top k?

    uv run python -m part_assistant.eval_retrieval [--models M1,M2] [--no-rerank] [--show-misses]

Metrics (averaged over cases):
  recall@k  share of a case's source pages found in the top k chunks
            (a 2-source question that finds 1 counts 0.5)
  MRR       1 / rank of the first chunk from a source page (0 if not in top 50)

With ~32 cases, one case is ~3 points: do not read much into differences of 1-2 cases.
"""

import argparse
import time

from part_assistant.chunk import Chunk, load_chunks
from part_assistant.golden import Case, load_cases
from part_assistant.retrieve import DEFAULT_MODEL, Embedder, Hit, Reranker, Retriever

KS = (1, 3, 5, 10)
DEPTH = 50

# (mode, model filter, rerank)
CONFIGS = [
    ("bm25", False, False),
    ("bm25", True, False),
    ("vector", False, False),
    ("vector", True, False),
    ("hybrid", False, False),
    ("hybrid", True, False),
    ("hybrid", True, True),
    ("vector", True, True),  # the default pipeline
]


def covers(chunk: Chunk, source: dict) -> bool:
    return chunk.file == source["file"] and chunk.page <= source["page"] <= chunk.page_end


def first_rank(hits: list[Hit], source: dict) -> int | None:
    return next((r for r, h in enumerate(hits, start=1) if covers(h.chunk, source)), None)


def score_case(hits: list[Hit], case: Case) -> dict:
    ranks = [first_rank(hits, s) for s in case.sources]
    found = [r for r in ranks if r is not None]
    out = {f"r@{k}": sum(r <= k for r in found) / len(ranks) for k in KS}
    out["mrr"] = 1 / min(found) if found else 0.0
    out["ranks"] = ranks
    return out


def evaluate(retriever: Retriever, cases: list[Case], mode: str = "vector",
             use_filter: bool = True, rerank: bool = True) -> tuple[dict, list]:
    rows = []
    t0 = time.perf_counter()
    for case in cases:
        where = retriever.model_filter(case.question) if use_filter else None
        hits = retriever.search(case.question, k=DEPTH, mode=mode, depth=DEPTH,
                                where=where, rerank=rerank)
        rows.append((case, score_case(hits, case), hits))
    metrics = [f"r@{k}" for k in KS] + ["mrr"]
    summary = {m: sum(r[m] for _, r, _ in rows) / len(rows) for m in metrics}
    summary["ms/query"] = (time.perf_counter() - t0) * 1000 / len(cases)
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODEL, help="comma-separated embedding models")
    ap.add_argument("--no-rerank", action="store_true", help="skip configs with a reranker")
    ap.add_argument("--show-misses", action="store_true", help="list cases not ranked #1")
    args = ap.parse_args()

    # The golden questions are about the public datasheets, asked as a public user:
    # internal documents are invisible to that user, so they are not candidates here.
    chunks = [c for c in load_chunks() if c.access_level == "public"]
    cases = [c for c in load_cases() if c.sources]
    reranker = None if args.no_rerank else Reranker()
    print(f"{len(chunks)} chunks, {len(cases)} cases with a source page\n")
    print(f"{'embedding':22} {'mode':7} {'filter':6} {'rerank':6} "
          + " ".join(f"{'r@' + str(k):>5}" for k in KS) + f" {'MRR':>5} {'ms/q':>6}")

    for n, model in enumerate(args.models.split(",")):
        t0 = time.perf_counter()
        emb = Embedder(model)
        retriever = Retriever(chunks, emb, reranker)
        setup = time.perf_counter() - t0
        for mode, use_filter, rerank in CONFIGS:
            if (mode == "bm25" and n > 0) or (rerank and reranker is None):
                continue  # bm25 does not depend on the embedding model
            summary, rows = evaluate(retriever, cases, mode, use_filter, rerank)
            name = "-" if mode == "bm25" else model.split("/")[-1]
            print(f"{name:22} {mode:7} {'model' if use_filter else '-':6} "
                  f"{'yes' if rerank else '-':6} "
                  + " ".join(f"{summary[f'r@{k}']:5.2f}" for k in KS)
                  + f" {summary['mrr']:5.2f} {summary['ms/query']:6.1f}")
            if args.show_misses:
                for case, r, hits in rows:
                    if r["r@1"] < 1:
                        top = ", ".join(f"{h.chunk.file.split('.')[0]}:p{h.chunk.page}"
                                        for h in hits[:3])
                        print(f"    not #1 {case.id:12} ranks={r['ranks']}  top3: {top}")
        print(f"  ({model}: load + embed {setup:.1f}s, {emb.n_embedded} chunks embedded)\n")


if __name__ == "__main__":
    main()
