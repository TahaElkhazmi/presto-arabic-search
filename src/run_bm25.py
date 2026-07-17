"""Full-catalog BM25 retrieval (rank-bm25 Okapi, word-level normalized Arabic tokens).

Writes the same ranking in both run formats:
  runs/<name>.csv    submission format (query_id,product_id)
  runs/<name>.jsonl  official evaluate.py format

Per COMPETITION_GUIDE.md fact 1, this should score ~0.2407 nDCG@10 on the first
1,000 queries.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from common import BENCH, RUNS, load_jsonl, select_queries, setup_utf8, tokenize, write_run_csv


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--n-queries", type=int, default=1000, help="0 = all queries")
    p.add_argument("--qids", type=Path, default=None, help="optional qid file, e.g. splits/val_qids.txt")
    p.add_argument("--queries", type=Path, default=BENCH / "queries.jsonl", help="query jsonl file")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--name", default=None, help="run name (default bm25_first<N>)")
    args = p.parse_args()

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = select_queries(load_jsonl(args.queries), args.qids, args.n_queries)
    name = args.name or f"bm25_first{len(queries)}"
    print(f"corpus={len(corpus)} queries={len(queries)} run={name}")

    item_ids = [str(row["item_id"]) for row in corpus]
    t0 = time.time()
    corpus_tokens = [tokenize(row["name"]) for row in tqdm(corpus, desc="tokenize corpus")]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"index built in {time.time() - t0:.1f}s")

    run: dict[str, list[str]] = {}
    # Overfetch before sorting: argpartition alone picks an arbitrary subset of a
    # tie group at the cutoff, but sorting ties by doc index (= low product id
    # first) is measurably better, so the tie group must be fully inside the pool.
    pool = min(args.top_k + 512, len(item_ids))
    for row in tqdm(queries, desc="retrieve"):
        scores = bm25.get_scores(tokenize(row["query"]))
        top = np.argpartition(-scores, pool - 1)[:pool]
        top = top[np.lexsort((top, -scores[top]))]  # score desc, then doc index asc
        run[str(row["query_id"])] = [item_ids[i] for i in top[: args.top_k]]

    RUNS.mkdir(exist_ok=True)
    write_run_csv(RUNS / f"{name}.csv", {q: ids[:10] for q, ids in run.items()})
    with open(RUNS / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in run.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"wrote runs/{name}.csv and runs/{name}.jsonl")


if __name__ == "__main__":
    main()
