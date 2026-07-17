"""Export (query, candidate) pool pairs to parquet for off-machine teacher scoring.

Pools replicate src/rerank.py exactly: per query, the deduped union of the top
--depth ids from each input run. Rows carry raw texts plus the pre-formatted
template strings (text_a, text_b) so the scoring side needs zero knowledge of
the chat template; a template drift on the scorer becomes impossible.

Output columns: qid, iid, query, name, text_a, text_b (zstd parquet).
Row order is deterministic (query order of queries.jsonl, pool order within),
so the scorer's block indices are stable across resumes.
"""

import argparse
from pathlib import Path

import pandas as pd

from common import BENCH, load_jsonl, select_queries, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import build_pools
from reranker_common import format_document, format_query


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--n-queries", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert len(args.runs) == len(args.depths) == len(args.weights)

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, args.n_queries)
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    pools, _ = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)

    rows = {"qid": [], "iid": [], "query": [], "name": [], "text_a": [], "text_b": []}
    for qid in qids:
        q = qtext[qid]
        ta = format_query(q)
        for iid in pools[qid]:
            rows["qid"].append(qid)
            rows["iid"].append(iid)
            rows["query"].append(q)
            rows["name"].append(names[iid])
            rows["text_a"].append(ta)
            rows["text_b"].append(format_document(names[iid]))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} : {len(df)} pairs, {len(qids)} queries, {mb:.1f} MB")


if __name__ == "__main__":
    main()
