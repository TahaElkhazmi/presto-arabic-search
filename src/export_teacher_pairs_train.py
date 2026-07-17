"""Export TRAINING-query pools for teacher scoring (phase 3 step 4).

Per PHASE3_DISTILLATION_PLAN.md section 2.3: for each of the 82,140 unique
training queries, candidates = top --top-n of the RRF fusion (2:1, k=10) of the
BM25 and dense training-query runs, plus every provided positive for that query
injected. Candidates are deduped by raw product NAME (the teacher scores
(query, name) pairs; duplicate names would waste teacher time), keeping the
lowest catalog id as the representative iid. Positives whose name is absent
from the catalog keep iid "" (they still get scored; distillation only needs
the margin between names).

Output columns: qid, iid, query, name, text_a, text_b, is_pos (zstd parquet).
Row order is deterministic for stable resume blocks on the scoring side.

This uses ONLY training data and retrieval over the public catalog; no benchmark
queries, qrels, or judgments are involved (rule 2.4.b).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import BENCH, DATA, TRAIN, load_jsonl, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import _id_key
from reranker_common import format_document, format_query


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="training-query run jsonls")
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--top-n", type=int, default=22, help="fused candidates per query")
    p.add_argument("--queries", type=Path, default=DATA / "teacher" / "train_queries.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1, help="split output into N parquet shards")
    args = p.parse_args()
    assert len(args.runs) == len(args.weights)

    queries = load_jsonl(args.queries)
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    # canonical (lowest) id per raw name, for representative iids
    name_to_lowid: dict[str, str] = {}
    for iid, nm in names.items():
        if nm not in name_to_lowid or iid < name_to_lowid[nm]:
            name_to_lowid[nm] = iid

    run_files = [load_run_jsonl(path) for path in args.runs]

    tri = pd.read_parquet(TRAIN / "pairs_with_negatives" / "train.parquet")
    positives: dict[str, list[str]] = defaultdict(list)
    for q, pos in zip(tri["user_query"], tri["positive_product_name"]):
        if pos not in positives[q]:
            positives[q].append(pos)

    rows = {"qid": [], "iid": [], "query": [], "name": [], "text_a": [], "text_b": [], "is_pos": []}
    n_pos_injected = 0
    n_pos_already = 0
    for row in queries:
        qid, q = str(row["query_id"]), row["query"]
        fused: dict[str, float] = defaultdict(float)
        for run, w in zip(run_files, args.weights):
            for rank, iid in enumerate(run.get(qid, []), start=1):
                fused[iid] += w / (args.rrf_k + rank)
        top = sorted(fused.items(), key=lambda x: (-x[1], _id_key(x[0])))[: args.top_n]

        pool_names: list[str] = []
        seen: set[str] = set()
        for iid, _ in top:
            nm = names[iid]
            if nm not in seen:
                seen.add(nm)
                pool_names.append(nm)
        pos_set = set()
        for pos in positives.get(q, []):
            pos_set.add(pos)
            if pos in seen:
                n_pos_already += 1
            else:
                seen.add(pos)
                pool_names.append(pos)
                n_pos_injected += 1

        ta = format_query(q)
        for nm in pool_names:
            rows["qid"].append(qid)
            rows["iid"].append(name_to_lowid.get(nm, ""))
            rows["query"].append(q)
            rows["name"].append(nm)
            rows["text_a"].append(ta)
            rows["text_b"].append(format_document(nm))
            rows["is_pos"].append(nm in pos_set)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        per = (len(df) + args.shards - 1) // args.shards
        for s in range(args.shards):
            part = df.iloc[s * per : (s + 1) * per]
            path = args.out.with_suffix(f".shard{s}.parquet")
            part.to_parquet(path, index=False, compression="zstd")
            print(f"wrote {path} : {len(part)} pairs")
    else:
        df.to_parquet(args.out, index=False, compression="zstd")
    print(
        f"total {len(df)} pairs over {len(queries)} queries "
        f"(mean pool {len(df) / len(queries):.1f}); positives injected {n_pos_injected}, "
        f"already in pool {n_pos_already}"
    )


if __name__ == "__main__":
    main()
