"""Mine retrieval hard negatives for v3 reranker training groups.

Per query: fuse the training-query BM25 + dense runs with the inference RRF
settings (2:1, k=10), walk the fused ranking, and collect candidate names that
survive hygiene filters:
  - normalized name not in the query's provided-positive name set (the v2
    false-negative rule);
  - normalized name not already collected (duplicate names add nothing).
The first --skip-top survivors are discarded (retrieval rank correlates with
true relevance, so the top survivors are the likeliest unlabeled positives;
the 4.1% label-coverage fact makes this a real risk), then the next --n-mined
are kept in fused order.

Output parquet: user_query, mined (list of raw catalog names).
Training queries + public catalog only; no benchmark data (rule 2.4.b).
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from pathlib import Path

import pandas as pd

from common import BENCH, DATA, TRAIN, load_jsonl, normalize_arabic, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import _id_key


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+",
                   default=[Path("runs/bm25_train82k.jsonl"), Path("runs/dense_lora_v1_train82k.jsonl")])
    p.add_argument("--weights", type=float, nargs="+", default=[2.0, 1.0])
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--depth", type=int, default=50, help="fused ranks to consider")
    p.add_argument("--skip-top", type=int, default=3, help="surviving candidates to discard first")
    p.add_argument("--n-mined", type=int, default=8)
    p.add_argument("--queries", type=Path, default=DATA / "teacher" / "train_queries.jsonl")
    p.add_argument("--positives", type=Path,
                   default=TRAIN / "pairs_with_negatives" / "train.parquet",
                   help="parquet with user_query + positive_product_name for the pos-filter")
    p.add_argument("--out", type=Path, default=DATA / "teacher" / "mined_negatives.parquet")
    args = p.parse_args()
    assert len(args.runs) == len(args.weights)

    queries = load_jsonl(args.queries)
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    run_files = [load_run_jsonl(path) for path in args.runs]

    tri = pd.read_parquet(args.positives)
    tri["pos_norm"] = tri["positive_product_name"].map(normalize_arabic)
    pos_norm_by_q = tri.groupby("user_query")["pos_norm"].agg(set)

    rows = []
    n_short = 0
    for row in queries:
        qid, q = str(row["query_id"]), row["query"]
        fused: dict[str, float] = {}
        for run, w in zip(run_files, args.weights):
            for rank, iid in enumerate(run.get(qid, [])[: args.depth], start=1):
                fused[iid] = fused.get(iid, 0.0) + w / (args.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda x: (-x[1], _id_key(x[0])))

        pos_set = pos_norm_by_q.get(q, set())
        mined: list[str] = []
        seen: set[str] = set()
        survivors = 0
        for iid, _ in ranked:
            nm = names[iid]
            norm = normalize_arabic(nm)
            if norm in pos_set or norm in seen:
                continue
            seen.add(norm)
            survivors += 1
            if survivors <= args.skip_top:
                continue
            mined.append(nm)
            if len(mined) >= args.n_mined:
                break
        if len(mined) < args.n_mined:
            n_short += 1
        rows.append((q, mined))

    df = pd.DataFrame(rows, columns=["user_query", "mined"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    lens = df["mined"].map(len)
    print(f"wrote {args.out} : {len(df)} queries, mined/query mean {lens.mean():.2f} "
          f"min {lens.min()} ({n_short} queries under target)")


if __name__ == "__main__":
    main()
