"""Extract the unique training queries into a benchmark-shaped jsonl for retrieval runs.

Default source: data/training/pairs_with_negatives/train.parquet (the reranker-v2
training universe, 82,140 unique user_query values). With --all, the union with
positives/train.parquet (316,663 unique queries) is used instead, ids U000001..
Synthetic ids are assigned in first-occurrence order, deterministically. These
queries are provided TRAINING data; no benchmark information is involved
(competition rule 2.4.b compliance).
"""

import argparse
import json

import pandas as pd

from common import DATA, TRAIN, setup_utf8


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="union of pairs_with_negatives + positives")
    args = p.parse_args()

    frames = [pd.read_parquet(TRAIN / "pairs_with_negatives" / "train.parquet")["user_query"]]
    prefix, name = "T", "train_queries.jsonl"
    if args.all:
        frames.append(pd.read_parquet(TRAIN / "positives" / "train.parquet")["user_query"])
        prefix, name = "U", "train_queries_all.jsonl"
    seen: dict[str, int] = {}
    for series in frames:
        for q in series:
            if q not in seen:
                seen[q] = len(seen) + 1
    out = DATA / "teacher" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q, i in seen.items():
            f.write(json.dumps({"query_id": f"{prefix}{i:06d}", "query": q}, ensure_ascii=False) + "\n")
    print(f"wrote {out} : {len(seen)} unique training queries")


if __name__ == "__main__":
    main()
