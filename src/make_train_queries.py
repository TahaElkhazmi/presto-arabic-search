"""Extract the unique training queries into a benchmark-shaped jsonl for retrieval runs.

Source: data/training/pairs_with_negatives/train.parquet (the reranker-v2 training
universe, 82,140 unique user_query values). Synthetic ids T000001.. are assigned in
first-occurrence order, deterministically. These queries are provided TRAINING data;
no benchmark information is involved (competition rule 2.4.b compliance).
"""

import json

import pandas as pd

from common import DATA, TRAIN, setup_utf8


def main():
    setup_utf8()
    df = pd.read_parquet(TRAIN / "pairs_with_negatives" / "train.parquet")
    seen: dict[str, int] = {}
    for q in df["user_query"]:
        if q not in seen:
            seen[q] = len(seen) + 1
    out = DATA / "teacher" / "train_queries.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q, i in seen.items():
            f.write(json.dumps({"query_id": f"T{i:06d}", "query": q}, ensure_ascii=False) + "\n")
    print(f"wrote {out} : {len(seen)} unique training queries")


if __name__ == "__main__":
    main()
