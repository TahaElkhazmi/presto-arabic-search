"""Build (anchor, positive, hard-negative) triplets for dense retriever v2.

Source positives: data/training/positives/train.parquet (498,686 unique
query+positive pairs over 316,663 queries; its negative column is all NaN).
Each positive pair is paired with one retrieval-mined hard negative for that
query (from src/mine_hard_negatives.py), rotating through the mined list so a
query with k positives spreads across its mined negatives. Positives without any
mined negative (rare) fall back to a random catalog name that is not one of the
query's positives.

Output parquet columns: anchor, positive, negative - the exact schema
train_lora.py consumes via --triplets. Training data + public catalog only.
"""

import argparse
from pathlib import Path

import pandas as pd

from common import BENCH, DATA, TRAIN, load_jsonl, normalize_arabic, setup_utf8


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--positives", type=Path, default=TRAIN / "positives" / "train.parquet")
    p.add_argument("--mined", type=Path, default=DATA / "teacher" / "mined_negatives_316k.parquet")
    p.add_argument("--out", type=Path, default=DATA / "teacher" / "dense_triplets_316k.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pos = pd.read_parquet(args.positives)[["user_query", "positive_product_name"]].dropna()
    pos = pos.drop_duplicates().reset_index(drop=True)
    mined = pd.read_parquet(args.mined)
    mined_by_q = {q: list(m) for q, m in zip(mined["user_query"], mined["mined"])}

    corpus_names = [row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")]
    rng = pd.Series(corpus_names).sample(frac=1.0, random_state=args.seed).tolist()
    pos_norm_by_q = (
        pos.assign(pn=pos["positive_product_name"].map(normalize_arabic))
        .groupby("user_query")["pn"].agg(set)
    )

    anchors, positives, negatives = [], [], []
    counter: dict[str, int] = {}
    i = 0
    n_fallback = 0
    for q, pname in zip(pos["user_query"], pos["positive_product_name"]):
        cand = mined_by_q.get(q, [])
        if cand:
            k = counter.get(q, 0)
            neg = cand[k % len(cand)]
            counter[q] = k + 1
        else:
            names = pos_norm_by_q.get(q, set())
            while normalize_arabic(rng[i % len(rng)]) in names:
                i += 1
            neg = rng[i % len(rng)]
            i += 1
            n_fallback += 1
        anchors.append(q)
        positives.append(pname)
        negatives.append(neg)

    df = pd.DataFrame({"anchor": anchors, "positive": positives, "negative": negatives})
    # drop any triplet whose normalized negative equals the positive (false negative)
    keep = df["positive"].map(normalize_arabic) != df["negative"].map(normalize_arabic)
    df = df[keep].reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False, compression="zstd")
    print(f"wrote {args.out} : {len(df)} triplets over {df['anchor'].nunique()} queries "
          f"({n_fallback} random-neg fallbacks, {(~keep).sum()} false-neg dropped)")


if __name__ == "__main__":
    main()
