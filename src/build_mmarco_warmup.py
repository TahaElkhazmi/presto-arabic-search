"""Ingest a slice of Arabic MS MARCO into BCE warm-up pairs for Lever 6 Stage-1.

Source: AbderrahmanSkiredj1/MSMARCO_arabic_deduplicated_1m6 (parquet-native,
columns query/positive/negative; MS MARCO translated to Arabic, MSA web-QA
register). The register mismatch vs our Libyan-dialect e-commerce queries is the
exact thing the L6 scout measures - warming general Arabic relevance ranking and
testing whether it transfers through the Stage-2 LambdaLoss adaptation.

Emits DATA/teacher/mmarco_ar_warmup_pairs.parquet with columns q, d, label (raw
text; label 1 = positive passage, 0 = negative passage), ready for
train_reranker.py --loss bce --pairs-parquet.
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from common import DATA, setup_utf8


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--n-triples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DATA / "teacher" / "mmarco_ar_warmup_pairs.parquet")
    args = p.parse_args()

    ds = load_dataset("AbderrahmanSkiredj1/MSMARCO_arabic_deduplicated_1m6",
                      split="train", streaming=True)
    rows = []
    for ex in ds:
        if len(rows) >= args.n_triples:
            break
        q, pos, neg = ex.get("query"), ex.get("positive"), ex.get("negative")
        if q and pos and neg:
            rows.append((q, pos, neg))
    print(f"pulled {len(rows)} triples")

    pos_pairs = pd.DataFrame({"q": [r[0] for r in rows], "d": [r[1] for r in rows], "label": 1.0})
    neg_pairs = pd.DataFrame({"q": [r[0] for r in rows], "d": [r[2] for r in rows], "label": 0.0})
    pairs = pd.concat([pos_pairs, neg_pairs], ignore_index=True)
    pairs = pairs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    pairs["label"] = pairs["label"].astype("float32")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.out, index=False)
    print(f"wrote {len(pairs)} pairs ({pairs['label'].mean():.3f} positive) to {args.out}")


if __name__ == "__main__":
    main()
