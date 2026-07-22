"""Play 2 pre-registered generator validation (runs BEFORE any training GO).

Checks the synthetic curriculum groups against real query statistics and measures
residual label contamination on synthetic queries whose text exactly matches a
benchmark query (qrels for MEASUREMENT ONLY, per rule 2.4.b discipline).
"""

import re
import sys
from pathlib import Path

import pandas as pd

from common import BENCH, DATA, load_jsonl, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates
from stacker_features import char_ngrams, containment

setup_utf8()

DIALECT_FILLERS = {normalize_arabic(w) for w in (
    "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
    "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}


def stats(queries: list[str], label: str):
    toks = [normalize_arabic(q).split() for q in queries]
    n = len(queries)
    tc = pd.Series([len(t) for t in toks])
    filler = sum(any(w in DIALECT_FILLERS for w in t) for t in toks) / n
    digit = sum(bool(re.search(r"[0-9]", normalize_arabic(q))) for q in queries) / n
    chars = pd.Series([len(normalize_arabic(q)) for q in queries])
    print(f"{label:28s}: n={n}, tokens p50={tc.median():.0f} mean={tc.mean():.2f} "
          f"dist={{1: {(tc == 1).mean():.2f}, 2: {(tc == 2).mean():.2f}, 3: {(tc == 3).mean():.2f}, "
          f"4+: {(tc >= 4).mean():.2f}}}, filler={filler:.3f}, digit={digit:.3f}, "
          f"chars p50={chars.median():.0f}")


def main():
    syn = pd.read_parquet(DATA / "teacher" / "synthetic_curriculum_groups.parquet")
    bench_q = [q["query"] for q in load_jsonl(BENCH / "queries.jsonl")]
    train_q = [q["query"] for q in load_jsonl(DATA / "teacher" / "train_queries.jsonl")]

    print("=== QUERY SHAPE ===")
    stats(bench_q, "benchmark (29,014)")
    stats(train_q, "real training queries")
    stats(list(syn["query"]), "synthetic")

    print("\n=== CONTAINMENT SEPARATION (5k sample) ===")
    sample = syn.sample(n=min(5000, len(syn)), random_state=0)
    cpos_all, cneg_all = [], []
    for _, row in sample.iterrows():
        qn = normalize_arabic(row["query"])
        core = " ".join(t for t in qn.split() if t not in DIALECT_FILLERS)
        qg = char_ngrams(core, 3)
        docs, labels = list(row["docs"]), list(row["labels"])
        for d, l in zip(docs, labels):
            c = containment(qg, char_ngrams(normalize_arabic(d), 3))
            (cpos_all if l > 0 else cneg_all).append(c)
    cp, cn = pd.Series(cpos_all), pd.Series(cneg_all)
    print(f"cont(q,pos): mean {cp.mean():.3f}  p10 {cp.quantile(0.1):.3f}")
    print(f"cont(q,neg): mean {cn.mean():.3f}  p90 {cn.quantile(0.9):.3f}")

    print("\n=== CONTAMINATION ON EXACT BENCHMARK-QUERY MATCHES (measurement only) ===")
    qrels = load_candidates(BENCH / "candidates.jsonl")
    bynorm: dict[str, list[str]] = {}
    for q in load_jsonl(BENCH / "queries.jsonl"):
        bynorm.setdefault(normalize_arabic(q["query"]), []).append(str(q["query_id"]))
    name_ids: dict[str, list[str]] = {}
    for r in load_jsonl(BENCH / "corpus.jsonl"):
        name_ids.setdefault(normalize_arabic(r["name"]), []).append(str(r["item_id"]))

    matched = n_neg = n_contam = pos_judged = pos_rows = 0
    for _, row in syn.iterrows():
        qn = normalize_arabic(row["query"])
        qids = bynorm.get(qn)
        if not qids:
            continue
        matched += 1
        rel = {}
        for qid in qids:
            rel.update(qrels.get(qid, {}))
        for d, l in zip(row["docs"], row["labels"]):
            ids = name_ids.get(normalize_arabic(d), [])
            judged_pos = any(rel.get(i, 0) > 0 for i in ids)
            if l > 0:
                pos_rows += 1
                pos_judged += judged_pos
            else:
                n_neg += 1
                n_contam += judged_pos
    print(f"synthetic queries exactly matching a benchmark query: {matched}")
    if n_neg:
        print(f"negatives on matched queries: {n_neg}, judged-relevant (contaminated): "
              f"{n_contam} ({n_contam / n_neg:.3%})  [v3 poison was 10.2%; provided hards 1.5%]")
        print(f"positives on matched queries judged-relevant: {pos_judged}/{pos_rows} "
              f"({pos_judged / max(pos_rows, 1):.1%})  [provided-label anchor: 4.1%]")

    print("\n=== SAMPLES ===")
    for _, row in syn.sample(n=20, random_state=1).iterrows():
        negs = [d for d, l in zip(row["docs"], row["labels"]) if l == 0]
        print(f"Q: {row['query']}")
        print(f"   POS: {row['docs'][0]}")
        print(f"   NEG: {' | '.join(n[:34] for n in negs[:4])}")


if __name__ == "__main__":
    main()
