"""Pre-registered validation of the v2 graded curriculum (run BEFORE any training GO).

The v1 validator assumed binary labels. v2's central claim is that our synthetically
assigned grades (3 source / 2 size-variant / 1 close-variant / 0 other) carry REAL
relevance information. That is a falsifiable prediction, and this script tests it:

  On synthetic queries whose text exactly matches a benchmark query, does the rate of
  benchmark-judged relevance INCREASE monotonically with our assigned grade?

If yes, the grading scheme is sound and LambdaLoss (gains = 2^label - 1) will consume
real signal. If the rates are flat across grades, the grades are noise and training
would be a waste of a night - in which case v2 should be abandoned, not trained.

qrels are used for MEASUREMENT ONLY, per the project's standing discipline.
"""

import re

import numpy as np
import pandas as pd

from common import BENCH, DATA, load_jsonl, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates

setup_utf8()

DIALECT_FILLERS = {normalize_arabic(w) for w in (
    "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
    "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}


def shape(queries, label):
    toks = [normalize_arabic(q).split() for q in queries]
    n = len(queries)
    tc = pd.Series([len(t) for t in toks])
    filler = sum(any(w in DIALECT_FILLERS for w in t) for t in toks) / n
    digit = sum(bool(re.search(r"[0-9]", normalize_arabic(q))) for q in queries) / n
    print(f"{label:26s}: n={n}, tokens mean {tc.mean():.2f} "
          f"{{1:{(tc == 1).mean():.2f}, 2:{(tc == 2).mean():.2f}, 3:{(tc == 3).mean():.2f}, "
          f"4+:{(tc >= 4).mean():.2f}}}, filler={filler:.3f}, digit={digit:.3f}")


def main():
    syn = pd.read_parquet(DATA / "teacher" / "synthetic_curriculum_v2_groups.parquet")
    bench_q = [q["query"] for q in load_jsonl(BENCH / "queries.jsonl")]

    print("=== QUERY SHAPE ===")
    shape(bench_q, "benchmark (29,014)")
    shape(list(syn["query"]), "synthetic v2")

    lab = np.concatenate([np.asarray(x) for x in syn["labels"]])
    print("\n=== GRADE DISTRIBUTION ===")
    for g in (3, 2, 1, 0):
        print(f"  grade {g}: {int((lab == g).sum()):>7} ({(lab == g).mean():.1%})")

    print("\n=== THE DECISIVE TEST: does synthetic grade predict benchmark relevance? ===")
    qrels = load_candidates(BENCH / "candidates.jsonl")
    bynorm: dict[str, list[str]] = {}
    for q in load_jsonl(BENCH / "queries.jsonl"):
        bynorm.setdefault(normalize_arabic(q["query"]), []).append(str(q["query_id"]))
    name_ids: dict[str, list[str]] = {}
    for r in load_jsonl(BENCH / "corpus.jsonl"):
        name_ids.setdefault(normalize_arabic(r["name"]), []).append(str(r["item_id"]))

    tot = {g: 0 for g in (3, 2, 1, 0)}
    rel = {g: 0 for g in (3, 2, 1, 0)}
    graded_sum = {g: [] for g in (3, 2, 1, 0)}
    matched = 0
    for _, row in syn.iterrows():
        qids = bynorm.get(normalize_arabic(row["query"]))
        if not qids:
            continue
        matched += 1
        judged = {}
        for qid in qids:
            judged.update(qrels.get(qid, {}))
        for d, l in zip(row["docs"], row["labels"]):
            g = int(l)
            ids = name_ids.get(normalize_arabic(d), [])
            grades = [judged.get(i, 0) for i in ids]
            best = max(grades) if grades else 0
            tot[g] += 1
            rel[g] += 1 if best > 0 else 0
            graded_sum[g].append(best)

    print(f"synthetic queries exactly matching a benchmark query: {matched}\n")
    print(f"{'our grade':>10} {'n':>8} {'% benchmark-relevant':>22} {'mean benchmark grade':>22}")
    rates = {}
    for g in (3, 2, 1, 0):
        if tot[g]:
            r = rel[g] / tot[g]
            mg = float(np.mean(graded_sum[g]))
            rates[g] = r
            print(f"{g:>10} {tot[g]:>8} {r:>21.1%} {mg:>22.3f}")

    print()
    mono = all(rates.get(a, 0) >= rates.get(b, 0) for a, b in ((3, 2), (2, 1), (1, 0)) if a in rates and b in rates)
    if mono:
        print("VERDICT: grades are MONOTONE in benchmark relevance - the grading scheme")
        print("         carries real signal; LambdaLoss will consume it. Training justified.")
    else:
        print("VERDICT: grades are NOT monotone - the scheme is noise. DO NOT TRAIN;")
        print("         abandon v2 or redesign the grade rule.")
    if 0 in rates:
        print(f"\ngrade-0 false-negative rate: {rates[0]:.1%} "
              f"(v1 was 5.2%; provided hards 1.5%; v3 poison 10.2%)")


if __name__ == "__main__":
    main()
