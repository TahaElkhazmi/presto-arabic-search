"""Is there a LEGAL graded training signal? Repeat count is the only candidate.

Taha's guideline is to train ListNet against 2^rel - 1 gain weights. The blocker is that
those grades live ONLY in `candidates.jsonl`, which rule 2.4.b makes measurement-only:
using them as training targets is disqualifying, and top rankers hand over code and
methodology for audit. The training files carry no grade column at all - both are just
(user_query, positive_product_name, negative_product_name).

The one grade-like signal that IS inside the training data is repetition: a (query,
positive) pair can appear more than once (22,186 pairs twice, 6,315 three times, up to 8),
which plausibly reflects interaction strength.

This asks whether that proxy actually tracks the benchmark grade. Legal because it only
READS qrels to measure a correlation, exactly as every gate in this project reads them to
score. If repeat count does not track grade, guideline 1 has no legal implementation and
that should be recorded rather than discovered later during a training run.

Matching is on normalized text, since the training files carry names and not item ids.
Expect a small matched set: training positives appear in the same-text benchmark query's
judged names only 4.1% of the time (FINDINGS 2026-07-19).
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, TRAIN, load_jsonl, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates

setup_utf8()


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    queries = {str(q["query_id"]): q["query"] for q in load_jsonl(BENCH / "queries.jsonl")}
    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}

    # (normalized query text, normalized item name) -> best grade seen
    grade_of = {}
    for qid, rel in qrels.items():
        qn = normalize_arabic(queries[qid])
        for iid, g in rel.items():
            k = (qn, normalize_arabic(names[iid]))
            if k not in grade_of or g > grade_of[k]:
                grade_of[k] = g
    print(f"benchmark lookup built: {len(grade_of):,} (query, name) keys")

    d = pd.read_parquet(TRAIN / "positives" / "train.parquet")
    cnt = d.groupby(["user_query", "positive_product_name"]).size()
    print(f"training positives: {len(cnt):,} unique pairs")

    by_count = defaultdict(list)
    matched = 0
    for (q, nm), c in cnt.items():
        g = grade_of.get((normalize_arabic(q), normalize_arabic(nm)))
        if g is None:
            continue
        matched += 1
        by_count[min(int(c), 4)].append(g)

    print(f"matched to a benchmark judgement: {matched:,} "
          f"({100*matched/len(cnt):.2f}% - the 4.1% grounding fact)\n")
    print(f"{'repeat count':<16s}{'n':>9s}{'mean grade':>12s}{'% grade 3':>11s}"
          f"{'% grade<=0':>12s}")
    for c in sorted(by_count):
        g = by_count[c]
        lab = f"{c}" if c < 4 else "4+"
        print(f"{lab:<16s}{len(g):>9,}{sum(g)/len(g):>12.3f}"
              f"{100*sum(1 for x in g if x == 3)/len(g):>11.1f}"
              f"{100*sum(1 for x in g if x <= 0)/len(g):>12.1f}")

    ones = by_count.get(1, [])
    many = [g for c, gs in by_count.items() if c >= 2 for g in gs]
    if ones and many:
        m1, m2 = sum(ones) / len(ones), sum(many) / len(many)
        print(f"\nmean grade: count==1 {m1:.3f}   count>=2 {m2:.3f}   diff {m2-m1:+.3f}")
        pairs = [(a, b) for a in many[:3000] for b in ones[:3000]]
        w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs)
        print(f"AUC of repeat>=2 predicting a HIGHER grade than repeat==1: "
              f"{w/len(pairs):.4f}")
        print("\n0.50 means repetition carries NO grade information and guideline 1 has no")
        print("legal implementation on this data. Above ~0.55 it is worth a training pass.")


if __name__ == "__main__":
    main()
