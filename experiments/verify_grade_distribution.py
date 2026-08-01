"""Verify the reported grade semantics of the benchmark, especially grade -1.

A research pass claims the judged pools carry FIVE grade levels, not four:
  3 fully matched, 2 relevant but not fully matched, 1 somehow relevant,
  0 irrelevant, and -1 "hard negative: very similar but not relevant",
with -1 the LARGEST single class at 96,510 instances.

That last class matters strategically: it means the benchmark deliberately seeded
each pool with near-duplicate distractors, which is exactly the containment-saturation
failure we cannot solve. Confirming it tells us the discriminating information is
product-type semantics rather than text overlap, and that our zero cohort is failing
on distractors the annotators explicitly marked as look-alikes.

Measurement only; nothing here feeds a submission.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, setup_utf8
from eval_ndcg import load_candidates

setup_utf8()

CLAIMED = {3: 69965, 2: 65029, 1: 27555, 0: 3540, -1: 96510}


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    c = Counter()
    for q, d in qrels.items():
        for iid, g in d.items():
            c[g] += 1
    total = sum(c.values())

    print(f"{len(qrels):,} queries, {total:,} judged candidates, "
          f"{total / len(qrels):.2f} per query\n")
    print(f"  {'grade':>6s}{'count':>12s}{'claimed':>12s}{'match':>8s}{'share':>9s}")
    for g in sorted(c, reverse=True):
        cl = CLAIMED.get(g)
        ok = "yes" if cl == c[g] else ("--" if cl is None else "NO")
        print(f"  {g:>6d}{c[g]:>12,}{(cl if cl is not None else 0):>12,}"
              f"{ok:>8s}{100 * c[g] / total:>8.1f}%")

    pos = sum(v for g, v in c.items() if g > 0)
    hn = c.get(-1, 0)
    print(f"\n  positives (grade>0): {pos:,} = {pos / len(qrels):.2f} per query")
    print(f"  hard negatives (-1): {hn:,} = {hn / len(qrels):.2f} per query")

    if hn:
        print("\nSTRATEGIC READING")
        print("  Grade -1 is 'very similar but NOT relevant'. The benchmark seeds every")
        print("  pool with deliberate look-alike distractors, which is precisely the")
        print("  containment-saturation cohort we fail on: dozens of textually identical")
        print("  candidates of which only a few are relevant. The separating signal is")
        print("  product-type semantics, not character overlap - and our char-3gram key")
        print("  is by construction blind to it.")

    # do hard negatives concentrate in the queries we fail?
    from common import ROOT, load_run_csv
    from eval_ndcg import ndcg_at_k
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    zero = [q for q, v in nd.items() if v == 0.0]
    good = [q for q, v in nd.items() if v >= 0.9]

    def hn_rate(qs):
        h = sum(sum(1 for g in qrels[q].values() if g == -1) for q in qs)
        t = sum(len(qrels[q]) for q in qs)
        return 100 * h / t if t else 0.0

    print(f"\n  hard-negative share of the judged pool:")
    print(f"    zero-scoring queries : {hn_rate(zero):.1f}%  (n={len(zero)})")
    print(f"    high-scoring queries : {hn_rate(good):.1f}%  (n={len(good)})")
    print("  (higher on the zero cohort would confirm they are distractor-heavy)")


if __name__ == "__main__":
    main()
