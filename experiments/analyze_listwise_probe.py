"""Significance and failure analysis for the PLAY B constrained listwise probe.

The headline delta is a PAIRED comparison in which most queries are untouched, so
the relevant noise band is the standard error of the per-query delta, not the
bootstrap spread of nDCG itself. Reports a paired t-statistic and a bootstrap CI,
then the queries the judge changed, so the effect can be read qualitatively.
"""

import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, setup_utf8

setup_utf8()

REC = ROOT / "outputs" / "probe_listwise_constrained.json"


def paired(deltas, label):
    n = len(deltas)
    mean = sum(deltas) / n
    sd = st.pstdev(deltas) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n else 0.0
    t = mean / se if se else 0.0
    rng = random.Random(0)
    boots = []
    for _ in range(5000):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo, hi = boots[125], boots[4875]
    print(f"{label:<26s} n={n:<5d} mean={mean:+.6f}  se={se:.6f}  t={t:+.2f}  "
          f"95% CI [{lo:+.6f}, {hi:+.6f}]")


def main():
    recs = json.load(open(REC, encoding="utf-8"))
    d = [r["ls_ndcg"] - r["base_ndcg"] for r in recs]
    dw = [r["ls_ndcg"] - r["base_ndcg"] for r in recs if r["wide"]]
    dn = [r["ls_ndcg"] - r["base_ndcg"] for r in recs if not r["wide"]]
    changed = [r for r in recs if abs(r["ls_ndcg"] - r["base_ndcg"]) > 1e-9]

    print("PAIRED DELTAS (listwise minus live stack)\n")
    paired(d, "all queries")
    paired(dw, "wide band")
    paired(dn, "narrow band")
    print()
    paired([r["ls_ndcg"] - r["base_ndcg"] for r in changed], "changed queries only")

    print(f"\nqueries whose top-10 changed at all: {len(changed)}/{len(recs)} "
          f"({100 * len(changed) / len(recs):.0f}%)")
    gain = sum(x for x in d if x > 0)
    lose = sum(x for x in d if x < 0)
    print(f"total gain {gain:+.4f} over {sum(1 for x in d if x > 0)} queries, "
          f"total loss {lose:+.4f} over {sum(1 for x in d if x < 0)} queries")

    # how confident was the judge when it moved, and did confidence predict success?
    hi = [r for r in changed if r["conf"] and r["conf"][0] >= 0.9]
    lo = [r for r in changed if r["conf"] and r["conf"][0] < 0.9]
    for lab, grp in (("first-pick conf >= 0.90", hi), ("first-pick conf < 0.90", lo)):
        if grp:
            m = sum(r["ls_ndcg"] - r["base_ndcg"] for r in grp) / len(grp)
            print(f"  {lab:<26s} n={len(grp):<4d} mean delta {m:+.4f}")

    print("\n--- biggest LOSSES ---")
    for r in sorted(changed, key=lambda r: r["ls_ndcg"] - r["base_ndcg"])[:6]:
        print(f"  {r['ls_ndcg'] - r['base_ndcg']:+.3f}  wide={r['wide']}  "
              f"conf1={r['conf'][0]:.2f}  {r['query'][:48]}")
        print(f"      base top1: {r['base_names'][0][:60]}")
        print(f"      ls   top1: {r['ls_names'][0][:60]}")
    print("\n--- biggest GAINS ---")
    for r in sorted(changed, key=lambda r: -(r["ls_ndcg"] - r["base_ndcg"]))[:6]:
        print(f"  {r['ls_ndcg'] - r['base_ndcg']:+.3f}  wide={r['wide']}  "
              f"conf1={r['conf'][0]:.2f}  {r['query'][:48]}")
        print(f"      base top1: {r['base_names'][0][:60]}")
        print(f"      ls   top1: {r['ls_names'][0][:60]}")


if __name__ == "__main__":
    main()
