"""Compare two submission CSVs across every evaluation slice this project trusts.

Generic replacement for the per-experiment validate_* scripts: those each rebuilt the
harness to compare one specific pair, which is slow and re-implements the transform.
Two files that already exist can be compared directly, and both come from the production
rerank.py path, so there is no harness-versus-production drift to worry about.

Reports the CLEAN arbiter (untouched remainder MINUS holdout_remainder8012) separately,
because the plain "untouched remainder" is 71% the pool recent campaigns tuned on.
"""

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k

setup_utf8()

SPENT = ("dev_rand2000", "dev_unseen2000", "dev_probe2000", "dev_recal2000",
         "dev_ens2000", "dev_final2000", "dev_listwise200", "probe_filler800",
         "probe_band600", "probe_judge150")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="run csv treated as the incumbent")
    p.add_argument("--cand", required=True, help="run csv treated as the challenger")
    a = p.parse_args()

    qrels = load_candidates(BENCH / "candidates.jsonl")
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    base = load_run_csv(ROOT / "runs" / a.base)
    cand = load_run_csv(ROOT / "runs" / a.cand)

    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in SPENT:
        q = ROOT / "splits" / f"{f}.txt"
        if q.exists():
            used |= load_qid_file(q)
    arb = set(all_qids) - used
    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")

    import random
    shuf = sorted(hold, key=int)
    random.Random(77).shuffle(shuf)
    tune, conf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])

    slices = [
        ("tune half (fitted)", tune),
        ("confirm half", conf),
        ("ARBITER (9,675)", arb),
        ("CLEAN arbiter (no holdout)", arb - hold),
        ("val", val),
        ("FULL BENCHMARK", set(all_qids)),
    ]

    print(f"base {a.base}\ncand {a.cand}\n")
    for label, qs in slices:
        qs = [q for q in sorted(qs, key=int) if q in qrels and q in base and q in cand]
        x = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
        y = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
        d = [b - a_ for a_, b in zip(x, y)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<28s} n={len(qs):>6,}  base {sum(x)/len(x):.6f}  "
              f"cand {sum(y)/len(y):.6f}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")

    ch = sum(1 for q in all_qids if q in base and q in cand
             and base[q][:10] != cand[q][:10])
    print(f"\n  top-10s changed: {ch:,} ({100*ch/len(all_qids):.1f}%)")


if __name__ == "__main__":
    main()
