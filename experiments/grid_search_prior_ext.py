"""Bracketing pass for the prior composition: extend k upward, ratios upward, and
joint-check eps (changing k reshapes blend-score gaps, so the band widths tuned at
k=10 need re-verification at the winning composition).

Pass 1: k=60 won with 10/15 top-15 consensus but sat at the grid MAX; ratio
consensus 3:1-4:1. Same discipline as the stack recalibration: bracket every
dimension on both sides, then STOP and validate a robust consensus pick.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from collections import Counter

from common import ROOT, load_qid_file, setup_utf8
from grid_search_prior import set_prior
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"

KS = [40, 60, 80, 100, 150]
RATIOS = [(2.5, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0), (6.0, 1.0)]
LAMS = [0.55, 0.60, 0.65]
EPS_FIXED = dict(eps=0.010, eps_hi=0.040, conf_t=0.15, key="cont3")


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "dev_final2000.txt"))
    h = Harness(qids, T=2.6, cache_stem=V7)
    base = h.evaluate(lam=0.60, **EPS_FIXED)  # submitted composition k=10, 2:1
    print(f"BASELINE (k=10, 2:1): {base:.4f}\n", flush=True)

    results = []
    for k in KS:
        for w in RATIOS:
            set_prior(h, k, w)
            for lam in LAMS:
                s = h.evaluate(lam=lam, **EPS_FIXED)
                results.append((s, k, w, lam))
        print(f"  k={k} done", flush=True)

    results.sort(key=lambda r: -r[0])
    print(f"\n=== TOP 15 of {len(results)} ===")
    print(f"{'nDCG':>8} {'d_base':>8}  {'k':>4} {'ratio':>7} {'lam':>5}")
    for s, k, w, lam in results[:15]:
        print(f"{s:8.4f} {s - base:+8.4f}  {k:4d} {w[0]:4.1f}:1 {lam:5.2f}")
    print(f"top-15 k: {Counter(r[1] for r in results[:15]).most_common()}")
    print(f"top-15 ratio: {Counter(r[2][0] for r in results[:15]).most_common()}")
    print(f"top-15 lam: {Counter(r[3] for r in results[:15]).most_common()}")

    # joint eps re-check at the top composition
    s0, k0, w0, lam0 = results[0]
    set_prior(h, k0, w0)
    print(f"\neps joint check at k={k0}, {w0[0]}:1, lam={lam0}:")
    eps_results = []
    for eps in (0.005, 0.010, 0.015):
        for eps_hi in (0.02, 0.04, 0.06):
            s = h.evaluate(lam=lam0, eps=eps, eps_hi=eps_hi, conf_t=0.15, key="cont3")
            eps_results.append((s, eps, eps_hi))
            print(f"  eps={eps:.3f} hi={eps_hi:.2f}: {s:.4f} ({s - s0:+.4f})", flush=True)
    eps_results.sort(key=lambda r: -r[0])

    out = {"baseline_k10": base,
           "top15": [{"score": r[0], "k": r[1], "ratio": r[2][0], "lam": r[3]}
                     for r in results[:15]],
           "eps_check": [{"score": r[0], "eps": r[1], "eps_hi": r[2]} for r in eps_results]}
    (ROOT / "outputs" / "prior_grid_ext.json").write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")
    print("\nwrote outputs/prior_grid_ext.json")


if __name__ == "__main__":
    main()
