"""Head-to-head screen: v8graded vs v7synth, each at its OWN optimal stack.

v8graded improved pure-CE ordering (+0.0050). Whether that survives end to end
depends on the stack, and v8graded was trained on GRADED targets so its score
distribution should differ in shape from every previous model - temperature and band
widths fitted to binary-trained models may be badly sized for it. That is the same
era-mismatch that recovered +0.0028 for v7synth on 2026-07-26.

This screen runs the identical grid for BOTH models on dev_rand2000 (both caches
already exist, so it is CPU-only and free) and compares their best configurations.
dev_rand2000 is tuning-contaminated, so the ABSOLUTE numbers are inflated - but both
models are inflated by the same instrument, so the COMPARISON is informative. It is a
screen, not a verdict: a promising result escalates to a pristine split plus val and
the untouched remainder.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
from collections import Counter

from common import ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

MODELS = {
    "v7synth ": "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16",
    "v8graded": "rerank_qwen3-reranker-arabic-ecom-rerank_v8graded_len160_bfloat16",
}
TS = [1.6, 2.0, 2.6, 3.2, 4.0]
LAMS = [0.45, 0.55, 0.60, 0.65]
EPSS = [0.005, 0.010, 0.015]
EPS_HIS = [0.030, 0.040, 0.060]
CONF_T = 0.15


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "dev_rand2000.txt"))
    out = {}
    for name, stem in MODELS.items():
        results = []
        for T in TS:
            h = Harness(qids, T=T, cache_stem=stem, weights=[3.0, 1.0], rrf_k=60)
            if h.missing_ce:
                print(f"WARNING {name}: {h.missing_ce} pool pairs missing from cache")
            for lam in LAMS:
                for eps in EPSS:
                    for eps_hi in EPS_HIS:
                        s = h.evaluate(lam=lam, eps=eps, eps_hi=eps_hi,
                                       conf_t=CONF_T, key="cont3")
                        results.append((s, T, lam, eps, eps_hi))
            del h
        results.sort(key=lambda r: -r[0])
        best = results[0]
        out[name] = {"best": best, "top10": results[:10]}
        print(f"\n=== {name} (dev_rand2000, own optimal stack) ===")
        print(f"{'nDCG':>8}  {'T':>4} {'lam':>5} {'eps':>6} {'eps_hi':>7}")
        for s, T, lam, eps, eps_hi in results[:6]:
            print(f"{s:8.4f}  {T:4.1f} {lam:5.2f} {eps:6.3f} {eps_hi:7.3f}")
        print(f"top-10 T consensus: {Counter(r[1] for r in results[:10]).most_common()}")

    b7, b8 = out["v7synth "]["best"], out["v8graded"]["best"]
    print(f"\n{'=' * 62}")
    print(f"v7synth  best: {b7[0]:.4f}  (T={b7[1]}, lam={b7[2]}, eps={b7[3]}, hi={b7[4]})")
    print(f"v8graded best: {b8[0]:.4f}  (T={b8[1]}, lam={b8[2]}, eps={b8[3]}, hi={b8[4]})")
    d = b8[0] - b7[0]
    print(f"delta        : {d:+.4f}")
    if d >= 0.0015:
        print("SCREEN PASSED -> escalate: pristine-split recalibration, then val + untouched.")
    elif d > 0:
        print("SCREEN: positive but sub-gate on an inflated instrument; escalation is")
        print("        unlikely to clear +0.0015 on the untouched remainder.")
    else:
        print("SCREEN FAILED -> the pure-CE gain does not survive the stack; close it.")

    (ROOT / "outputs" / "screen_v8_vs_v7.json").write_text(
        json.dumps({k: {"best": list(v["best"])} for k, v in out.items()}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
