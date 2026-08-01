"""Extension pass: bracket the v7synth recalibration optimum.

Pass 1 (grid_search_v7synth.py) put eps at the grid MIN (17/20 of the top configs)
and eps_hi at the grid MIN (18/20), i.e. the optimum was never bracketed - it lies
BELOW the searched range. T wanted the high end (2.0/2.3 in 16/20, 1.3 never). lam
was FLAT (0.60/0.45/0.50 all present in the top 20), so pass 1's lam=0.60 argmax is
fitted noise, not signal.

Mechanistic reading: v7synth orders better (+0.011..+0.016 pure-CE), so it needs
LESS lexical rescue - a narrower near-tie band. This pass tests that directly by
extending eps and eps_hi downward, including the degenerate no-widening case
(eps_hi == eps), which is the logical endpoint of "narrower is better".

Same pristine split (dev_recal2000). Output feeds a val + untouched-remainder
validation of ROBUST configs (chosen from consistent signals), not the raw argmax.
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

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
SUBMITTED = dict(T=1.6, lam=0.50, eps=0.020, eps_hi=0.080, conf_t=0.10)

TS = [2.0, 2.3, 2.6]
LAMS = [0.45, 0.50, 0.55, 0.60]
EPSS = [0.005, 0.010, 0.015]
EPS_HIS = [0.02, 0.03, 0.04, 0.06]
CONF_TS = [0.05, 0.10, 0.15]


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "dev_recal2000.txt"))

    results = []
    submitted = None
    for T in TS + ([SUBMITTED["T"]] if SUBMITTED["T"] not in TS else []):
        h = Harness(qids, T=T, cache_stem=V7)
        if T == SUBMITTED["T"]:
            submitted = h.evaluate(lam=SUBMITTED["lam"], eps=SUBMITTED["eps"],
                                   eps_hi=SUBMITTED["eps_hi"], conf_t=SUBMITTED["conf_t"],
                                   key="cont3")
            print(f"SUBMITTED config bar: {submitted:.4f}", flush=True)
            if T not in TS:
                del h
                continue
        for lam in LAMS:
            for eps in EPSS:
                for eps_hi in EPS_HIS:
                    if eps_hi < eps:
                        continue
                    for conf_t in CONF_TS:
                        s = h.evaluate(lam=lam, eps=eps, eps_hi=eps_hi,
                                       conf_t=conf_t, key="cont3")
                        results.append((s, T, lam, eps, eps_hi, conf_t))
            # degenerate: no adaptive widening at all (eps_hi == eps)
            for eps in EPSS:
                s = h.evaluate(lam=lam, eps=eps, eps_hi=eps, conf_t=0.10, key="cont3")
                results.append((s, T, lam, eps, eps, 0.10))
        print(f"  T={T} done ({len(results)} configs)", flush=True)
        del h

    results.sort(reverse=True)
    print(f"\n=== TOP 20 of {len(results)} (bar = submitted {submitted:.4f}) ===")
    print(f"{'nDCG':>8} {'d_sub':>8}  {'T':>4} {'lam':>5} {'eps':>6} {'eps_hi':>7} {'conf_t':>7}")
    for s, T, lam, eps, eps_hi, conf_t in results[:20]:
        print(f"{s:8.4f} {s - submitted:+8.4f}  {T:4.1f} {lam:5.2f} {eps:6.3f} {eps_hi:7.3f} {conf_t:7.2f}")

    s, T, lam, eps, eps_hi, conf_t = results[0]
    print(f"\nbest: {s:.4f} ({s - submitted:+.4f} vs submitted)")
    edges = []
    for name, val, grid in (("T", T, TS), ("lam", lam, LAMS), ("eps", eps, EPSS),
                            ("eps_hi", eps_hi, EPS_HIS), ("conf_t", conf_t, CONF_TS)):
        if val in (min(grid), max(grid)):
            edges.append(f"{name}={val}({'min' if val == min(grid) else 'MAX'})")
    print(f"boundary audit: {len(edges)}/5 at an edge" + (f" -> {', '.join(edges)}" if edges else ""))

    top = results[:20]
    print("\nconsistency across top-20 (the robust signal, vs the argmax):")
    for name, idx in (("T", 1), ("lam", 2), ("eps", 3), ("eps_hi", 4), ("conf_t", 5)):
        c = Counter(r[idx] for r in top).most_common()
        print(f"  {name:7s}: {c}")

    near = [r for r in results if r[0] >= s - 0.0005]
    print(f"plateau within 0.0005: {len(near)} configs")

    out = {"pass": "ext", "submitted_bar": submitted,
           "best": {"score": s, "T": T, "lam": lam, "eps": eps, "eps_hi": eps_hi,
                    "conf_t": conf_t},
           "n_at_edge": len(edges), "edges": edges, "plateau": len(near),
           "top20": [{"score": r[0], "T": r[1], "lam": r[2], "eps": r[3],
                      "eps_hi": r[4], "conf_t": r[5]} for r in top]}
    (ROOT / "outputs" / "v7synth_recal_ext.json").write_text(json.dumps(out, indent=2),
                                                             encoding="utf-8")
    print("\nwrote outputs/v7synth_recal_ext.json")


if __name__ == "__main__":
    main()
