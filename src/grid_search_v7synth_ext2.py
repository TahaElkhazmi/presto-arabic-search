"""Final targeted pass: bracket lam / T / conf_t, then STOP tuning.

Pass 2 bracketed the real finding (eps 0.020 -> 0.010, eps_hi 0.080 -> 0.040; both
interior, 12/20 and 16/20 of the top configs, mechanistically coherent with v7synth
being a better orderer that needs less lexical rescue). Three dims remained at grid
maximums: lam=0.60 (15/20 - now a strong signal, not the flat/noisy dimension it was
in pass 1), T=2.6, conf_t=0.15.

This pass extends only those three, holding eps/eps_hi near their bracketed optimum.
lam is pushed toward the pure-CE limit (lam=1.0 scores ~0.62, far worse), so an
interior optimum must exist somewhere below it.

DISCIPLINE: this is the LAST tuning pass on dev_recal2000. A fourth pass would be
fitting the split, not the model. Whatever this finds goes straight to val + the
untouched-remainder arbiter, using a ROBUST config from the top-15 consensus rather
than the raw argmax.
"""

import json
from collections import Counter

from common import ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"

TS = [2.3, 2.6, 3.0, 3.5]
LAMS = [0.60, 0.65, 0.70, 0.75, 0.80]
EPSS = [0.010]
EPS_HIS = [0.030, 0.040, 0.050]
CONF_TS = [0.15, 0.20, 0.25, 0.30]


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "dev_recal2000.txt"))
    results = []
    for T in TS:
        h = Harness(qids, T=T, cache_stem=V7)
        for lam in LAMS:
            for eps in EPSS:
                for eps_hi in EPS_HIS:
                    for conf_t in CONF_TS:
                        s = h.evaluate(lam=lam, eps=eps, eps_hi=eps_hi,
                                       conf_t=conf_t, key="cont3")
                        results.append((s, T, lam, eps, eps_hi, conf_t))
        print(f"  T={T} done ({len(results)} configs)", flush=True)
        del h

    results.sort(reverse=True)
    print(f"\n=== TOP 15 of {len(results)} (pass-2 best was 0.6652) ===")
    print(f"{'nDCG':>8}  {'T':>4} {'lam':>5} {'eps':>6} {'eps_hi':>7} {'conf_t':>7}")
    for s, T, lam, eps, eps_hi, conf_t in results[:15]:
        print(f"{s:8.4f}  {T:4.1f} {lam:5.2f} {eps:6.3f} {eps_hi:7.3f} {conf_t:7.2f}")

    s, T, lam, eps, eps_hi, conf_t = results[0]
    edges = []
    for name, val, grid in (("T", T, TS), ("lam", lam, LAMS),
                            ("eps_hi", eps_hi, EPS_HIS), ("conf_t", conf_t, CONF_TS)):
        if val in (min(grid), max(grid)):
            edges.append(f"{name}={val}({'min' if val == min(grid) else 'MAX'})")
    print(f"\nbest: {s:.4f}")
    print(f"boundary audit: {len(edges)}/4 at an edge" + (f" -> {', '.join(edges)}" if edges else ""))

    top = results[:15]
    print("\ntop-15 consensus (basis for the ROBUST config taken to val):")
    consensus = {}
    for name, idx in (("T", 1), ("lam", 2), ("eps", 3), ("eps_hi", 4), ("conf_t", 5)):
        c = Counter(r[idx] for r in top).most_common()
        consensus[name] = c[0][0]
        print(f"  {name:7s}: {c}   -> robust pick {c[0][0]}")

    out = {"pass": "ext2",
           "best": {"score": s, "T": T, "lam": lam, "eps": eps, "eps_hi": eps_hi,
                    "conf_t": conf_t},
           "robust_consensus": consensus,
           "n_at_edge": len(edges), "edges": edges,
           "top15": [{"score": r[0], "T": r[1], "lam": r[2], "eps": r[3],
                      "eps_hi": r[4], "conf_t": r[5]} for r in top]}
    (ROOT / "outputs" / "v7synth_recal_ext2.json").write_text(json.dumps(out, indent=2),
                                                              encoding="utf-8")
    print("\nwrote outputs/v7synth_recal_ext2.json")


if __name__ == "__main__":
    main()
