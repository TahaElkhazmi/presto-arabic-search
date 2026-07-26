"""Full recalibration of the post-processing stack against v7synth's own distribution.

The gate search only probed lam in {0.40,0.45,0.50} with eps/eps-hi/conf-t FROZEN at
v6warm's tuned values - and v7synth peaked at lam=0.50, the TOP of that grid, i.e. a
boundary, not an optimum. v7synth is a better orderer (+0.011..+0.016 pure-CE) and
wants to be trusted more, so its bands were never tuned for it.

Protocol (project discipline):
  - a FOURTH pristine dev split (dev_recal2000, seed 44, zero overlap with
    dev_rand2000 / dev_unseen2000 / dev_probe2000);
  - joint grid over T, lam, eps, eps_hi, conf_t;
  - report the top-N configs, not just the argmax, so a broad plateau can be told
    apart from a needle (the documented false-peak signature);
  - flag boundary-hugging winners explicitly BEFORE any val run (121-config wave-2,
    hi=0.08, and the v6warm retune all hugged edges and failed val).
Val + full-benchmark confirmation happen in a separate step, only if the dev winner
clears the bar and is not a boundary artifact.
"""

import json
import random
import sys
from pathlib import Path

from common import ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
V6 = "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16"

# submitted operating point (the bar to beat)
LIVE = dict(T=1.6, lam=0.50, eps=0.020, eps_hi=0.080, conf_t=0.10)

TS = [1.3, 1.6, 2.0, 2.3]
LAMS = [0.45, 0.50, 0.55, 0.60]          # extends ABOVE the old 0.50 boundary
EPSS = [0.015, 0.020, 0.025]
EPS_HIS = [0.06, 0.08, 0.10, 0.12]        # extends above the old 0.08
CONF_TS = [0.05, 0.10, 0.15]


def mint_split() -> list[str]:
    path = ROOT / "splits" / "dev_recal2000.txt"
    if not path.exists():
        dev_all = sorted(load_qid_file(ROOT / "splits" / "dev_qids.txt"))
        used = set()
        for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt"):
            used |= load_qid_file(ROOT / "splits" / f)
        avail = sorted(q for q in dev_all if q not in used)
        random.seed(44)
        picked = random.sample(avail, 2000)
        path.write_text("\n".join(picked) + "\n", encoding="utf-8")
        print(f"minted dev_recal2000.txt from {len(avail)} untouched dev qids")
    return sorted(load_qid_file(path))


def main():
    qids = mint_split()

    # reference bars on this fresh split
    h6 = Harness(qids, T=2.3, cache_stem=V6)
    v6_live = h6.evaluate(lam=0.45, eps=0.020, eps_hi=0.080, conf_t=0.10, key="cont3")
    del h6
    print(f"v6warm live (T=2.3,lam=0.45)       : {v6_live:.4f}")

    results = []
    submitted = None
    for T in TS:
        h = Harness(qids, T=T, cache_stem=V7)
        if T == LIVE["T"]:
            submitted = h.evaluate(lam=LIVE["lam"], eps=LIVE["eps"], eps_hi=LIVE["eps_hi"],
                                   conf_t=LIVE["conf_t"], key="cont3")
            print(f"v7synth SUBMITTED config (T={T},lam={LIVE['lam']}): {submitted:.4f}", flush=True)
        for lam in LAMS:
            for eps in EPSS:
                for eps_hi in EPS_HIS:
                    for conf_t in CONF_TS:
                        s = h.evaluate(lam=lam, eps=eps, eps_hi=eps_hi,
                                       conf_t=conf_t, key="cont3")
                        results.append((s, T, lam, eps, eps_hi, conf_t))
        print(f"  T={T} done ({len(results)} configs so far)", flush=True)
        del h

    results.sort(reverse=True)
    print(f"\n=== TOP 20 of {len(results)} configs (fresh split, bar = submitted {submitted:.4f}) ===")
    print(f"{'nDCG':>8} {'d_sub':>8}  {'T':>4} {'lam':>5} {'eps':>6} {'eps_hi':>7} {'conf_t':>7}")
    for s, T, lam, eps, eps_hi, conf_t in results[:20]:
        print(f"{s:8.4f} {s - submitted:+8.4f}  {T:4.1f} {lam:5.2f} {eps:6.3f} {eps_hi:7.2f} {conf_t:7.2f}")

    best = results[0]
    s, T, lam, eps, eps_hi, conf_t = best
    print(f"\nbest: {s:.4f} ({s - submitted:+.4f} vs submitted, {s - v6_live:+.4f} vs v6warm)")

    # boundary-hugging audit (the documented overfitting signature)
    edges = []
    for name, val, grid in (("T", T, TS), ("lam", lam, LAMS), ("eps", eps, EPSS),
                            ("eps_hi", eps_hi, EPS_HIS), ("conf_t", conf_t, CONF_TS)):
        if val in (min(grid), max(grid)):
            edges.append(f"{name}={val} ({'min' if val == min(grid) else 'MAX'})")
    print(f"boundary audit: {len(edges)}/5 dims at an edge" + (f" -> {', '.join(edges)}" if edges else ""))

    # plateau check: how many configs are within 0.0005 of the peak, and their spread
    near = [r for r in results if r[0] >= s - 0.0005]
    print(f"plateau: {len(near)} configs within 0.0005 of peak")
    for name, idx in (("T", 1), ("lam", 2), ("eps", 3), ("eps_hi", 4), ("conf_t", 5)):
        vals = sorted({r[idx] for r in near})
        print(f"  {name}: {vals}")

    out = {"split": "dev_recal2000", "submitted_config_score": submitted,
           "v6warm_live": v6_live,
           "best": {"score": s, "T": T, "lam": lam, "eps": eps,
                    "eps_hi": eps_hi, "conf_t": conf_t},
           "delta_vs_submitted": s - submitted,
           "n_at_edge": len(edges), "edges": edges,
           "plateau_size": len(near),
           "top20": [{"score": r[0], "T": r[1], "lam": r[2], "eps": r[3],
                      "eps_hi": r[4], "conf_t": r[5]} for r in results[:20]]}
    (ROOT / "outputs" / "v7synth_recal.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote outputs/v7synth_recal.json")


if __name__ == "__main__":
    main()
