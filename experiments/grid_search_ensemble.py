"""v6warm + v7synth logit-space ensemble, with the stack RE-TUNED for the mixture.

Four prior ensembles failed (zs+v2, v2+v3, v2+v3.1, v6warm+bge-FT), but all shared
two confounds this pass removes:
  1. each was judged under band parameters tuned for a SINGLE model, and an ensemble
     has its own score distribution - the exact error the 2026-07-26 recalibration
     proved material (it recovered +0.0013 -> +0.0028 on v7synth alone);
  2. each paired models with a large capability gap (0.03-0.14 pure-CE), which was
     the diagnosed failure mechanism. v6warm (0.605) vs v7synth (0.621) is a 0.016
     gap - the closest pairing ever tested, and the two diverge specifically at the
     synthetic-curriculum stage, so their errors should decorrelate without a
     capability penalty.

Fusion is a weighted average in LOGIT space with per-model temperatures (prior
finding: logit-mean loses less than sigmoid-mean; pure rank/percentile fusion is
strictly worse than magnitude-aware blending on this task):

    ce_ens = sigmoid( (1-w) * logit(s_v7)/T7  +  w * logit(s_v6)/T6 )

w=0 reduces EXACTLY to the current champion (v7synth T=2.6), which is the baseline
row in the grid and the bar to beat.

Fifth pristine split (dev_ens2000, seed 45), disjoint from every prior split.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import math
import random
from collections import Counter

import pandas as pd

from common import DATA, ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

V6 = "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16"
V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
T7, T6 = 2.6, 2.3   # each model's own validated temperature

WS = [0.0, 0.15, 0.25, 0.35, 0.50]
LAMS = [0.55, 0.60, 0.65]
EPSS = [0.005, 0.010, 0.015]
EPS_HIS = [0.030, 0.040, 0.050]
CONF_T = 0.15


def logit(s: float) -> float:
    s = min(max(s, 1e-12), 1.0 - 1e-12)
    return math.log(s / (1.0 - s))


def mint() -> list[str]:
    path = ROOT / "splits" / "dev_ens2000.txt"
    if not path.exists():
        dev_all = sorted(load_qid_file(ROOT / "splits" / "dev_qids.txt"))
        used = set()
        for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
                  "dev_recal2000.txt"):
            used |= load_qid_file(ROOT / "splits" / f)
        avail = sorted(q for q in dev_all if q not in used)
        random.seed(45)
        path.write_text("\n".join(random.sample(avail, 2000)) + "\n", encoding="utf-8")
        print(f"minted dev_ens2000.txt from {len(avail)} unused dev qids")
    return sorted(load_qid_file(path))


def load_raw(stem: str, qid_set: set[str]) -> dict:
    df = pd.read_parquet(DATA / "cache" / f"{stem}.parquet")
    df = df[df["qid"].isin(qid_set)]
    return {(q, i): s for q, i, s in zip(df["qid"], df["iid"], df["score"])}


def main():
    qids = mint()
    qid_set = set(qids)

    # one harness supplies pools / priors / cont3 / qrels; its ce is swapped per w
    h = Harness(qids, T=T7, cache_stem=V7)
    raw7 = load_raw(V7, qid_set)
    raw6 = load_raw(V6, qid_set)
    print(f"caches loaded; pools ready ({len(qids)} queries)", flush=True)

    # precompute per-model scaled logits once
    l7 = {k: logit(v) / T7 for k, v in raw7.items()}
    l6 = {k: logit(v) / T6 for k, v in raw6.items()}

    results = []
    for w in WS:
        # rebuild the ensemble CE and the confidence margins for this weight
        ce = {}
        for qid in qids:
            for iid in h.pools[qid]:
                k = (qid, iid)
                z = (1 - w) * l7.get(k, 0.0) + w * l6.get(k, 0.0)
                ce[k] = 1.0 / (1.0 + math.exp(-z))
        h.ce = ce
        h.conf = {}
        for qid in qids:
            v = sorted((ce[(qid, i)] for i in h.pools[qid]), reverse=True)
            h.conf[qid] = v[0] - v[4] if len(v) >= 5 else 1.0

        for lam in LAMS:
            for eps in EPSS:
                for eps_hi in EPS_HIS:
                    s = h.evaluate(lam=lam, eps=eps, eps_hi=eps_hi,
                                   conf_t=CONF_T, key="cont3")
                    results.append((s, w, lam, eps, eps_hi))
        print(f"  w={w} done ({len(results)} configs)", flush=True)

    results.sort(reverse=True)
    baseline = max(r[0] for r in results if r[1] == 0.0)  # best pure-v7synth config
    print(f"\nbaseline (w=0, pure v7synth, best stack): {baseline:.4f}")
    print(f"\n=== TOP 15 of {len(results)} ===")
    print(f"{'nDCG':>8} {'d_base':>8}  {'w':>5} {'lam':>5} {'eps':>6} {'eps_hi':>7}")
    for s, w, lam, eps, eps_hi in results[:15]:
        print(f"{s:8.4f} {s - baseline:+8.4f}  {w:5.2f} {lam:5.2f} {eps:6.3f} {eps_hi:7.3f}")

    s, w, lam, eps, eps_hi = results[0]
    print(f"\nbest: {s:.4f} at w={w} ({s - baseline:+.4f} vs pure v7synth)")
    print(f"top-15 w consensus: {Counter(r[1] for r in results[:15]).most_common()}")
    if w == 0.0:
        print("VERDICT: ensemble adds NOTHING - the best config is pure v7synth (w=0).")
    elif s - baseline < 0.0015:
        print(f"VERDICT: sub-gate (+{s - baseline:.4f} < +0.0015); not worth a val run.")
    else:
        print(f"VERDICT: clears the dev gate at w={w}; escalate to val + untouched.")

    (ROOT / "outputs" / "ensemble_grid.json").write_text(json.dumps(
        {"baseline_w0": baseline,
         "best": {"score": s, "w": w, "lam": lam, "eps": eps, "eps_hi": eps_hi},
         "delta": s - baseline,
         "top15": [{"score": r[0], "w": r[1], "lam": r[2], "eps": r[3], "eps_hi": r[4]}
                   for r in results[:15]]}, indent=2), encoding="utf-8")
    print("\nwrote outputs/ensemble_grid.json")


if __name__ == "__main__":
    main()
