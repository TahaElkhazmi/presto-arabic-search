"""Prior-composition retune for the v7synth-recal era (the last untuned component).

rrf_k=10 and lexical:dense = 2:1 were fixed in PHASE 1, tuned for a fusion whose
consumer was a weak reranker at lam=0.30 with wide bands. Today the CE is two
generations better, lam=0.60 (prior weight 0.40), and the bands are halved - the
same situation whose stack-analog paid +0.0028 this morning. A composition retune
was dev-flat in the v2 era (2026-07-19 coordinate ascent); it has never been
measured in this era.

Pools are the union of top-100s and do NOT depend on (k, weights), so the score
cache fully covers every variant; priors are recomputed in-place per variant.
Sixth pristine split (dev_final2000, seed 46). Bar = the submitted composition
(k=10, 2:1) under the submitted stack (T=2.6, lam=0.60, eps=0.010/0.040, conf_t=0.15).
"""

import json
import random
from collections import Counter

from common import ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
STACK = dict(eps=0.010, eps_hi=0.040, conf_t=0.15, key="cont3")

KS = [5, 10, 20, 40, 60]
WEIGHTS = [(2.0, 1.0), (1.5, 1.0), (2.5, 1.0), (3.0, 1.0), (1.0, 1.0), (2.0, 0.5), (4.0, 1.0)]
LAMS = [0.55, 0.60, 0.65]


def mint() -> list[str]:
    path = ROOT / "splits" / "dev_final2000.txt"
    if not path.exists():
        dev_all = sorted(load_qid_file(ROOT / "splits" / "dev_qids.txt"))
        used = set()
        for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
                  "dev_recal2000.txt", "dev_ens2000.txt"):
            used |= load_qid_file(ROOT / "splits" / f)
        avail = sorted(q for q in dev_all if q not in used)
        random.seed(46)
        path.write_text("\n".join(random.sample(avail, 2000)) + "\n", encoding="utf-8")
        print(f"minted dev_final2000.txt from {len(avail)} unused dev qids")
    return sorted(load_qid_file(path))


def set_prior(h: Harness, k: int, w: tuple[float, float]):
    h.rrf_k = k
    h.weights = list(w)
    priors = {}
    for qid in h.qids:
        seen = {iid: 0.0 for iid in h.pools[qid]}
        for run, wt in zip([h.run_bm25, h.run_dense], w):
            for rank, iid in enumerate(run.get(qid, [])[:100], start=1):
                seen[iid] += wt / (k + rank)
        priors[qid] = seen
    h.priors = priors


def main():
    qids = mint()
    h = Harness(qids, T=2.6, cache_stem=V7)
    print(f"harness ready, missing_ce={h.missing_ce}", flush=True)

    baseline = h.evaluate(lam=0.60, **STACK)
    print(f"BASELINE (k=10, 2:1, submitted stack): {baseline:.4f}\n", flush=True)

    results = []
    for k in KS:
        for w in WEIGHTS:
            set_prior(h, k, w)
            for lam in LAMS:
                s = h.evaluate(lam=lam, **STACK)
                results.append((s, k, w, lam))
        print(f"  k={k} done", flush=True)

    results.sort(key=lambda r: -r[0])
    print(f"\n=== TOP 15 of {len(results)} (bar = {baseline:.4f}) ===")
    print(f"{'nDCG':>8} {'d_base':>8}  {'k':>3} {'weights':>9} {'lam':>5}")
    for s, k, w, lam in results[:15]:
        print(f"{s:8.4f} {s - baseline:+8.4f}  {k:3d} {w[0]:4.1f}:{w[1]:<4.1f} {lam:5.2f}")

    s, k, w, lam = results[0]
    print(f"\nbest: {s:.4f} ({s - baseline:+.4f}) at k={k}, {w[0]}:{w[1]}, lam={lam}")
    print(f"top-15 k consensus: {Counter(r[1] for r in results[:15]).most_common()}")
    print(f"top-15 w consensus: {Counter(r[2] for r in results[:15]).most_common()}")
    verdict = ("clears dev gate; escalate to val + untouched" if s - baseline >= 0.0015
               else "sub-gate; composition is already optimal in this era too")
    print(f"VERDICT: {verdict}")

    (ROOT / "outputs" / "prior_grid.json").write_text(json.dumps(
        {"baseline": baseline,
         "best": {"score": s, "k": k, "weights": list(w), "lam": lam},
         "delta": s - baseline,
         "top15": [{"score": r[0], "k": r[1], "weights": list(r[2]), "lam": r[3]}
                   for r in results[:15]]}, indent=2), encoding="utf-8")
    print("wrote outputs/prior_grid.json")


if __name__ == "__main__":
    main()
