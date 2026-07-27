"""Final GATE test: apply the tie-break ONLY where the incumbent key is worthless.

Ungated, replacing the low-id in-group key with GATE costs -0.031: low-id scores AUC
0.637 inside tie groups on healthy queries and only 0.496 on the zero cohort, so a
global swap wrecks the majority to help a minority.

The targeted version applies GATE only to queries whose pool is heavily saturated,
which is where low-id provably carries no information. This is the one treatment in
the detect-and-treat family that was never testable before, because every earlier
candidate was a recombination of signals we already owned and GATE is genuinely new
information from a different model lineage.

The prior is still poor: a 240-cell sweep of that family found break-even precision
around 49% against 41% achievable, because recovery on true zeros converges to
0.11-0.14 for every treatment (the items are exchangeable) while false positives cost
-0.09 each. GATE has to beat that arithmetic, not merely be above chance.

Threshold selected on the tune half, reported on the confirm half.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import BENCH, DATA, ROOT, load_jsonl, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
SATS = (0.15, 0.25, 0.35, 0.50)


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    C = np.load(DATA / "cache" / "gate_corpus.npy")
    Q = np.load(DATA / "cache" / "gate_queries.npy")
    idx = {str(r["item_id"]): i
           for i, r in enumerate(load_jsonl(BENCH / "corpus.jsonl"))}
    qidx = {str(x["query_id"]): i
            for i, x in enumerate(load_jsonl(BENCH / "queries.jsonl"))}

    def gate(q, iid):
        return float(Q[qidx[q]] @ C[idx[iid]])

    # saturation fraction: share of the pool tied at maximum containment
    sat = {}
    for q in h.qids:
        cs = [h.cont3(q, i) for i in h.pools[q]]
        mx = max(cs)
        sat[q] = sum(1 for c in cs if c >= mx - 1e-12) / len(cs)

    base = h.build_run(**LIVE_STACK)

    def mean_nd(run, qs):
        v = [ndcg_at_k(run[q], h.qrels[q], 10) for q in qs if q in run]
        return sum(v) / len(v)

    def gated_run(thr, qs):
        """GATE breaks in-group ties only for saturated queries; live elsewhere."""
        hot = [q for q in qs if sat[q] >= thr]
        if not hot:
            return dict(base), 0
        cfg = dict(LIVE_STACK)
        cfg["key"] = lambda q, i: h.cont3(q, i) * 1000.0 + gate(q, i)
        treated = h.build_run(qids=hot, **cfg)
        out = {q: (treated[q] if q in treated else base[q]) for q in qs}
        return out, len(hot)

    print(f"tune {len(tune)}, confirm {len(conf)}")
    print(f"live baseline: tune {mean_nd(base, tune):.6f}  "
          f"confirm {mean_nd(base, conf):.6f}\n")
    print(f"  {'sat >=':>8s}{'treated':>10s}{'tune nDCG':>12s}{'delta':>11s}")
    best, best_thr = None, None
    for thr in SATS:
        run, n = gated_run(thr, tune)
        s = mean_nd(run, tune)
        d = s - mean_nd(base, tune)
        print(f"  {thr:>8.2f}{n:>10d}{s:>12.6f}{d:>+11.6f}")
        if best is None or s > best:
            best, best_thr = s, thr

    print(f"\nCONFIRM at sat >= {best_thr} (not fitted on this half):")
    run, n = gated_run(best_thr, conf)
    d = [ndcg_at_k(run[q], h.qrels[q], 10) - ndcg_at_k(base[q], h.qrels[q], 10)
         for q in conf]
    m = sum(d) / len(d)
    se = st.pstdev(d) / len(d) ** 0.5
    print(f"  treated {n}/{len(conf)} queries")
    print(f"  delta {m:+.6f}   se {se:.6f}   t {m / se if se else 0:+.2f}")
    print(f"  {sum(1 for x in d if x > 1e-9)} better, "
          f"{sum(1 for x in d if x < -1e-9)} worse, "
          f"{sum(1 for x in d if abs(x) <= 1e-9)} unchanged")
    print(f"\n  gate to clear: +0.0015")


if __name__ == "__main__":
    main()
