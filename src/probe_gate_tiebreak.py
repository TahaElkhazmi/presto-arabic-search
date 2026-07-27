"""Does GATE-AraBert add signal BEYOND our cross-encoder, and does it ship?

Established by probe_gate_arabert.py, inside the maximum-containment tie group of
zero-cohort queries (mean 73 items):
    low-id, the key we actually use   AUC 0.4955   (chance)
    GATE-AraBert                      AUC 0.5844
    our cross-encoder                 AUC 0.6422
On healthy queries low-id scores 0.6367 in the same position, so the incumbent prior
is informative in general and collapses to noise precisely where we depend on it.

Two questions remain, and only the second decides anything:
  1. Is GATE ORTHOGONAL to the CE, i.e. does CE+GATE beat CE alone inside the group?
     A feature that merely re-states the CE is worthless no matter how good its AUC.
  2. Does using it as the in-group tie-break actually raise nDCG@10 end to end?
     AUC is a pairwise measure over all positions; nDCG@10 only cares about the head,
     and this project has repeatedly seen strong signals absorb to nothing.

Protocol: the clean 8,012-query holdout is split in half by seed. The first half
selects the blend weight, the second half confirms it. Nothing is tuned on the half
it is reported on.
"""

import argparse
import bisect
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


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    ns = sorted(neg)
    conc = ties = 0
    for p in pos:
        conc += bisect.bisect_left(ns, p)
        ties += bisect.bisect_right(ns, p) - bisect.bisect_left(ns, p)
    return (conc + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="0,0.25,0.5,1,2,4")
    args = ap.parse_args()

    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    rng = random.Random(77)
    shuf = list(qids)
    rng.shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)
    print(f"holdout {len(qids)} -> tune {len(tune)}, confirm {len(conf)}\n")

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    C = np.load(DATA / "cache" / "gate_corpus.npy")
    Q = np.load(DATA / "cache" / "gate_queries.npy")
    item_ids = [str(r["item_id"]) for r in load_jsonl(BENCH / "corpus.jsonl")]
    idx = {iid: i for i, iid in enumerate(item_ids)}
    all_q = [str(x["query_id"]) for x in load_jsonl(BENCH / "queries.jsonl")]
    qidx = {q: i for i, q in enumerate(all_q)}

    def gate(q, iid):
        return float(Q[qidx[q]] @ C[idx[iid]])

    # ---- Q1: orthogonality inside the tie group ----
    print("Q1. AUC INSIDE the max-containment tie group (tune half)")
    combos = {"CE alone": lambda q, i: h.ce[(q, i)],
              "GATE alone": gate}
    for w in (0.25, 0.5, 1.0, 2.0):
        combos[f"CE + {w}*GATE"] = (lambda w: lambda q, i: h.ce[(q, i)] + w * gate(q, i))(w)
    for name, fn in combos.items():
        pos, neg = [], []
        for q in tune:
            pool = h.pools[q]
            cs = [h.cont3(q, i) for i in pool]
            mx = max(cs)
            grp = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
            if len(grp) < 2:
                continue
            gp = [fn(q, i) for i in grp if h.qrels[q].get(i, 0) > 0]
            gn = [fn(q, i) for i in grp if h.qrels[q].get(i, 0) <= 0]
            if gp and gn:
                pos.extend(gp)
                neg.extend(gn)
        print(f"    {name:<18s} AUC {auc(pos, neg):.4f}")

    # ---- Q2: end to end ----
    def mean_nd(run, qs):
        v = [ndcg_at_k(run[q], h.qrels[q], 10) for q in qs if q in run]
        return sum(v) / len(v)

    base_run = h.build_run(**LIVE_STACK)
    print(f"\nQ2. END TO END")
    print(f"    live baseline      tune {mean_nd(base_run, tune):.6f}   "
          f"confirm {mean_nd(base_run, conf):.6f}")

    # cont3 dominant, secondary signal breaks the tie inside equal-cont3 groups
    def key_with(sec, w):
        return lambda q, i: h.cont3(q, i) * 1000.0 + w * sec(q, i)

    weights = [float(x) for x in args.weights.split(",")]
    best, best_w = None, None
    print(f"\n    {'in-group tie-break':<26s}{'tune':>11s}{'delta':>11s}")
    for w in weights:
        cfg = dict(LIVE_STACK)
        cfg["key"] = key_with(gate, w) if w > 0 else "cont3"
        s = mean_nd(h.build_run(**cfg), tune)
        d = s - mean_nd(base_run, tune)
        lab = "cont3 + id (live)" if w == 0 else f"cont3 + {w}*GATE"
        print(f"    {lab:<26s}{s:>11.6f}{d:>+11.6f}")
        if w > 0 and (best is None or s > best):
            best, best_w = s, w

    cfg = dict(LIVE_STACK)
    cfg["key"] = key_with(lambda q, i: h.ce[(q, i)], 1.0)
    s_ce = mean_nd(h.build_run(**cfg), tune)
    print(f"    {'cont3 + CE (reference)':<26s}{s_ce:>11.6f}"
          f"{s_ce - mean_nd(base_run, tune):>+11.6f}")

    print(f"\n    CONFIRM at the tune-selected weight {best_w} "
          f"(never fitted on this half):")
    cfg = dict(LIVE_STACK)
    cfg["key"] = key_with(gate, best_w)
    run = h.build_run(**cfg)
    b, c = mean_nd(base_run, conf), mean_nd(run, conf)
    d = [ndcg_at_k(run[q], h.qrels[q], 10) - ndcg_at_k(base_run[q], h.qrels[q], 10)
         for q in conf]
    import statistics as st
    se = st.pstdev(d) / len(d) ** 0.5
    m = sum(d) / len(d)
    print(f"      live {b:.6f} -> GATE {c:.6f}   delta {m:+.6f}  "
          f"se {se:.6f}  t {m / se if se else 0:+.2f}")
    print(f"      {sum(1 for x in d if x > 1e-9)} better, "
          f"{sum(1 for x in d if x < -1e-9)} worse, "
          f"{sum(1 for x in d if abs(x) <= 1e-9)} unchanged")
    print(f"\n    gate to clear: +0.0015 full-benchmark")


if __name__ == "__main__":
    main()
