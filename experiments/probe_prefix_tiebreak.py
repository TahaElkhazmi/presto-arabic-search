"""Does "title starts with the query" survive end to end as an in-group tie-break?

Inside the saturated tie groups of the zero cohort it is the strongest signal we have
found there (per-query AUC 0.6008, against the incumbent low-id 0.5857 and our own
cross-encoder 0.5925), and unlike everything previously tried it is FREE: pure text,
zero GPU, computed from the title alone.

Why it might succeed where GATE-AraBert failed at the same injection point. GATE was
continuous, so inside an equal-containment group it fully displaced the ascending-id
ordering and cost -0.0306. A BINARY feature instead PARTITIONS the group into
"title begins with the query" and "does not", and the low-id prior still orders each
partition. The incumbent signal is preserved rather than overwritten, which is exactly
the property the -0.0105 replacement-cascade lesson says to respect.

Two things must both hold before this is shippable:
  1. it must not be INVERTED on healthy queries, the way id-contiguity was
     (0.5811 on zeros but 0.3586 on healthy, i.e. actively wrong for 91.9% of the
     benchmark);
  2. it must clear +0.0015 end to end on a half of the holdout it was never fitted on.

Threshold-free: there is nothing to tune except whether the bonus is applied at all,
which keeps dev-argmax inflation near zero.
"""

import bisect
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def per_query_auc(pairs):
    conc = disc = ties = 0
    for pos, neg in pairs:
        if not pos or not neg:
            continue
        ns = sorted(neg)
        for p in pos:
            lo, hi = bisect.bisect_left(ns, p), bisect.bisect_right(ns, p)
            conc += lo
            ties += hi - lo
            disc += len(ns) - hi
    tot = conc + disc + ties
    return (conc + 0.5 * ties) / tot if tot else float("nan")


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names = {str(r["item_id"]): r["name"] for r in corpus}
    norm = {i: normalize_arabic(n) for i, n in names.items()}

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    qstr = {}
    for q in h.qids:
        toks = [t for t in normalize_arabic(h.queries[q]).split()
                if t not in DIALECT_FILLERS]
        qstr[q] = " ".join(toks) if toks else normalize_arabic(h.queries[q])

    def prefix(q, iid):
        return 1.0 if norm[iid].startswith(qstr[q]) else 0.0

    # ---- 1. is it inverted on healthy queries? ----
    qrels = h.qrels
    base_run = h.build_run(**LIVE_STACK)
    nd = {q: ndcg_at_k(base_run[q], qrels[q], 10) for q in h.qids if q in qrels}
    zc = [q for q, v in nd.items() if v == 0.0]
    hc = [q for q, v in nd.items() if v >= 0.9]

    print("PER-QUERY AUC of 'title starts with query' inside the saturated tie group")
    for label, qs in (("zero cohort", zc), ("healthy queries", hc)):
        pairs = []
        for q in qs:
            pool = h.pools[q]
            cs = [h.cont3(q, i) for i in pool]
            mx = max(cs)
            g = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
            if len(g) < 2:
                continue
            p = [prefix(q, i) for i in g if qrels[q].get(i, 0) > 0]
            n = [prefix(q, i) for i in g if qrels[q].get(i, 0) <= 0]
            if p and n:
                pairs.append((p, n))
        print(f"  {label:<18s} n={len(pairs):<5d} AUC {per_query_auc(pairs):.4f}")
    print("  (an AUC below 0.5 on healthy queries would make this unusable globally)\n")

    # ---- 2. end to end ----
    def mean_nd(run, qs):
        v = [ndcg_at_k(run[q], qrels[q], 10) for q in qs if q in run]
        return sum(v) / len(v)

    b_tune, b_conf = mean_nd(base_run, tune), mean_nd(base_run, conf)
    print(f"live baseline: tune {b_tune:.6f}   confirm {b_conf:.6f}\n")

    variants = {
        "cont3 + prefix, then id": lambda q, i: h.cont3(q, i) * 1000.0 + prefix(q, i),
        "prefix + cont3, then id": lambda q, i: prefix(q, i) * 1000.0 + h.cont3(q, i),
    }
    print(f"  {'variant':<28s}{'tune':>11s}{'delta':>11s}")
    best, best_name, best_fn = None, None, None
    for name, keyf in variants.items():
        cfg = dict(LIVE_STACK)
        cfg["key"] = keyf
        s = mean_nd(h.build_run(**cfg), tune)
        print(f"  {name:<28s}{s:>11.6f}{s - b_tune:>+11.6f}")
        if best is None or s > best:
            best, best_name, best_fn = s, name, keyf

    print(f"\nCONFIRM the tune-best variant ({best_name}) on the untouched half:")
    cfg = dict(LIVE_STACK)
    cfg["key"] = best_fn
    run = h.build_run(**cfg)
    d = [ndcg_at_k(run[q], qrels[q], 10) - ndcg_at_k(base_run[q], qrels[q], 10)
         for q in conf]
    m = sum(d) / len(d)
    se = st.pstdev(d) / len(d) ** 0.5
    print(f"  live {b_conf:.6f} -> {mean_nd(run, conf):.6f}")
    print(f"  delta {m:+.6f}   se {se:.6f}   t {m / se if se else 0:+.2f}")
    print(f"  {sum(1 for x in d if x > 1e-9)} better, "
          f"{sum(1 for x in d if x < -1e-9)} worse, "
          f"{sum(1 for x in d if abs(x) <= 1e-9)} unchanged")
    print(f"\n  gate to clear: +0.0015")


if __name__ == "__main__":
    main()
