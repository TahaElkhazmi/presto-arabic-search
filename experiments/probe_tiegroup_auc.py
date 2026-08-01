"""Corrected within-tie-group AUC for every candidate signal.

METHODOLOGY FIX. Earlier passes (probe_gate_arabert.py, probe_id_structure.py) pooled
positives and negatives across ALL queries and computed one global AUC. That compares
an item from query A against an item from query B, which is meaningless for a ranking
feature, and it inflates or deflates scores depending on how comparable a feature's
values are across queries. The tell was that two provably equivalent orderings
(absolute low-id, and rank-by-id within the same group) scored 0.5121 and 0.6464:
identical rankings cannot have different AUCs, so the estimator was wrong.

Correct estimator: for each query, count concordant/discordant pairs ONLY between that
query's own positives and negatives inside its tie group, then aggregate. Equivalent
orderings now provably coincide, which is the built-in sanity check.

Everything is measured inside the maximum-containment tie group of zero-cohort
queries, because that is the exact decision our pipeline makes blind.
"""

import bisect
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import BENCH, DATA, ROOT, load_jsonl, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def per_query_auc(pairs):
    """pairs: list of (pos_scores, neg_scores) per query -> pooled-pair AUC."""
    conc = disc = ties = 0
    for pos, neg in pairs:
        if not pos or not neg:
            continue
        ns = sorted(neg)
        for p in pos:
            lo = bisect.bisect_left(ns, p)
            hi = bisect.bisect_right(ns, p)
            conc += lo
            ties += hi - lo
            disc += len(ns) - hi
    tot = conc + disc + ties
    return (conc + 0.5 * ties) / tot if tot else float("nan")


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}

    import random
    rng = random.Random(31)
    zeros = sorted(rng.sample([q for q, v in nd.items() if v == 0.0], 1500), key=int)
    goods = sorted(rng.sample([q for q, v in nd.items() if v >= 0.9], 1500), key=int)
    qids = sorted(set(zeros) | set(goods), key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names = {str(r["item_id"]): r["name"] for r in corpus}
    order = [str(r["item_id"]) for r in corpus]
    pos_in_file = {iid: i for i, iid in enumerate(order)}
    dup = defaultdict(list)
    for iid in order:
        dup[normalize_arabic(names[iid])].append(iid)
    dup_rank, dup_size = {}, {}
    for nm, ids in dup.items():
        for r, iid in enumerate(sorted(ids, key=num)):
            dup_rank[iid] = r
            dup_size[iid] = len(ids)

    print("building pools (0 GPU) ...", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    C = np.load(DATA / "cache" / "gate_corpus.npy")
    Q = np.load(DATA / "cache" / "gate_queries.npy")
    cidx = {str(r["item_id"]): i for i, r in enumerate(corpus)}
    qidx = {str(x["query_id"]): i
            for i, x in enumerate(load_jsonl(BENCH / "queries.jsonl"))}

    feats = {
        "low-id (incumbent)": lambda q, i, g: -num(i),
        "rank-by-id in group": lambda q, i, g: -sorted(g, key=num).index(i),
        "GATE-AraBert": lambda q, i, g: float(Q[qidx[q]] @ C[cidx[i]]),
        "our CE": lambda q, i, g: h.ce[(q, i)],
        "id contiguity": lambda q, i, g: (
            -st.mean(abs(num(i) - num(x)) for x in g if x != i) if len(g) > 1 else 0.0),
        "file-order proximity": lambda q, i, g: (
            -st.mean(abs(pos_in_file[i] - pos_in_file[x]) for x in g if x != i)
            if len(g) > 1 else 0.0),
        "rank among duplicates": lambda q, i, g: -dup_rank.get(i, 0),
        "duplicate-group size": lambda q, i, g: dup_size.get(i, 1),
    }

    for label, qs in (("ZERO cohort", zeros), ("high-scoring", goods)):
        groups = {}
        for q in qs:
            pool = h.pools[q]
            cs = [h.cont3(q, i) for i in pool]
            mx = max(cs)
            g = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
            if len(g) >= 2 and any(qrels[q].get(i, 0) > 0 for i in g) \
               and any(qrels[q].get(i, 0) <= 0 for i in g):
                groups[q] = g
        print(f"\n########## {label}: {len(groups)} usable tie groups, "
              f"mean size {st.mean(len(g) for g in groups.values()):.1f} ##########")
        for name, fn in feats.items():
            pairs = []
            for q, g in groups.items():
                p = [fn(q, i, g) for i in g if qrels[q].get(i, 0) > 0]
                n = [fn(q, i, g) for i in g if qrels[q].get(i, 0) <= 0]
                pairs.append((p, n))
            print(f"  {name:<24s} AUC {per_query_auc(pairs):.4f}")

    print("\nSANITY: 'low-id' and 'rank-by-id in group' are the SAME ordering within a")
    print("group, so their AUCs must be identical. If they are, the estimator is sound.")


if __name__ == "__main__":
    main()
