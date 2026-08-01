"""The last untested structural hypothesis: does the CORPUS ID SPACE leak relevance?

Everything we can compute from TEXT is tied inside a saturated group. The only
remaining per-item information is the product id itself and where the item sits in
the corpus file. The plain low-id prior is already used and scores exactly chance
(AUC 0.4955) inside those groups, but "lowest id wins" is only one of several ways id
could carry signal. An audit pass asserted several such structures; it had no way to
compute anything, so this measures them properly instead.

Hypotheses tested, all inside the maximum-containment tie group where we are blind:
  H1 CONTIGUITY   - relevant items cluster together in id space, so an item's
                    proximity to other pool members predicts relevance.
  H2 GROUP RANK   - what matters is rank-within-the-tie-group rather than absolute
                    id (the incumbent uses absolute order after a global sort).
  H3 ISOLATION    - relevant items sit in dense catalog neighbourhoods of similar
                    products, spurious matches are id-space outliers.
  H4 BLOCK        - the catalog was ingested in per-source blocks, so the relevant
                    item is the one in the same block as the query's other matches.
  H5 DUP GROUP    - within an exact-duplicate-name group, position among the
                    duplicates predicts relevance beyond taking the minimum id.

Reported as AUC inside the tie group, so 0.5 is worthless. Anything at or below the
incumbent's 0.4955 is dead on arrival. Zero GPU.
"""

import bisect
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
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


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}

    import random
    rng = random.Random(31)
    zeros = sorted(rng.sample([q for q, v in nd.items() if v == 0.0], 1500), key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names = {str(r["item_id"]): r["name"] for r in corpus}
    order = [str(r["item_id"]) for r in corpus]
    pos_in_file = {iid: i for i, iid in enumerate(order)}

    # exact-duplicate-name groups
    dup = defaultdict(list)
    for iid in order:
        dup[normalize_arabic(names[iid])].append(iid)
    dup_rank = {}
    dup_size = {}
    for nm, ids in dup.items():
        ids_sorted = sorted(ids, key=num)
        for r, iid in enumerate(ids_sorted):
            dup_rank[iid] = r
            dup_size[iid] = len(ids_sorted)

    print("building pools (0 GPU) ...", flush=True)
    h = Harness(zeros, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    # precompute tie groups
    groups = {}
    for q in zeros:
        pool = h.pools[q]
        cs = [h.cont3(q, i) for i in pool]
        mx = max(cs)
        grp = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
        if len(grp) >= 2 and any(qrels[q].get(i, 0) > 0 for i in grp) \
           and any(qrels[q].get(i, 0) <= 0 for i in grp):
            groups[q] = grp
    print(f"{len(groups)} usable tie groups, mean size "
          f"{st.mean(len(g) for g in groups.values()):.1f}\n")

    def h1_contiguity(q, iid, grp):
        """negative mean id-distance to the rest of the group (closer = higher)"""
        others = [num(x) for x in grp if x != iid]
        return -st.mean(abs(num(iid) - o) for o in others) if others else 0.0

    def h2_group_rank(q, iid, grp):
        return -sorted(grp, key=num).index(iid)

    def h3_isolation(q, iid, grp):
        """density of catalog neighbours sharing tokens with this item"""
        p = pos_in_file[iid]
        toks = set(normalize_arabic(names[iid]).split())
        hits = 0
        for d in range(-10, 11):
            j = p + d
            if d == 0 or j < 0 or j >= len(order):
                continue
            if toks & set(normalize_arabic(names[order[j]]).split()):
                hits += 1
        return hits

    def h4_block(q, iid, grp):
        """proximity in FILE order to the rest of the group"""
        others = [pos_in_file[x] for x in grp if x != iid]
        return -st.mean(abs(pos_in_file[iid] - o) for o in others) if others else 0.0

    def h5_duprank(q, iid, grp):
        return -dup_rank.get(iid, 0)

    def h5b_dupsize(q, iid, grp):
        return dup_size.get(iid, 1)

    def incumbent(q, iid, grp):
        return -num(iid)

    tests = (("H0 low-id (incumbent)", incumbent),
             ("H1 id contiguity", h1_contiguity),
             ("H2 rank within group", h2_group_rank),
             ("H3 neighbourhood density", h3_isolation),
             ("H4 file-order proximity", h4_block),
             ("H5 rank among duplicates", h5_duprank),
             ("H5b duplicate-group size", h5b_dupsize))

    print("AUC INSIDE the maximum-containment tie group, zero cohort")
    print("(0.5 = worthless; the incumbent is the bar to beat)\n")
    for label, fn in tests:
        pos, neg = [], []
        for q, grp in groups.items():
            for iid in grp:
                v = fn(q, iid, grp)
                (pos if qrels[q].get(iid, 0) > 0 else neg).append(v)
        a = auc(pos, neg)
        flag = ""
        if a == a:
            if a > 0.55:
                flag = "  <-- above chance"
            elif a < 0.45:
                flag = "  <-- inverted (usable if flipped)"
        print(f"  {label:<28s} AUC {a:.4f}{flag}")

    print("\nNOTE: an id-space structure would have to beat 0.4955 by a wide margin to")
    print("matter, since GATE-AraBert reached 0.5844 there and still cost -0.0057")
    print("end to end once false positives were priced in.")


if __name__ == "__main__":
    main()
