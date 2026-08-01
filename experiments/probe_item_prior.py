"""Does TRAINING-set item popularity predict benchmark relevance?

Motivation from `autopsy_zero_ordering.py`: the zero cohort is dominated by SHORT,
often brand-only queries (mean 1.79 tokens) such as 'مولي', 'enzo', 'كلاتشي',
'فاونديشن'. For those, many catalog products match the text about equally well, so
no amount of text modelling can discriminate; what separates the judged-relevant
items from the rest has to be a PRIOR OVER ITEMS. Our only item prior today is the
low-id tie-break, which is a crude proxy for catalog age.

`data/training/positives/train.parquet` gives 558,253 (user_query,
positive_product_name) pairs. Counting how often each product name appears as a
positive yields a direct popularity prior. This uses TRAINING data only, never
`candidates.jsonl`, so it is clean under rule 2.4.b.

This script only MEASURES discriminative power; it changes no ranking. Reported as
a within-pool AUC (probability a random relevant item outranks a random
non-relevant one), so 0.5 is worthless and the low-id prior is the benchmark to beat.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import BENCH, DATA, ROOT, load_jsonl, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def build_popularity() -> dict[str, int]:
    """normalized product name -> times it appears as a training positive."""
    df = pd.read_parquet(DATA / "training" / "positives" / "train.parquet")
    c = Counter(normalize_arabic(str(n)) for n in df["positive_product_name"].dropna())
    print(f"training positives: {len(df):,} rows -> {len(c):,} distinct normalized names")
    return c


def auc_within_pool(qids, h, qrels, score_fn) -> tuple[float, int]:
    """P(random relevant item scores above a random non-relevant one), pooled."""
    conc = disc = ties = 0
    for q in qids:
        rel, non = [], []
        for iid in h.pools[q]:
            (rel if qrels[q].get(iid, 0) > 0 else non).append(score_fn(q, iid))
        if not rel or not non:
            continue
        non_sorted = sorted(non)
        import bisect
        for r in rel:
            lo = bisect.bisect_left(non_sorted, r)
            hi = bisect.bisect_right(non_sorted, r)
            conc += lo
            ties += hi - lo
            disc += len(non_sorted) - hi
    tot = conc + disc + ties
    return ((conc + 0.5 * ties) / tot if tot else 0.5), tot


def main():
    pop = build_popularity()
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    scored = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    zeros = sorted(q for q, v in scored.items() if v == 0.0)

    rng = random.Random(11)
    others = [q for q, v in scored.items() if v > 0.0]
    sample_other = sorted(rng.sample(others, 2000), key=int)
    qids = sorted(set(zeros) | set(sample_other), key=int)

    print(f"\nbuilding pools for {len(qids)} queries (0 GPU)...", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    print("done\n")

    covered = sum(1 for iid in h.names if normalize_arabic(h.names[iid]) in pop)
    print(f"catalog coverage: {covered:,}/{len(h.names):,} items have a training "
          f"positive ({100 * covered / len(h.names):.1f}%)\n")

    def f_pop(q, iid):
        return pop.get(normalize_arabic(h.names[iid]), 0)

    def f_lowid(q, iid):
        digits = "".join(ch for ch in iid if ch.isdigit())
        return -int(digits) if digits else 0

    def f_ce(q, iid):
        return h.ce[(q, iid)]

    print("WITHIN-POOL AUC (0.5 = no signal; higher is better)\n")
    print(f"  {'slice':<26s}{'popularity':>12s}{'low-id':>10s}{'CE':>10s}")
    for label, qs in (("zero cohort", zeros),
                      ("non-zero sample", sample_other),
                      ("short queries (<=2 tok)",
                       [q for q in qids if len(normalize_arabic(h.queries[q]).split()) <= 2])):
        a_pop, _ = auc_within_pool(qs, h, qrels, f_pop)
        a_id, _ = auc_within_pool(qs, h, qrels, f_lowid)
        a_ce, _ = auc_within_pool(qs, h, qrels, f_ce)
        print(f"  {label:<26s}{a_pop:>12.4f}{a_id:>10.4f}{a_ce:>10.4f}")

    # is popularity merely restating the low-id prior?
    ids_seen, xs, ys = set(), [], []
    for q in qids[:500]:
        for iid in h.pools[q]:
            if iid in ids_seen:
                continue
            ids_seen.add(iid)
            d = "".join(ch for ch in iid if ch.isdigit())
            if d:
                xs.append(int(d))
                ys.append(pop.get(normalize_arabic(h.names[iid]), 0))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    print(f"\n  corr(product id, popularity) = {cov / (vx * vy):+.4f} over {n:,} items")
    print("  (near zero means popularity is genuinely NEW information, not the id prior)")


if __name__ == "__main__":
    main()
