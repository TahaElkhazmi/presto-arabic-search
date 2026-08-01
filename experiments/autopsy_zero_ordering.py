"""WHY do we score exactly zero on 1,988 queries whose answer is in the pool?

`autopsy_zero_cohort.py` established that 85% of the zero cohort is ordering-limited
and that a perfect re-ranker on our EXISTING 100+100 pools would be worth +0.0607
full-benchmark on this cohort alone, i.e. more than the +0.0455 gap to the leader.
Scoring zero means not one relevant item reached our top-10 while relevant items sat
in the pool. That is a catastrophic failure, not a subtle one, so it should have a
signature.

This characterises the cohort against a matched comparison group so the numbers mean
something:
  - where does the best relevant item actually LAND in our final ordering? A mass at
    ranks 11-20 is a near-miss a cheap nudge could fix; a mass at rank 100+ means the
    re-ranker actively hates these items and only a better model helps.
  - is the failure concentrated in a query TYPE (length, dialect fillers, generic
    category queries, digits/sizes)?
  - what does the CE think? If it is confidently wrong, that is different from
    confidently unsure.

Zero GPU: the v7synth cache covers every benchmark pair.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
LIVE_CSV = ROOT / "runs" / "submission_v7synth_margin3.csv"


def describe(label, vals):
    if not vals:
        print(f"  {label:<34s} (empty)")
        return
    print(f"  {label:<34s} n={len(vals):<6d} mean={st.mean(vals):7.3f} "
          f"median={st.median(vals):7.3f}")


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(LIVE_CSV)
    scored = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}

    zeros = sorted(q for q, v in scored.items() if v == 0.0)
    # a matched comparison group: queries we score WELL on
    good = sorted(q for q, v in scored.items() if v >= 0.9)
    print(f"zero cohort {len(zeros)}, high-scoring comparison group {len(good)}\n")

    sample_good = good[:: max(1, len(good) // 3000)][:3000]
    qids = sorted(set(zeros) | set(sample_good), key=int)
    print(f"building live ordering for {len(qids)} queries (cache hit, 0 GPU)...",
          flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    full = h.build_run(topn=10**6, **LIVE_STACK)
    print("done\n")

    zset = set(zeros)
    # only the ordering-limited ones (a relevant item IS in the pool)
    ordering_limited = []
    for q in zeros:
        if any(qrels[q].get(i, 0) > 0 for i in h.pools[q]):
            ordering_limited.append(q)
    print(f"ordering-limited zero queries: {len(ordering_limited)}\n")

    def best_rel_rank(q):
        for r, iid in enumerate(full[q], 1):
            if qrels[q].get(iid, 0) > 0:
                return r
        return None

    ranks = [r for r in (best_rel_rank(q) for q in ordering_limited) if r]
    print("WHERE THE BEST RELEVANT ITEM LANDS (ordering-limited zero cohort)")
    buckets = [(11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, 10**6)]
    for lo, hi in buckets:
        c = sum(1 for r in ranks if lo <= r <= hi)
        lab = f"ranks {lo}-{hi}" if hi < 10**6 else f"ranks {lo}+"
        bar = "#" * int(60 * c / max(len(ranks), 1))
        print(f"  {lab:<14s} {c:>5d} ({100 * c / len(ranks):>4.1f}%) {bar}")
    near = sum(1 for r in ranks if r <= 20)
    print(f"\n  NEAR MISS (rank 11-20): {near} queries = {100 * near / len(ranks):.1f}% "
          f"of the ordering-limited cohort")
    print(f"  median landing rank: {st.median(ranks):.0f}\n")

    # query shape: zero cohort vs high-scoring group
    def qfeat(qs):
        toks, has_digit, has_filler, latin = [], 0, 0, 0
        for q in qs:
            t = normalize_arabic(h.queries[q]).split()
            toks.append(len(t))
            if any(ch.isdigit() for ch in h.queries[q]):
                has_digit += 1
            if any(w in DIALECT_FILLERS for w in t):
                has_filler += 1
            if any("a" <= ch.lower() <= "z" for ch in h.queries[q]):
                latin += 1
        n = len(qs)
        return toks, 100 * has_digit / n, 100 * has_filler / n, 100 * latin / n

    print("QUERY SHAPE: zero cohort vs high-scoring group")
    for lab, qs in (("zero cohort", ordering_limited), ("high-scoring", sample_good)):
        toks, dig, fil, lat = qfeat(qs)
        print(f"  {lab:<16s} n={len(qs):<5d} tokens mean={st.mean(toks):.2f} "
              f"median={st.median(toks):.0f} | digits {dig:.1f}% | "
              f"fillers {fil:.1f}% | latin {lat:.1f}%")

    print("\nPOOL / LABEL STRUCTURE")
    for lab, qs in (("zero cohort", ordering_limited), ("high-scoring", sample_good)):
        npos = [sum(1 for g in qrels[q].values() if g > 0) for q in qs]
        pool = [len(h.pools[q]) for q in qs]
        inpool = [sum(1 for i in h.pools[q] if qrels[q].get(i, 0) > 0) for q in qs]
        conf = [h.conf[q] for q in qs]
        print(f"  --- {lab} ---")
        describe("judged positives per query", npos)
        describe("pool size", pool)
        describe("relevant items IN pool", inpool)
        describe("CE confidence margin", conf)

    print("\nWHAT WE PUT AT RANK 1 INSTEAD (10 examples)")
    for q in ordering_limited[:10]:
        r = best_rel_rank(q)
        got = h.names[full[q][0]]
        want_id = next(i for i in full[q] if qrels[q].get(i, 0) > 0)
        print(f"  q={h.queries[q][:40]!r}")
        print(f"    we ranked #1 : {got[:62]}")
        print(f"    relevant @{r:<4d}: {h.names[want_id][:62]}")


if __name__ == "__main__":
    main()
