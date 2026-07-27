"""Is the ZERO-SCORING cohort recall-limited or ordering-limited?

8.3% of queries score exactly 0 nDCG@10 and hold 24.7% of all remaining loss. We
closed pool deepening as "measured dead" on AVERAGE evidence: oracle 0.9602 at
100+100 versus 0.9919 at 1000+1000, and +0.0009 end to end. But an average over
29,014 queries can hide a concentrated pocket, and this is the most concentrated
pocket we have.

Two mutually exclusive explanations for a zero:
  (a) RECALL   - no judged-relevant item is in the 100+100 pool at all, so no
                 re-ranker of any quality could have scored above zero. Deeper pools
                 are then the ONLY fix, and our "deepening is dead" conclusion is an
                 averaging artifact for this cohort.
  (b) ORDERING - relevant items are in the pool but our stack buried them below
                 rank 10. Deepening is irrelevant; only a better orderer helps.

The split between (a) and (b) decides whether Play C is dead or is actually the
cheapest remaining lever, so it is worth measuring before anything expensive.

Zero GPU: run files plus qrels only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import stream_topk

setup_utf8()

LIVE = ROOT / "runs" / "submission_v7synth_margin3.csv"
DEPTHS = (100, 200, 500, 1000)


def oracle(pool: set[str], qrel: dict[str, int]) -> float:
    """nDCG@10 of the best ordering achievable from this pool."""
    rel = sorted(((g, i) for i, g in qrel.items() if g > 0 and i in pool),
                 key=lambda x: -x[0])
    return ndcg_at_k([i for _, i in rel[:10]], qrel, 10)


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(LIVE)
    print(f"live run {LIVE.name}: {len(run)} queries\n")

    scored = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    n = len(scored)
    overall = sum(scored.values()) / n
    zeros = sorted(q for q, v in scored.items() if v == 0.0)
    print(f"overall nDCG@10 = {overall:.6f} over {n} queries")
    print(f"ZERO cohort: {len(zeros)} queries ({100 * len(zeros) / n:.1f}%)")
    print(f"  they contribute {len(zeros)} * 0 to the mean; lifting them all to 1.0 "
          f"would add {len(zeros) / n:.4f} nDCG\n")

    # do these queries even have judged positives?
    no_pos = [q for q in zeros if not any(g > 0 for g in qrels[q].values())]
    print(f"  of those, {len(no_pos)} have NO judged-relevant item at all "
          f"(unwinnable by anyone; they cap everyone's score)")
    winnable = [q for q in zeros if q not in set(no_pos)]
    print(f"  WINNABLE zero cohort: {len(winnable)} queries\n")
    if not winnable:
        return

    qset = set(winnable)
    print("streaming depth-1000 run files (this takes a moment)...", flush=True)
    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", 1000, qset)
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", 1000, qset)
    print("done\n")

    print("Oracle nDCG@10 on the WINNABLE zero cohort, by pool depth")
    print("(what a PERFECT re-ranker could score on exactly these queries)\n")
    print(f"  {'depth':>7s}  {'oracle':>8s}  {'has >=1 relevant':>18s}  "
          f"{'full-bench value':>17s}")
    prev = None
    for d in DEPTHS:
        vals, hit = [], 0
        for q in winnable:
            pool = set(bm.get(q, [])[:d]) | set(dn.get(q, [])[:d])
            o = oracle(pool, qrels[q])
            vals.append(o)
            if o > 0:
                hit += 1
        mean_o = sum(vals) / len(vals)
        # these queries currently score 0, so oracle IS the recoverable headroom
        bench_value = mean_o * len(winnable) / n
        mark = "" if prev is None else f"  (+{mean_o - prev:.4f} vs prev)"
        print(f"  {d:>7d}  {mean_o:>8.4f}  {hit:>10d}/{len(winnable):<7d}  "
              f"{bench_value:>+17.4f}{mark}")
        prev = mean_o

    # the decisive split, at our LIVE depth
    vals100 = []
    for q in winnable:
        pool = set(bm.get(q, [])[:100]) | set(dn.get(q, [])[:100])
        vals100.append(oracle(pool, qrels[q]))
    ordering_limited = sum(1 for v in vals100 if v > 0)
    recall_limited = sum(1 for v in vals100 if v == 0)
    print(f"\nDECISIVE SPLIT at our live depth (100+100):")
    print(f"  ORDERING-limited: {ordering_limited} ({100 * ordering_limited / len(winnable):.1f}%) "
          f"- relevant item IS in the pool, we buried it")
    print(f"  RECALL-limited:   {recall_limited} ({100 * recall_limited / len(winnable):.1f}%) "
          f"- nothing relevant in the pool; only deeper pools can help")

    # of the recall-limited ones, how many does depth 1000 rescue?
    rescued = 0
    for q in winnable:
        p100 = set(bm.get(q, [])[:100]) | set(dn.get(q, [])[:100])
        if oracle(p100, qrels[q]) > 0:
            continue
        p1000 = set(bm.get(q, [])[:1000]) | set(dn.get(q, [])[:1000])
        if oracle(p1000, qrels[q]) > 0:
            rescued += 1
    print(f"  of the recall-limited, depth 1000 rescues {rescued}/{recall_limited}")
    print(f"  (the rest are outside our retrieval entirely at any depth we run)")


if __name__ == "__main__":
    main()
