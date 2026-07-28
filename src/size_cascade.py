"""Size Play A's cascade BEFORE buying any GPU for it.

The 4B plan has always been costed as "score the wide band at depth 50" = ~0.77M pairs =
~8.5 h locally. That number was never derived from what the re-scoring can actually reach;
it was chosen by analogy with the existing pools. With a hard GPU budget (5.34 h of Kaggle
quota left, refreshing only AFTER the deadline) the right question is the inverse: what is
the SMALLEST top-N re-scoring that still contains the achievable gain?

The bound is exact and costs nothing. If the 4B re-orders only the top-N of a query's
current ranking and leaves the rest untouched, then the final top-10 is drawn entirely from
those N, so a PERFECT 4B scores exactly the best ten grades available inside the top-N. That
is the ceiling for any cascade at that depth, and it is computable from the shipped run plus
the judged grades.

Also reported: the wide-band restriction. The cascade was always meant to run only on
low-confidence queries, so the ceiling is computed both over all queries and over the band.

candidates.jsonl is MEASUREMENT-ONLY under rule 2.4.b; nothing here feeds retrieval,
training or tuning, it only prices a decision.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
T_LIVE, BETA = 3.2, 0.35
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2

DEPTHS = (10, 20, 30, 50, 100)
# measured throughputs for the 4B seq-cls at seq 160
RATE_LOCAL = 25.0      # pairs/s, RTX 4060 4-bit, batch 32
RATE_KAGGLE = 40.0     # pairs/s aggregate, 2x T4 fp16


def oracle_topn(order, qrel, n):
    got = sorted((max(qrel.get(i, 0), 0) for i in order[:n]), reverse=True)[:10]
    dcg = sum((2.0 ** g - 1.0) / math.log2(k + 2) for k, g in enumerate(got))
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:10]
    idcg = sum((2.0 ** g - 1.0) / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    qrels = load_candidates(BENCH / "candidates.jsonl")

    print("building harness over the full benchmark ...", flush=True)
    h = Harness(all_qids, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
    mpc, ftc = {}, {}

    def mp(q, i):
        v = mpc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            mpc[(q, i)] = v
        return v

    def ft(q, i):
        v = ftc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
            ftc[(q, i)] = v
        return v

    key = lambda q, i: (h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT)
    full = h.build_run(key=key, bonus_fn=lambda q, i: BETA * mp(q, i), topn=200, **LIVE)

    live = load_run_csv(ROOT / "runs" / "submission_v7synth_firsttok.csv")
    same = sum(1 for q in live if q in full and live[q][:10] == full[q][:10])
    print(f"REGRESSION vs the file on the board (LB 0.68822): {same:,}/{len(live):,}\n")

    qs = [q for q in all_qids if q in qrels]
    band = [q for q in qs if h.conf[q] < LIVE["conf_t"]]
    cur = sum(ndcg_at_k(full[q], qrels[q], 10) for q in qs) / len(qs)
    print(f"current local score {cur:.6f}   wide band {len(band):,}/{len(qs):,} "
          f"queries ({len(band)/len(qs):.1%})\n")

    def report(label, targets):
        tset = set(targets)
        print(f"{label}  (re-scoring only these queries)")
        print(f"  {'depth':>6}{'pairs':>12}{'local h':>10}{'kaggle h':>10}"
              f"{'CEILING':>11}{'headroom':>11}")
        for n in DEPTHS:
            pairs = sum(min(n, len(h.pools[q])) for q in targets)
            tot = 0.0
            for q in qs:
                tot += oracle_topn(full[q], qrels[q], n) if q in tset \
                    else ndcg_at_k(full[q], qrels[q], 10)
            ceil = tot / len(qs)
            print(f"  {n:>6}{pairs:>12,}{pairs/RATE_LOCAL/3600:>10.1f}"
                  f"{pairs/RATE_KAGGLE/3600:>10.1f}{ceil:>11.6f}{ceil-cur:>+11.6f}")
        print()

    report("WIDE BAND ONLY", band)
    report("ALL QUERIES", qs)

    print("How to read this: CEILING is what a PERFECT 4B would score at that depth, so the")
    print("realistic gain is a fraction of the headroom column. Play A's own honest EV is")
    print("+0.005 to +0.015; any depth whose CEILING headroom is near that is a depth where")
    print("even a flawless model could not pay for itself.")
    print(f"\nKaggle GPU remaining: 5.34 h, and the quota refreshes 2026-08-01, AFTER close.")


if __name__ == "__main__":
    main()
