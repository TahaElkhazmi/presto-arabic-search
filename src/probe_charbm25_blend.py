"""TASK 1 stage 1: the FREE half of the char-ngram third voter.

A third retrieval voter does two separable things:
  (a) admits candidates the other two missed - costs a GPU pass, since new candidates
      have no cached cross-encoder score. Stage 0 (probe_charbm25_pool.py) prices that
      half: +0.0059 of pool-oracle headroom, an upper bound needing a PERFECT re-ranker.
  (b) re-weights the RRF prior for candidates already in the pool - completely free,
      because every one of those pairs is already in the v7synth cache.

This measures (b) in isolation, at the CURRENT live operating point (T=3.2, lam=0.50,
eps .030/.050, conf_t 0.10 on the top1-top3 margin, cont3pos in-band key, beta=0.35
match-position blend bonus). Restricting the votes to existing pool members is what
makes it zero-GPU and what keeps the comparison honest: any movement here is prior
re-weighting alone, not recall.

Protocol: tune the voter weight on the tune half of holdout_remainder8012 (seed 77,
the same halves every recent campaign used), confirm on the other half, then report the
full benchmark and the untouched arbiter. Weight 0 must reproduce the file on the board.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, stream_topk

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE = 3.2
BETA = 0.35
RRF_K = 60
WEIGHTS = (3.0, 1.0)
GRID = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def build_harness(qids, ntok):
    h = Harness(qids, T=T_LIVE, cache_stem=V7, weights=list(WEIGHTS), rrf_k=RRF_K)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
    cache = {}

    def mp(q, i):
        v = cache.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            cache[(q, i)] = v
        return v
    return h, mp


def char_votes(h, ch_run, weight):
    """RRF votes from the char voter, RESTRICTED to items already in the pool.

    Items outside the pool are dropped rather than admitted: admitting them would
    require cross-encoder scores we do not have, which is exactly the cost stage 0
    priced. So this isolates prior re-weighting from recall.
    """
    if weight <= 0:
        return None
    out = {}
    for q in h.qids:
        pool = set(h.pools[q])
        d = {}
        for rank, iid in enumerate(ch_run.get(q, []), start=1):
            if iid in pool:
                d[iid] = weight / (RRF_K + rank)
        out[q] = d
    return out


def run_at(h, mp, ch_run, weight, qids=None):
    ev = char_votes(h, ch_run, weight)
    return h.build_run(key=lambda q, i: h.cont3(q, i) * 1000.0 + mp(q, i),
                       bonus_fn=lambda q, i: BETA * mp(q, i),
                       extra_votes=ev, extra_weight=weight,
                       qids=qids, **LIVE)


def mean_ndcg(run, qrels, qids):
    v = [ndcg_at_k(run[q], qrels[q], 10) for q in qids if q in qrels and q in run]
    return sum(v) / len(v)


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ch_run = stream_topk(ROOT / "runs" / "charbm25_34gram_all.jsonl", 100, set(all_qids))

    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    print("building harness over the full benchmark ...", flush=True)
    h, mp = build_harness(all_qids, ntok)
    qrels = h.qrels

    # REGRESSION: weight 0 must reproduce the file currently scoring 0.68527.
    base = run_at(h, mp, ch_run, 0.0)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_retune_bonus.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs submission_v7synth_retune_bonus.csv (LB 0.68527): "
          f"{same:,}/{len(live):,}")
    if same != len(live):
        print("  *** harness does NOT reproduce the live file - stop and diagnose ***")
        return
    print()

    print("voter weight sweep (tune half of holdout_remainder8012, n=4,006)")
    b_tune = mean_ndcg(base, qrels, tune)
    best = None
    for w in GRID:
        r = base if w == 0 else run_at(h, mp, ch_run, w, qids=tune)
        s = mean_ndcg(r, qrels, tune)
        print(f"  weight {w:>5.2f}   {s:.6f}   {s - b_tune:+.6f}")
        if w > 0 and (best is None or s > best[1]):
            best = (w, s)
    print(f"\n  best positive weight on tune: {best[0]:.2f} ({best[1] - b_tune:+.6f})")

    w = best[0]
    if w in (GRID[1], GRID[-1]):
        print("  NOTE: optimum sits on a grid edge - unbracketed, treat with suspicion")

    cand = run_at(h, mp, ch_run, w)
    print(f"\nFULL EVALUATION at weight {w:.2f}")
    used = set(load_qid_file(ROOT / "splits" / "val_qids.txt"))
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt",
              "dev_listwise200.txt", "probe_filler800.txt", "probe_band600.txt",
              "probe_judge150.txt"):
        p = ROOT / "splits" / f
        if p.exists():
            used |= load_qid_file(p)
    untouched = sorted(set(all_qids) - used, key=int)

    for label, qs in (("confirm half", conf), ("FULL BENCHMARK", all_qids),
                      ("UNTOUCHED arbiter", untouched),
                      ("val", sorted(load_qid_file(ROOT / "splits" / "val_qids.txt"), key=int))):
        qs = [q for q in qs if q in qrels]
        a = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
        b = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
        d = [y - x for x, y in zip(a, b)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<20s} n={len(qs):>6,}  live {sum(a)/len(a):.6f}  "
              f"cand {sum(b)/len(b):.6f}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")

    print("\nGATE: +0.0015 on the UNTOUCHED arbiter, or it does not ship.")


if __name__ == "__main__":
    main()
