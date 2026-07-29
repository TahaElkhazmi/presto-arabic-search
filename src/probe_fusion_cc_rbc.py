"""Two PUBLISHED fusion functions against our RRF prior, on Taha's request.

RBC - Rank-Biased Centroids. Geometric rank decay w * (1-phi) * phi^(rank-1) instead of
RRF's reciprocal w / (k + rank). Reported to beat Borda-fuse and CombMNZ on nDCG. Uses
RANKS ONLY, so it is pure CPU and can run beside a GPU training job.

CC - convex combination of per-query normalized scores, from Bruch, Gai & Ingber,
"An Analysis of Fusion Functions for Hybrid Retrieval" (TOIS 2023), which reports CC
beating RRF in and out of domain and calls RRF parameter-sensitive. The paper also shows
the choice of normalization is rank-equivalent, so min-max is used here.

WHY THIS IS NOT THE TEST WE ALREADY RAN. `probe_score_prior.py` interpolated rank-form
against score-form for the LEXICAL VOTER ONLY, keeping RRF's structure, and found it
monotonically worse. Its own docstring says why it stopped there: the dense scores "are
not cached and would need a GPU pass". CC replaces the whole fusion and needs BOTH voters'
scores, so it needs that pass. It also carries a recorded caveat: probe_score_prior used
the harness default top1-top5 confidence rather than the shipped top1-top3, so its
baseline was slightly off.

CC is skipped, loudly, unless the dense pool scores exist (src/dense_pool_scores.py).
Everything else in the stack3 transform is held fixed, and lam is swept alongside because
changing the prior's shape changes what the blend should weight (protocol rule 2).
"""

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, DATA, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    setup_utf8, tokenize
from eval_ndcg import ndcg_at_k
from probe_ensemble_current import RRF_MAX, Stack3
from probe_harness import stream_topk
from probe_score_prior import build_word_bm25

setup_utf8()

DENSE_CACHE = DATA / "cache" / "dense_pool_scores_lora_v1.parquet"


def rbc_prior(ranks, weights, phi, qids):
    """w * (1-phi) * phi^(rank-1), summed over voters. prior_max mirrors RRF_MAX: the
    value an item ranked 1 by every voter receives."""
    out = {}
    for q in qids:
        acc = {}
        for run, w in zip(ranks, weights):
            for r, iid in enumerate(run.get(q, []), start=1):
                acc[iid] = acc.get(iid, 0.0) + w * (1.0 - phi) * (phi ** (r - 1))
        out[q] = acc
    return out, sum(weights) * (1.0 - phi)


def minmax(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    rng = hi - lo
    if rng <= 0:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / rng for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phis", default="0.60,0.70,0.80,0.90,0.95,0.98")
    ap.add_argument("--alphas", default="0.30,0.50,0.65,0.75,0.85")
    ap.add_argument("--lams", default="0.45,0.50,0.55")
    args = ap.parse_args()

    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")
    qids = sorted(hold, key=int)
    print(f"building stack3 harness over {len(qids):,} holdout queries ...", flush=True)
    S = Stack3(qids)
    h = S.h
    qrels = h.qrels

    base = S.build(h.ce)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    same = sum(1 for q in h.qids if live.get(q, [])[:10] == base[q][:10])
    print(f"REGRESSION vs submission_v7synth_stack3.csv: {same:,}/{len(h.qids):,}")
    if same != len(h.qids):
        print("  *** baseline does not reproduce - stop ***")
        return

    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune, cf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])

    def halves(run):
        out = []
        for qs in (tune, cf):
            sel = [q for q in h.qids if q in qs and q in qrels]
            out.append([ndcg_at_k(run[q], qrels[q], 10) for q in sel])
        return out

    b_t, b_c = halves(base)
    print(f"baseline  tune {sum(b_t)/len(b_t):.6f}  confirm {sum(b_c)/len(b_c):.6f}\n")

    def report(label, run):
        c_t, c_c = halves(run)
        dt = sum(y - x for x, y in zip(b_t, c_t)) / len(b_t)
        dd = [y - x for x, y in zip(b_c, c_c)]
        dc = sum(dd) / len(dd)
        se = st.pstdev(dd) / len(dd) ** 0.5
        print(f"{label:<26s}{dt:>+12.6f}{dc:>+12.6f}{dc/se if se else 0:>+7.2f}", flush=True)
        return dc

    bm_rank = stream_topk(ROOT / "runs" / "bm25_all.jsonl", 100, set(qids))
    dn_rank = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", 100, set(qids))

    # ---- RBC (ranks only, pure CPU) ----
    print("=== RBC, rank-biased centroids (geometric decay instead of reciprocal)")
    print(f"{'config':<26s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")
    best = []
    for phi in [float(x) for x in args.phis.split(",")]:
        pri, pmax = rbc_prior([bm_rank, dn_rank], [3.0, 1.0], phi, h.qids)
        for lam in [float(x) for x in args.lams.split(",")]:
            d = report(f"phi={phi:.3f} lam={lam:.2f}", S.build(h.ce, lam=lam,
                                                               priors=pri, prior_max=pmax))
            best.append((d, f"RBC phi={phi} lam={lam}"))

    # ---- CC (needs both voters' scores) ----
    if not DENSE_CACHE.exists():
        print(f"\n=== CC SKIPPED: {DENSE_CACHE.name} absent.")
        print("    CC needs dense scores, which are not cached and need a short GPU pass.")
        print("    Run src/dense_pool_scores.py when the GPU is free, then re-run this.")
    else:
        print("\n=== CC, convex combination of min-max normalized scores (Bruch 2023)")
        t0 = time.time()
        corpus = load_jsonl(BENCH / "corpus.jsonl")
        W, vocab = build_word_bm25(corpus)
        item_ids = [str(r["item_id"]) for r in corpus]
        row_of = {iid: n for n, iid in enumerate(item_ids)}
        print(f"  word BM25 index rebuilt in {time.time()-t0:.1f}s", flush=True)

        dn = pd.read_parquet(DENSE_CACHE)
        dense = {(q, i): float(s) for q, i, s in zip(dn["qid"], dn["iid"], dn["score"])}
        miss = sum(1 for q in h.qids for i in h.pools[q] if (q, i) not in dense)
        print(f"  dense pool scores: {len(dense):,} rows, {miss:,} pool pairs missing")
        if miss:
            print("  *** dense cache does not cover the pools - skipping CC ***")
            return

        lex = {}
        for q in h.qids:
            idx = [vocab[t] for t in tokenize(h.queries[q]) if t in vocab]
            if idx:
                col = np.asarray(W[:, idx].sum(axis=1)).ravel()
            else:
                col = np.zeros(len(item_ids))
            lex[q] = {i: float(col[row_of[i]]) for i in h.pools[q]}

        print(f"{'config':<26s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")
        for alpha in [float(x) for x in args.alphas.split(",")]:
            pri = {}
            for q in h.qids:
                L = minmax(lex[q])
                D = minmax({i: dense[(q, i)] for i in h.pools[q]})
                pri[q] = {i: alpha * L.get(i, 0.0) + (1 - alpha) * D.get(i, 0.0)
                          for i in h.pools[q]}
            for lam in [float(x) for x in args.lams.split(",")]:
                d = report(f"alpha={alpha:.2f} lam={lam:.2f}",
                           S.build(h.ce, lam=lam, priors=pri, prior_max=1.0))
                best.append((d, f"CC alpha={alpha} lam={lam}"))

    print("\nbest by confirm half (live RRF prior is the 0.000000 reference):")
    for d, label in sorted(best, reverse=True)[:5]:
        print(f"  {label:<28s} {d:+.6f}")


if __name__ == "__main__":
    main()
