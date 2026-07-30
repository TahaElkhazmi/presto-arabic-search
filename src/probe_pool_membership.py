"""SCOPING: is reproducing the Meilisearch pool ordering worth building?

THE IDEA. The judged pools were produced by Meilisearch hybrid search (BM25 + GATE-AraBert
at semanticRatio 0.3) over the same corpus. Meilisearch is a deterministic function of
(corpus, query), so it can be reimplemented from legal inputs only - no use of
candidates.jsonl, which rule 2.4.b makes measurement-only.

WHY IT MIGHT MATTER, stated precisely. nDCG@10 only credits items that were JUDGED. Any
item we place in the top-10 that Meilisearch never retrieved is unjudged and scores zero
gain no matter how relevant it actually is. Our pool is ~166 candidates but only ~9 per
query were ever judged, so "would Meilisearch have retrieved this?" is a different and
possibly EASIER prediction target than "is this relevant?" - the first is a reproducible
algorithm, the second is a human judgement.

WHAT THIS SCRIPT DECIDES, before any implementation effort:
  1. How much of our shipped top-10 is unjudged right now (the size of the prize).
  2. Whether GATE-AraBert cosine + BM25 - the two ingredients Meilisearch actually used -
     predict JUDGED MEMBERSHIP better than our own blend does. If our blend already ranks
     membership as well as the ingredients do, a faithful reimplementation cannot add much
     and the idea is dead without writing it.

AUC is computed WITHIN each query and aggregated, never pooled - the estimator error this
project already made once and corrected.

CORRECTED 2026-07-30. This script originally loaded our live ranking with `load_run_csv`,
which TRUNCATES TO THE TOP 10, and then scored every remaining pool item with a single
sentinel. Our 166-item ordering was therefore evaluated as a 10-item list plus a 156-way
tie, and its membership AUC came out at 0.7476 against the hybrid's 0.8446 - which read as
"a signal our system is measurably bad at" and was recorded that way. Scored on the FULL
ranking from the run's .jsonl, our ordering is at 0.8903 and BEATS the hybrid at every
depth. The conclusion (membership does not convert) is unchanged; the reason is the
opposite of what was written. See probe_pool_membership_conditioned output in FINDINGS.

candidates.jsonl is read for MEASUREMENT ONLY here.
"""

import bisect
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, DATA, ROOT, load_jsonl, load_run_csv, setup_utf8, tokenize
from eval_ndcg import load_candidates, load_run_jsonl
from probe_harness import stream_topk
from probe_phase3_predicates import per_query_auc
from probe_score_prior import build_word_bm25

setup_utf8()


def main():
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = load_jsonl(BENCH / "queries.jsonl")
    item_ids = [str(r["item_id"]) for r in corpus]
    ipos = {i: k for k, i in enumerate(item_ids)}
    qids = [str(q["query_id"]) for q in queries]
    qpos = {q: k for k, q in enumerate(qids)}
    qrels = load_candidates(BENCH / "candidates.jsonl")

    # top-10 view, used only for the composition table below
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack5.csv")
    # FULL ranking, used for the AUC comparison. load_run_csv truncates to 10; using it
    # here is the bug this file documents, so the .jsonl is read instead.
    live_full = load_run_jsonl(ROOT / "runs" / "submission_v7synth_stack5.jsonl")

    # ---------- 1. how much of the shipped top-10 is unjudged? ----------
    judged = pos = tot = 0
    for q, ids in live.items():
        if q not in qrels:
            continue
        r = qrels[q]
        for i in ids[:10]:
            tot += 1
            if i in r:
                judged += 1
                if r[i] > 0:
                    pos += 1
    print("SHIPPED TOP-10 COMPOSITION (LB 0.69362)")
    print(f"  judged at all      {judged/tot:.1%}")
    print(f"  judged POSITIVE    {pos/tot:.1%}")
    print(f"  UNJUDGED (0 gain)  {1 - judged/tot:.1%}  <- items no ranking can be paid for")
    print(f"  mean judged pool size per query: "
          f"{sum(len(v) for v in qrels.values())/len(qrels):.2f}\n")

    # ---------- 2. do the Meilisearch ingredients predict MEMBERSHIP better? ----------
    print("loading pools and signals ...", flush=True)
    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", 100, set(qids))
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", 100, set(qids))
    gc = np.load(DATA / "cache" / "gate_corpus.npy", mmap_mode="r")
    gq = np.load(DATA / "cache" / "gate_queries.npy", mmap_mode="r")
    W, vocab = build_word_bm25(corpus)

    sample = [q for q in qids if q in qrels][:4000]
    pairs_gate, pairs_bm, pairs_blend, pairs_lowid, pairs_hyb = [], [], [], [], []

    for q in sample:
        pool = list(dict.fromkeys(bm.get(q, []) + dn.get(q, [])))
        if len(pool) < 5:
            continue
        r = qrels[q]
        inset = [1 if i in r else 0 for i in pool]
        if not any(inset) or all(inset):
            continue

        idx = [ipos[i] for i in pool]
        v = np.asarray(gq[qpos[q]], dtype=np.float32)
        M = np.asarray(gc[idx], dtype=np.float32)
        cos = (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-9)

        ti = [vocab[t] for t in tokenize(queries[qpos[q]]["query"]) if t in vocab]
        if ti:
            s = np.asarray(W[ti].sum(axis=0)).ravel()
            bms = np.array([s[ipos[i]] for i in pool], dtype=np.float32)
        else:
            bms = np.zeros(len(pool), dtype=np.float32)
        bmn = bms / (bms.max() + 1e-9)
        cosn = (cos - cos.min()) / (cos.max() - cos.min() + 1e-9)
        hyb = 0.7 * bmn + 0.3 * cosn          # Meilisearch semanticRatio 0.3

        # rank of the live system over its FULL ordering (not the top-10 truncation)
        order = {i: -k for k, i in enumerate(live_full.get(q, []))}
        worst = -len(pool) - 1
        blend = np.array([order.get(i, worst) for i in pool], dtype=np.float32)
        lid = np.array([-int("".join(c for c in i if c.isdigit()) or 0) for i in pool],
                       dtype=np.float64)

        for store, sig in ((pairs_gate, cos), (pairs_bm, bmn), (pairs_hyb, hyb),
                           (pairs_blend, blend), (pairs_lowid, lid)):
            p = [float(sig[k]) for k in range(len(pool)) if inset[k]]
            n = [float(sig[k]) for k in range(len(pool)) if not inset[k]]
            if p and n:
                store.append((p, n))

    print(f"queries used: {len(pairs_gate):,}\n")
    print("WITHIN-QUERY AUC for predicting JUDGED MEMBERSHIP (not relevance)")
    print("  i.e. would Meilisearch have retrieved this item into the judged pool?")
    for nm, pr in (("GATE-AraBert cosine", pairs_gate), ("word BM25", pairs_bm),
                   ("hybrid 0.7*bm25+0.3*gate", pairs_hyb),
                   ("OUR live ranking", pairs_blend), ("low product id", pairs_lowid)):
        print(f"  {nm:<26s}{per_query_auc(pr):.4f}")

    print("\nDECISION RULE: a faithful Meilisearch reimplementation can only help if its")
    print("ingredients predict membership BETTER than our live ranking already does.")
    print("If our ranking is already at or above the hybrid, the idea is dead on arrival.")


if __name__ == "__main__":
    main()
