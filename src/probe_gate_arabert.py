"""Does GATE-AraBert-v1 carry signal where our stack has NONE?

The zero cohort fails because dozens of pool items tie at maximum char-3gram
containment, so our in-band key is constant across the head and ordering collapses to
ascending product id among textually identical candidates. Any signal that separates
items INSIDE those tie groups is worth more than a signal that is merely good on
average, because on average we are already strong.

So the headline metric here is not pool-wide AUC. It is AUC computed WITHIN the
maximum-containment tie group, which is the exact decision our pipeline currently
makes at random-ish. Baselines on the same groups: the low product id prior (the
incumbent key) and the cross-encoder itself.

GATE-AraBert-v1 is the encoder the benchmark's own Meilisearch baseline used, and it
is from a different lineage than our all-Qwen stack, which is the property the five
failed ensembles were missing.

Embeddings are cached, so re-running any downstream experiment is free.
"""

import argparse
import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import BENCH, DATA, MODELS, ROOT, load_jsonl, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
MODEL = MODELS / "gate-arabert-v1"
CACHE = DATA / "cache"


def embed(texts, tag, batch=256, seq=64):
    """Encode with caching keyed by tag; returns L2-normalized float32."""
    path = CACHE / f"gate_{tag}.npy"
    if path.exists():
        v = np.load(path)
        if len(v) == len(texts):
            print(f"  cache hit {path.name}: {v.shape}")
            return v
    from sentence_transformers import SentenceTransformer
    import torch

    m = SentenceTransformer(str(MODEL), device="cuda")
    m.max_seq_length = seq
    print(f"  encoding {len(texts):,} texts for {tag} ...", flush=True)
    v = m.encode(texts, batch_size=batch, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=True)
    v = v.astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, v)
    del m
    torch.cuda.empty_cache()
    return v


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random positive scores above a random negative), ties at 0.5."""
    if not pos or not neg:
        return float("nan")
    ns = sorted(neg)
    conc = ties = 0
    for p in pos:
        lo = bisect.bisect_left(ns, p)
        hi = bisect.bisect_right(ns, p)
        conc += lo
        ties += hi - lo
    return (conc + 0.5 * ties) / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-zero", type=int, default=1200)
    p.add_argument("--n-good", type=int, default=1200)
    args = p.parse_args()

    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    import random
    rng = random.Random(21)
    zeros = sorted(rng.sample([q for q, v in nd.items() if v == 0.0], args.n_zero), key=int)
    goods = sorted(rng.sample([q for q, v in nd.items() if v >= 0.9], args.n_good), key=int)
    qids = sorted(set(zeros) | set(goods), key=int)

    print("building pools (0 GPU) ...", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    item_ids = [str(r["item_id"]) for r in corpus]
    idx = {iid: i for i, iid in enumerate(item_ids)}
    print("embedding corpus ...", flush=True)
    C = embed([r["name"] for r in corpus], "corpus")

    print("embedding queries ...", flush=True)
    all_q = [str(x["query_id"]) for x in load_jsonl(BENCH / "queries.jsonl")]
    qtext = {str(x["query_id"]): x["query"] for x in load_jsonl(BENCH / "queries.jsonl")}
    Q = embed([qtext[q] for q in all_q], "queries")
    qidx = {q: i for i, q in enumerate(all_q)}

    def gate(q, iid):
        return float(Q[qidx[q]] @ C[idx[iid]])

    def lowid(q, iid):
        d = "".join(ch for ch in iid if ch.isdigit())
        return -int(d) if d else 0

    def ce(q, iid):
        return h.ce[(q, iid)]

    feats = (("GATE-AraBert", gate), ("low-id (incumbent)", lowid), ("our CE", ce))

    for label, qs in (("ZERO cohort", zeros), ("high-scoring", goods)):
        print(f"\n########## {label} (n={len(qs)}) ##########")

        print("\n  A. WHOLE POOL")
        for name, fn in feats:
            pos, neg = [], []
            for q in qs:
                for iid in h.pools[q]:
                    (pos if qrels[q].get(iid, 0) > 0 else neg).append(fn(q, iid))
            print(f"    {name:<22s} AUC {auc(pos, neg):.4f}")

        print("\n  B. INSIDE THE MAX-CONTAINMENT TIE GROUP  <-- where we have no signal")
        sizes = []
        for name, fn in feats:
            pos, neg = [], []
            for q in qs:
                pool = h.pools[q]
                cs = [h.cont3(q, i) for i in pool]
                mx = max(cs)
                grp = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
                if len(grp) < 2:
                    continue
                if name == feats[0][0]:
                    sizes.append(len(grp))
                gp = [fn(q, i) for i in grp if qrels[q].get(i, 0) > 0]
                gn = [fn(q, i) for i in grp if qrels[q].get(i, 0) <= 0]
                if gp and gn:
                    pos.extend(gp)
                    neg.extend(gn)
            print(f"    {name:<22s} AUC {auc(pos, neg):.4f}")
        if sizes:
            print(f"    (mean tie-group size {sum(sizes) / len(sizes):.1f}, "
                  f"{len(sizes)} queries with a usable group)")

    print("\nREADING: pool-wide AUC near 0.5 would make GATE useless. What decides")
    print("whether it ships is column B: if GATE beats low-id INSIDE the tie group,")
    print("it supplies information our pipeline provably lacks at that exact point.")


if __name__ == "__main__":
    main()
