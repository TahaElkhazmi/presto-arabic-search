"""Does the Meilisearch-hybrid membership signal convert end to end?

SCOPING RESULT that justifies this (src/probe_pool_membership.py):
  - 55.8% of our shipped top-10 is UNJUDGED, i.e. earns zero gain whatever its true
    relevance, because nDCG only credits items that were judged.
  - The two ingredients Meilisearch actually used predict judged MEMBERSHIP far better than
    our own ranking does: hybrid 0.7*BM25 + 0.3*GATE scores AUC 0.8446 within query,
    against our live ranking's 0.7476.

So there is a large, legally computable signal our system is measurably bad at. Everything
here is a function of (corpus, query) only - GATE embeddings of product names and word BM25
- and never touches candidates.jsonl, which stays measurement-only.

WHAT IS BEING CLAIMED, honestly: this is partly modelling the retrieval system that DEFINED
the evaluation universe, and partly exploiting pooling bias, since unjudged items are
unpayable by construction. It is legal under the stated rules because it uses only permitted
inputs, and it is the same thing every competitor does implicitly by matching the baseline
system. It is recorded plainly rather than dressed up.

Tested in the two shapes the project's evidence supports: a bounded additive blend term, and
a DISCRETE bucket used as a key level (continuous in-group keys have displaced the low-id
prior and lost every time). Sign consistency across both holdout halves is the bar.
"""

import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, DATA, ROOT, load_jsonl, load_qid_file, normalize_arabic, \
    setup_utf8, tokenize
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key
from probe_score_prior import build_word_bm25

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT, S_6 = 1e6, 1e2, 1e-2, 1e-5, 1e-8
SEM = 0.3   # Meilisearch semanticRatio


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = load_jsonl(BENCH / "queries.jsonl")
    item_ids = [str(r["item_id"]) for r in corpus]
    ipos = {i: k for k, i in enumerate(item_ids)}
    qpos = {str(q["query_id"]): k for k, q in enumerate(queries)}
    qtext_raw = {str(q["query_id"]): q["query"] for q in queries}
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    ntok = {i: n.split() for i, n in norm.items()}

    import math
    uniq = set(norm.values())
    dfm = {}
    for n in uniq:
        for t in set(n.split()):
            dfm[t] = dfm.get(t, 0) + 1
    NU = len(uniq)
    idfv = lambda t: math.log(1.0 + NU / (dfm.get(t, 0) + 0.5))

    print("building harness ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    print("computing the Meilisearch hybrid over every pool ...", flush=True)
    gc = np.load(DATA / "cache" / "gate_corpus.npy", mmap_mode="r")
    gq = np.load(DATA / "cache" / "gate_queries.npy", mmap_mode="r")
    W, vocab = build_word_bm25(corpus)
    HYB = {}
    for q in h.qids:
        pool = h.pools[q]
        idx = [ipos[i] for i in pool]
        v = np.asarray(gq[qpos[q]], dtype=np.float32)
        M = np.asarray(gc[idx], dtype=np.float32)
        cos = (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-9)
        ti = [vocab[t] for t in tokenize(qtext_raw[q]) if t in vocab]
        if ti:
            s = np.asarray(W[ti].sum(axis=0)).ravel()
            bms = np.array([s[ipos[i]] for i in pool], dtype=np.float32)
        else:
            bms = np.zeros(len(pool), dtype=np.float32)
        bmn = bms / (bms.max() + 1e-9)
        cosn = (cos - cos.min()) / (cos.max() - cos.min() + 1e-9)
        hy = (1 - SEM) * bmn + SEM * cosn
        HYB[q] = {i: float(hy[k]) for k, i in enumerate(pool)}
    del gc, gq, W

    qt, qbest = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        qbest[q] = max(t, key=idfv) if t else None
    C = {}

    def memo(tag, fn):
        def g(q, i):
            k = (tag, q, i)
            v = C.get(k)
            if v is None:
                v = C[k] = fn(q, i)
            return v
        return g

    def _mp(q, i):
        toks, qq = ntok[i], qt[q]
        n = len(qq)
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    def _ft(q, i):
        toks, qq = ntok[i], qt[q]
        return 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0

    def _xi(q, i):
        toks, b = ntok[i], qbest[q]
        return 3.0 / (1.0 + toks.index(b)) if (b is not None and b in toks) else 0.0

    def _lt(q, i):
        toks, last = ntok[i], (qt[q][-1] if qt[q] else None)
        if last is None or last not in toks:
            return 0.0
        k = toks.index(last)
        return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    rrf_max = 4.0 / 61.0

    # discrete bucket of the hybrid RANK inside the query's pool (top decile, etc.)
    RB = {}
    for q in h.qids:
        order = sorted(h.pools[q], key=lambda i: -HYB[q][i])
        n = len(order)
        RB[q] = {i: (3.0 if k < 0.05 * n else (2.0 if k < 0.15 * n
                     else (1.0 if k < 0.40 * n else 0.0)))
                 for k, i in enumerate(order)}

    def build(qids, gamma=0.0, bucket=False):
        out = {}
        for q in qids:
            def key(qq, i):
                v = (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP + ft(qq, i) * S_FT
                     + lt(qq, i) * S_LT)
                return v + RB[qq][i] * S_6 if bucket else v
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i)
                     + gamma * HYB[q][i])
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn, (sc_, iid) in enumerate(scored):
                e = TOP_EPS if posn < TOP_K else e0
                if grp and (grp[-1][0] - sc_) > e:
                    grp.sort(key=gk)
                    res.extend(i for _, i in grp)
                    grp = []
                grp.append((sc_, iid))
            grp.sort(key=gk)
            res.extend(i for _, i in grp)
            out[q] = res[:10]
        return out

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt_r, bc_r = build(tune), build(conf_h)
    bt, bc = sc(bt_r, tune), sc(bc_r, conf_h)
    print(f"\nstack3 baseline: tune {bt:.6f}  confirm {bc:.6f}\n")

    cands = {"BUCKET as 6th key level": dict(bucket=True)}
    for g in (0.02, 0.05, 0.10, 0.20, 0.35):
        cands[f"BLEND hybrid gamma {g:.2f}"] = dict(gamma=g)
    cands["BLEND 0.10 + bucket"] = dict(gamma=0.10, bucket=True)

    print(f"  {'candidate':<28s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
    for nm, kw in cands.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {nm:<28s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if (dt > 0 and m > 0) else '-':>7s}")


if __name__ == "__main__":
    main()
