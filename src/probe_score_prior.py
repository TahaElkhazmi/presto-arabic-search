"""The last untested structural axis: the prior throws away BM25 SCORE MAGNITUDE.

Reciprocal Rank Fusion uses only RANKS. A document that BM25 scores far above everything
else and one that barely edges out its neighbour contribute identically at the same rank.
That robustness is why RRF is a good default, and it is also information deliberately
discarded - on a catalog of short names, where 36% of titles are exact duplicates, the gap
between rank 1 and rank 2 is sometimes enormous and sometimes nothing.

The prior carries (1-lam) = 0.50 of the blend and re-tuning its composition (k, ratio) was
worth +0.0024 once, so this is a high-leverage component. Its FORM has never been varied.

Cheap by construction: only the LEXICAL half is converted, because dense query embeddings
are not cached and would need a GPU pass, while BM25 is recomputable on CPU in seconds. The
lexical voter also carries the 3:1 majority of the prior weight, so it is the half worth
testing first.

Interpolation rather than replacement, so alpha=0 is EXACTLY the shipped system and the
regression check is built in:

    lex_term(alpha) = (1-alpha) * 1/(k+rank) + alpha * (score/score_max) * 1/(k+1)

The second form is scaled by 1/(k+1) so that a query's best document contributes the same
maximum under either form, which keeps rrf_max and therefore the blend scale unchanged.
"""

import random
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from build_charbm25_voter import K1, B
from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8, \
    tokenize
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key, stream_topk
from scipy.sparse import csr_matrix

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2
RRF_K, W_LEX = 60, 3.0
ALPHAS = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)


def build_word_bm25(corpus):
    vocab, rows, cols, tfs = {}, [], [], []
    dl = np.zeros(len(corpus))
    for d, row in enumerate(corpus):
        c = {}
        for t in tokenize(row["name"]):
            c[t] = c.get(t, 0) + 1
        dl[d] = sum(c.values())
        for t, n in c.items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(j); cols.append(d); tfs.append(n)
    tf = np.asarray(tfs, dtype=np.float64)
    rows = np.asarray(rows); cols = np.asarray(cols)
    dfv = np.bincount(rows, minlength=len(vocab)).astype(np.float64)
    idf = np.log(1.0 + (len(corpus) - dfv + 0.5) / (dfv + 0.5))
    avgdl = dl.mean()
    w = idf[rows] * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl[cols] / avgdl))
    return csr_matrix((w, (rows, cols)), shape=(len(vocab), len(corpus))), vocab


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    item_ids = [str(r["item_id"]) for r in corpus]
    pos = {iid: k for k, iid in enumerate(item_ids)}
    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split() for r in corpus}

    t0 = time.time()
    W, vocab = build_word_bm25(corpus)
    print(f"word BM25 index: {len(vocab):,} terms, {W.nnz:,} nnz, {time.time()-t0:.1f}s")

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[W_LEX, 1.0], rrf_k=RRF_K)
    qrels = h.qrels
    bm_rank = stream_topk(ROOT / "runs" / "bm25_all.jsonl", 100, set(hold))

    # REGRESSION: the recomputed BM25 must reproduce the shipped run's ordering.
    agree = tot = 0
    for q in hold[:300]:
        idx = [vocab[t] for t in tokenize(h.queries[q]) if t in vocab]
        if not idx:
            continue
        s = np.asarray(W[idx].sum(axis=0)).ravel()
        top = np.argpartition(-s, 200)[:200]
        top = top[np.lexsort((top, -s[top]))][:10]
        mine = [item_ids[i] for i in top]
        agree += sum(1 for a, b in zip(mine, bm_rank.get(q, [])[:10]) if a == b)
        tot += 10
    print(f"BM25 reimplementation agrees with the shipped run on "
          f"{agree}/{tot} top-10 slots ({agree/tot:.1%})\n")

    qt, mpc, ftc, bms = {}, {}, {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

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

    print("scoring BM25 for pool members ...", flush=True)
    for q in h.qids:
        idx = [vocab[t] for t in tokenize(h.queries[q]) if t in vocab]
        if not idx:
            bms[q] = {}
            continue
        s = np.asarray(W[idx].sum(axis=0)).ravel()
        d = {i: float(s[pos[i]]) for i in h.pools[q] if i in pos}
        mx = max(d.values()) if d else 0.0
        bms[q] = {i: (v / mx if mx > 0 else 0.0) for i, v in d.items()}

    keyf = lambda q, i: h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT
    rrf_max = (W_LEX + 1.0) / (RRF_K + 1)
    lex_rank = {q: {iid: r for r, iid in enumerate(bm_rank.get(q, []), start=1)}
                for q in h.qids}

    def build(qids, alpha):
        out = {}
        for q in qids:
            prior = h.priors[q]
            lr, bs = lex_rank[q], bms[q]
            res, grp = [], []
            scored = []
            for i in h.pools[q]:
                p = prior[i]
                if alpha and i in lr:
                    old = W_LEX / (RRF_K + lr[i])
                    new = W_LEX * bs.get(i, 0.0) / (RRF_K + 1)
                    p = p - old + ((1 - alpha) * old + alpha * new)
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * p / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-keyf(qq, x[1]), _id_key(x[1]))
            for sc_, iid in scored:
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

    base_t, base_c = build(tune, 0.0), build(conf_h, 0.0)
    bt, bc = sc(base_t, tune), sc(base_c, conf_h)
    print(f"baseline (pure rank RRF): tune {bt:.6f}  confirm {bc:.6f}\n")
    print(f"  {'alpha':>7}{'tune d':>12}{'confirm d':>12}{'t':>8}")
    for a in ALPHAS:
        if a == 0.0:
            print(f"  {a:>7.2f}{0.0:>+12.6f}{0.0:>+12.6f}{0.0:>+8.2f}")
            continue
        rt, rc = build(tune, a), build(conf_h, a)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base_c[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {a:>7.2f}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}")


if __name__ == "__main__":
    main()
