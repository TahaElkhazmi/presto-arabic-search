"""Second stacking round on top of the +0.0039 trio.

Two things are open after the first round.

1. MORE SUB-GATE POSITIVES EXIST. Measured earlier against older baselines and each
   reproducibly positive but short of the gate: query-token coverage (+0.00072),
   all-tokens-present (+0.00066), ordered subsequence (+0.00065), proximity/min-span
   bucket (+0.00093). If the additivity that made round 1 work extends, another level or
   two should still pay. If instead they were restating the levels now in the key, this
   round lands flat - which is itself the useful answer.

2. THE TRIO'S OWN CONSTANTS WERE NEVER SWEPT. beta_idf was copied from beta_ft (0.05),
   and top_eps/top_k_eps were taken from the single coarse grid point that happened to be
   tested (0.080 over 5 ranks). They are now load-bearing and deserve a proper sweep.

Both are screened on the holdout halves; only the confirm half decides.
"""

import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key
from probe_combined_key import min_span

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT, S_LT, S_X6 = 1e6, 1e2, 1e-2, 1e-5, 1e-8
B_IDF, TOP_EPS, TOP_K = 0.05, 0.080, 5


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    ntok = {i: n.split() for i, n in norm.items()}
    uniq = set(norm.values())
    df = {}
    for n in uniq:
        for t in set(n.split()):
            df[t] = df.get(t, 0) + 1
    N = len(uniq)
    idf = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, qbest = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        qbest[q] = max(t, key=idf) if t else None
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

    def _qcov(q, i):
        s = set(ntok[i])
        return float(sum(1 for t in qt[q] if t in s))

    def _prox(q, i):
        s = min_span(ntok[i], qt[q])
        if s == 0:
            return 0.0
        slack = s - len(qt[q])
        return 3.0 if slack == 0 else (2.0 if slack == 1 else (1.0 if slack <= 3 else 0.5))

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    qcov, prox = memo("qc", _qcov), memo("px", _prox)
    rrf_max = 4.0 / 61.0

    def build(qids, sixth=None, b_idf=B_IDF, top_eps=TOP_EPS, top_k=TOP_K):
        out = {}
        for q in qids:
            def key(qq, i):
                v = (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP + ft(qq, i) * S_FT
                     + lt(qq, i) * S_LT)
                return v + sixth(qq, i) * S_X6 if sixth else v
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + b_idf * xi(q, i))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn, (sc_, iid) in enumerate(scored):
                e = top_eps if posn < top_k else e0
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
    print(f"stack3 baseline (local 0.693827): tune {bt:.6f}  confirm {bc:.6f}\n")

    def report(nm, rt, rc):
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        print(f"  {nm:<28s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}"
              f"{'CARRY' if (m >= 0.0015 and t > 2) else 'drop':>9s}")
        return m

    print("A) a SIXTH key level")
    print(f"  {'variant':<28s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    for nm, fn in (("query-token coverage", qcov), ("proximity bucket", prox)):
        report(nm, build(tune, sixth=fn), build(conf_h, sixth=fn))

    print("\nB) sweep the trio's own constants")
    print(f"  {'variant':<28s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    for b in (0.03, 0.08, 0.12):
        report(f"beta_idf {b:.2f}", build(tune, b_idf=b), build(conf_h, b_idf=b))
    for te in (0.060, 0.100, 0.130):
        report(f"top_eps {te:.3f}", build(tune, top_eps=te), build(conf_h, top_eps=te))
    for tk in (3, 8, 12):
        report(f"top_k {tk}", build(tune, top_k=tk), build(conf_h, top_k=tk))


if __name__ == "__main__":
    main()
