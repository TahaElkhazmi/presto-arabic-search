"""Combine the SUB-GATE POSITIVES, which has precedent here.

Five batches of single-change experiments have now come back null, but several changes
were consistently POSITIVE on both holdout halves while falling short of +0.0015 alone:

  idf-token position as a blend bonus   +0.00124 tune / +0.00114 confirm
  wider eps on the top-5 positions      +0.00126 tune / +0.00118 confirm
  last-token position as a 4th level    +0.00131 tune / +0.00119 confirm

Each is reproducible across both halves, which is the signature of a small real effect
rather than noise. The project has already shipped exactly this shape once: on 2026-07-28
the re-tuned constants (+0.0016 arbiter) and the blend bonus (+0.0012 arbiter) each failed
the gate alone and cleared it together (+0.0020).

They also touch three DIFFERENT parts of the pipeline - the blend, the band geometry and
the tie-break key - so there is a real chance they are close to additive rather than
restating one another. That is the hypothesis under test; if they are redundant the
combination will land near the best single one instead of near their sum.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import math

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5
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
    mpc, ftc, xic, ltc = {}, {}, {}, {}

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

    def xi(q, i):
        v = xic.get((q, i))
        if v is None:
            toks, b = ntok[i], qbest[q]
            v = 3.0 / (1.0 + toks.index(b)) if (b is not None and b in toks) else 0.0
            xic[(q, i)] = v
        return v

    def lt(q, i):
        v = ltc.get((q, i))
        if v is None:
            toks, last = ntok[i], (qt[q][-1] if qt[q] else None)
            if last is None or last not in toks:
                v = 0.0
            else:
                k = toks.index(last)
                v = 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)
            ltc[(q, i)] = v
        return v

    rrf_max = 4.0 / 61.0

    def build(qids, use_idf=False, use_topeps=False, use_lasttok=False):
        out = {}
        for q in qids:
            key = (lambda qq, i: h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP
                   + ft(qq, i) * S_FT + (lt(qq, i) * S_LT if use_lasttok else 0.0))
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i)
                     + (B_IDF * xi(q, i) if use_idf else 0.0))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn, (sc_, iid) in enumerate(scored):
                e = TOP_EPS if (use_topeps and posn < TOP_K) else e0
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
    print(f"baseline (current best): tune {bt:.6f}  confirm {bc:.6f}\n")

    combos = {
        "idf + topeps": dict(use_idf=True, use_topeps=True),
        "idf + lasttok": dict(use_idf=True, use_lasttok=True),
        "topeps + lasttok": dict(use_topeps=True, use_lasttok=True),
        "ALL THREE": dict(use_idf=True, use_topeps=True, use_lasttok=True),
    }
    print(f"  {'combination':<22s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    surv = []
    for nm, kw in combos.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        ok = m >= 0.0015 and t > 2
        if ok:
            surv.append(nm)
        print(f"  {nm:<22s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}{'CARRY' if ok else 'drop':>9s}")

    print(f"\n  survivors: {surv if surv else 'none'}")
    print("  additive would be ~+0.0035 for all three; near +0.0012 means redundant.")


if __name__ == "__main__":
    main()
