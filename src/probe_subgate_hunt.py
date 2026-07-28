"""Systematic hunt for MORE sub-gate positives, to be stacked.

The overnight lesson was that three changes each measured ~+0.0012 and each individually
discarded were worth +0.0039 together, because they act on different pipeline components.
That converts the search problem: instead of hunting for one change worth +0.0015, hunt for
many changes worth +0.0005 each that touch different machinery.

The acceptance criterion here is therefore NOT the shipping gate. It is CONSISTENCY: a
candidate is retained if its sign is positive on BOTH holdout halves, whatever the size.
Sign agreement across two disjoint 4,006-query halves is what separates a small real effect
from noise, and magnitude is then recovered by stacking.

Candidates deliberately span three components that the current system touches separately:
  BLEND      - additive bonuses for signals that are currently only tie-break levels
  BAND       - eps geometry at ranks the top-5 override does not cover
  KEY        - orderings inside the tie-break that are not yet levels
"""

import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_combined_key import min_span
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT, S_6 = 1e6, 1e2, 1e-2, 1e-5, 1e-8


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
    idfv = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

    print("building harness ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, qbest, qs_ = {}, {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        qbest[q] = max(t, key=idfv) if t else None
        qs_[q] = " ".join(t)
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

    def _pf(q, i):
        return 1.0 if norm[i].startswith(qs_[q]) else 0.0

    def _px(q, i):
        s = min_span(ntok[i], qt[q])
        if s == 0:
            return 0.0
        sl = s - len(qt[q])
        return 3.0 if sl == 0 else (2.0 if sl == 1 else (1.0 if sl <= 3 else 0.5))

    mp, ft, xi = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi)
    lt, pf, px = memo("lt", _lt), memo("pf", _pf), memo("px", _px)
    rrf_max = 4.0 / 61.0

    def build(qids, extra_bonus=None, sixth=None, mid_eps=None, mid_lo=5, mid_hi=15):
        out = {}
        for q in qids:
            def key(qq, i):
                v = (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP + ft(qq, i) * S_FT
                     + lt(qq, i) * S_LT)
                return v + sixth(qq, i) * S_6 if sixth else v
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i))
                if extra_bonus:
                    v += extra_bonus(q, i)
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn, (sc_, iid) in enumerate(scored):
                if posn < TOP_K:
                    e = TOP_EPS
                elif mid_eps is not None and mid_lo <= posn < mid_hi:
                    e = mid_eps
                else:
                    e = e0
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
    print(f"stack3 baseline: tune {bt:.6f}  confirm {bc:.6f}\n")

    cands = {}
    for b in (0.02, 0.04):
        cands[f"BLEND last-token {b:.2f}"] = dict(extra_bonus=lambda q, i, b=b: b * lt(q, i))
        cands[f"BLEND proximity {b:.2f}"] = dict(extra_bonus=lambda q, i, b=b: b * px(q, i))
        cands[f"BLEND prefix {b:.2f}"] = dict(extra_bonus=lambda q, i, b=b: b * pf(q, i))
    for me in (0.060, 0.070):
        cands[f"BAND mid eps {me:.3f} r5-15"] = dict(mid_eps=me)
        cands[f"BAND mid eps {me:.3f} r5-10"] = dict(mid_eps=me, mid_hi=10)
    cands["KEY 6th = prefix"] = dict(sixth=pf)
    cands["KEY 6th = idf-token pos"] = dict(sixth=xi)

    print(f"  {'candidate':<30s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
    keep = []
    for nm, kw in cands.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        both = dt > 0 and m > 0
        if both:
            keep.append(nm)
        print(f"  {nm:<30s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if both else '-':>7s}")

    print(f"\n  consistent-sign candidates for stacking: {keep if keep else 'none'}")


if __name__ == "__main__":
    main()
