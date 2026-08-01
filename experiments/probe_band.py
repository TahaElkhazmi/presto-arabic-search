"""The BAND DEFINITION - which queries get the wide eps - has one untested degree of freedom.

Everything about the band has been tuned except how membership is DECIDED. The confidence
margin is CE top1 minus top3 with a threshold; the top1-top3 form was itself a measured
win over top1-top5, so the shape of this statistic demonstrably matters, and only two
shapes have ever been tried.

Coverage argument, in the terms of the validated screen: 65.3% of queries are in the wide
band, so any change to membership touches a very large share of decisions. That is the
opposite profile to the text predicates that keep failing on coverage below 0.05, which is
why this is worth a pass even though the band constants themselves are settled.

Two families:
  MARGIN SHAPE  - top1-top2, top1-top3 (live), top1-top5, top1 minus the pool mean, and the
                  RELATIVE margin (top1-top3)/top1, which is scale-free and therefore does
                  not conflate "confident" with "high-scoring".
  BAND TIERS    - the band is currently binary. Tier 2 showed that MORE re-sorting is
                  better than less (capping chains cost up to -0.021), which argues for a
                  third, wider tier on the least confident queries rather than a single
                  wide setting for all of them.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    qrels = h.qrels

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

    keyf = lambda q, i: h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT
    rrf_max = 4.0 / 61.0

    sortedce = {q: sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True) for q in h.qids}

    def margin(q, kind):
        v = sortedce[q]
        if len(v) < 6:
            return 1.0
        if kind == "top2":
            return v[0] - v[1]
        if kind == "top3":
            return v[0] - v[2]
        if kind == "top5":
            return v[0] - v[4]
        if kind == "mean":
            return v[0] - sum(v) / len(v)
        if kind == "rel3":
            return (v[0] - v[2]) / v[0] if v[0] > 0 else 1.0
        raise ValueError(kind)

    scored_cache = {}

    def scored_for(q):
        s = scored_cache.get(q)
        if s is None:
            s = [((LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                   + BETA_MP * mp(q, i) + BETA_FT * ft(q, i)), i) for i in h.pools[q]]
            s.sort(key=lambda x: (-x[0], _id_key(x[1])))
            scored_cache[q] = s
        return s

    def build(qids, kind, t1, e_lo, e_hi, t2=None, e_hi2=None):
        out = {}
        for q in qids:
            m = margin(q, kind)
            e = e_lo
            if t2 is not None and m < t2:
                e = e_hi2
            elif m < t1:
                e = e_hi
            gk = lambda x, qq=q: (-keyf(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for sc_, iid in scored_for(q):
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

    base_t = build(tune, "top3", CONF_T, EPS, EPS_HI)
    base_c = build(conf_h, "top3", CONF_T, EPS, EPS_HI)
    bt, bc = sc(base_t, tune), sc(base_c, conf_h)
    print(f"baseline (top1-top3, ct 0.10): tune {bt:.6f}  confirm {bc:.6f}\n")

    variants = {}
    for kind, ts in (("top2", (0.05, 0.10, 0.15)), ("top5", (0.15, 0.22, 0.30)),
                     ("mean", (0.15, 0.25, 0.35)), ("rel3", (0.10, 0.18, 0.28))):
        for t1 in ts:
            variants[f"margin {kind} ct {t1:.2f}"] = dict(kind=kind, t1=t1,
                                                          e_lo=EPS, e_hi=EPS_HI)
    for t2, e2 in ((0.04, 0.075), (0.04, 0.100), (0.02, 0.090)):
        variants[f"3-tier t2 {t2:.2f} eps2 {e2:.3f}"] = dict(
            kind="top3", t1=CONF_T, e_lo=EPS, e_hi=EPS_HI, t2=t2, e_hi2=e2)

    print(f"  {'variant':<30s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    surv = []
    for nm, kw in variants.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base_c[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        ok = m >= 0.0015 and t > 2
        if ok:
            surv.append(nm)
        print(f"  {nm:<30s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}{'CARRY' if ok else 'drop':>9s}")

    print(f"\n  survivors: {surv if surv else 'none'}")


if __name__ == "__main__":
    main()
