"""TIER 2: the eps-chain machinery itself, which has never been varied.

Every experiment for two weeks has proposed new FEATURES for the in-group key. The
machinery that decides WHICH candidates get re-sorted together has been fixed since the
day it was written, and it embeds three arbitrary choices:

  C. eps is CONSTANT ACROSS RANK POSITIONS, but nDCG@10 weights rank 1 at 1.000 and rank 9
     at 0.315. A re-sort near the top is worth roughly three times one near the bottom, so
     the band width arguably should not be uniform.

  D. Grouping is TRANSITIVE: items chain while each CONSECUTIVE gap is <= eps, so a chain
     can span an unlimited total score range and drag genuinely-separated candidates into
     one re-sort. Nothing caps chain length.

  F. The alternative to transitive chaining is a WINDOW: group members must be within eps
     of the group's LEADER rather than of their neighbour. Same eps, completely different
     grouping, never tested.

  G. Only re-sort the top-N positions at all, leaving the deep tail on blend order.

The baseline here is rebuilt locally rather than taken from the harness, so the first thing
it does is assert that the local rebuild reproduces the harness run exactly; without that,
any delta measured could be an artifact of the reimplementation.
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
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
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

    def scored_for(q):
        s = []
        for i in h.pools[q]:
            v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                 + BETA_MP * mp(q, i) + BETA_FT * ft(q, i))
            s.append((v, i))
        s.sort(key=lambda x: (-x[0], _id_key(x[1])))
        return s

    def build(qids, mode="chain", cap=0, top_eps=None, top_k=0, resort_top=0):
        out = {}
        for q in qids:
            s = scored_for(q)
            wide = h.conf[q] < CONF_T
            eps = EPS_HI if wide else EPS
            gk = lambda x, qq=q: (-keyf(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for pos, (sc_, iid) in enumerate(s):
                e = eps
                if top_eps is not None and pos < top_k:
                    e = top_eps
                brk = bool(grp) and (grp[-1][0] - sc_) > e
                if mode == "window" and grp and not brk:
                    brk = (grp[0][0] - sc_) > e
                if cap and len(grp) >= cap:
                    brk = True
                if resort_top and pos >= resort_top:
                    brk = True
                if brk:
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

    # regression: the local rebuild must equal the harness run exactly
    ref = h.build_run(key=keyf, bonus_fn=lambda q, i: BETA_MP * mp(q, i) + BETA_FT * ft(q, i),
                      lam=LAM, eps=EPS, eps_hi=EPS_HI, conf_t=CONF_T)
    loc = build(h.qids)
    bad = sum(1 for q in h.qids if ref[q][:10] != loc[q][:10])
    print(f"REGRESSION local rebuild vs harness: {len(h.qids)-bad:,}/{len(h.qids):,} identical")
    if bad:
        print("  *** rebuild differs - every delta below would be an artifact, stopping ***")
        return

    bt, bc = sc(loc, tune), sc(loc, conf)
    print(f"baseline: tune {bt:.6f}  confirm {bc:.6f}\n")

    variants = {
        "D cap chain 3": dict(cap=3),
        "D cap chain 5": dict(cap=5),
        "D cap chain 10": dict(cap=10),
        "D cap chain 20": dict(cap=20),
        "F window grouping": dict(mode="window"),
        "C top5 eps 0.080": dict(top_eps=0.080, top_k=5),
        "C top5 eps 0.015": dict(top_eps=0.015, top_k=5),
        "C top10 eps 0.080": dict(top_eps=0.080, top_k=10),
        "C top10 eps 0.015": dict(top_eps=0.015, top_k=10),
        "G resort top 15 only": dict(resort_top=15),
        "G resort top 30 only": dict(resort_top=30),
        "G resort top 60 only": dict(resort_top=60),
    }

    print(f"  {'variant':<24s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    surv = []
    for nm, kw in variants.items():
        rt, rc = build(tune, **kw), build(conf, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(loc[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        ok = m >= 0.0015 and t > 2
        if ok:
            surv.append(nm)
        print(f"  {nm:<24s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}{'CARRY' if ok else 'drop':>9s}")

    print(f"\n  survivors: {surv if surv else 'none'}")


if __name__ == "__main__":
    main()
