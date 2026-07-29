"""Stack the three Meilisearch match levels, which were each consistent but tiny.

prefix-of-word, contained-in-word and typo-1 each came back positive on BOTH holdout halves
(best: prefix as a blend bonus, +0.00066 tune / +0.00063 confirm, t +2.08) and each fires on
only 1-3% of candidates, so none can clear the gate alone. That is exactly the profile the
first additive stack had.

The caution is that these three OVERLAP by construction: "contained" (`a in t`) is a
superset of "prefix-of-word" (`t.startswith(a)`), and a typo-1 neighbour can also be a
prefix. So unlike the first trio - which acted on the blend, the band and the key - these
act on the same component through overlapping predicates, and redundancy is the expected
failure mode rather than additivity. Combinations that exclude the subset relation
(prefix+typo) are therefore tested separately from the full three.
"""

import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_exactness_levels import ed1
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5


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

    def _first_where(q, i, pred):
        qq = qt[q]
        if not qq:
            return 0.0
        a, toks = qq[0], ntok[i]
        if a in toks:
            return 0.0
        for k, t in enumerate(toks):
            if pred(a, t):
                return 3.0 / (1.0 + k)
        return 0.0

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    pfx = memo("pfx", lambda q, i: _first_where(q, i, lambda a, t: t.startswith(a) and t != a))
    con = memo("con", lambda q, i: _first_where(q, i, lambda a, t: len(a) >= 3 and a in t))
    typ = memo("typ", lambda q, i: _first_where(q, i, lambda a, t: len(a) >= 4 and ed1(a, t)))
    rrf_max = 4.0 / 61.0

    def build(qids, bonus=None):
        out = {}
        for q in qids:
            key = lambda qq, i: (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP
                                 + ft(qq, i) * S_FT + lt(qq, i) * S_LT)
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i))
                if bonus:
                    v += bonus(q, i)
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
    print(f"stack3 baseline: tune {bt:.6f}  confirm {bc:.6f}\n")

    cands = {
        "P prefix .05": lambda q, i: 0.05 * pfx(q, i),
        "P prefix .08": lambda q, i: 0.08 * pfx(q, i),
        "P+T prefix+typo .05": lambda q, i: 0.05 * (pfx(q, i) + typ(q, i)),
        "P+T prefix+typo .08": lambda q, i: 0.08 * (pfx(q, i) + typ(q, i)),
        "ALL three .05": lambda q, i: 0.05 * (pfx(q, i) + con(q, i) + typ(q, i)),
        "C contained .08": lambda q, i: 0.08 * con(q, i),
    }
    print(f"  {'candidate':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
    for nm, fn in cands.items():
        rt, rc = build(tune, fn), build(conf_h, fn)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {nm:<26s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if (dt > 0 and m > 0) else '-':>7s}")


if __name__ == "__main__":
    main()
