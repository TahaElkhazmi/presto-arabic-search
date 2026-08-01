"""Meilisearch's POSITION rule, reimplemented from its source, as an in-band signal.

A source-level reading of the engine (crates/milli/src/search/new/ranking_rule_graph/
position/) shows the attribute rule computes something we do NOT have. Our match_position
fires only on a CONTIGUOUS occurrence of the whole query, and first_token_position looks
only at the query's first token. Meilisearch instead scores EVERY query term independently
and sums an alignment cost between where that term sits in the document and where it sits
in the query:

    cost = sum over query terms j of
           min over occurrences k of term j in the title of
           cost_from_distance(|bucketed_position(k) - j|)

with a bucketed document position (exact below 16, then coarse powers of two) and a
step-function cost curve. So a title whose words line up with the query in order scores 0,
and misalignment is penalised term by term rather than all-or-nothing.

That is a genuine generalisation of both shipped position features, and it has the profile
the validated screen asks for: it is INTEGER-valued so it partitions rather than displacing
the low-id prior, and it takes a value for essentially every candidate, so coverage is high
unlike the 1-3% exactness variants that just failed.

Two corrections this reimplementation encodes, from the same source reading:
  - the query-side index is the term's position IN THE QUERY, not a document offset;
  - a term absent from the title takes the maximum cost rather than being skipped, which is
    what makes the sum comparable across candidates.

Legal: a function of (query, product name) only. Nothing here reads candidates.jsonl
except the evaluator.
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

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT, S_6 = 1e6, 1e2, 1e-2, 1e-5, 1e-8
MAXC = 10


def bucketed_position(p: int) -> int:
    if p < 16:
        return p
    if p < 24:
        return 24
    return 2 ** math.ceil(math.log2(p))


def cost_from_distance(d: int) -> int:
    if d == 0:
        return 0
    if d == 1:
        return 1
    for lim, c in ((4, 2), (7, 3), (11, 4), (16, 5), (24, 6), (64, 7), (256, 8), (1024, 9)):
        if d <= lim:
            return c
    return MAXC


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

    def _pos(q, i):
        """Meilisearch Position cost, negated so that higher = better."""
        toks, qq = ntok[i], qt[q]
        if not qq:
            return 0.0
        where = {}
        for k, t in enumerate(toks):
            where.setdefault(t, []).append(k)
        total = 0
        for j, t in enumerate(qq):
            occ = where.get(t)
            if not occ:
                total += MAXC
            else:
                total += min(cost_from_distance(abs(bucketed_position(k) - j)) for k in occ)
        return -float(total)

    mp, ft, xi = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi)
    lt, pos = memo("lt", _lt), memo("pos", _pos)
    rrf_max = 4.0 / 61.0

    # normalised version for use as a bounded blend term
    def posn(q, i):
        return (pos(q, i) + MAXC * len(qt[q])) / max(MAXC * len(qt[q]), 1)

    def build(qids, gamma=0.0, level=False):
        out = {}
        for q in qids:
            def key(qq, i):
                v = (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP + ft(qq, i) * S_FT
                     + lt(qq, i) * S_LT)
                return v + pos(qq, i) * S_6 if level else v
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i)
                     + gamma * posn(q, i))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn_, (sc_, iid) in enumerate(scored):
                e = TOP_EPS if posn_ < TOP_K else e0
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

    cands = {"POS as 5th key level": dict(level=True)}
    for g in (0.02, 0.05, 0.10, 0.20):
        cands[f"POS blend gamma {g:.2f}"] = dict(gamma=g)
    cands["POS level + blend 0.05"] = dict(level=True, gamma=0.05)

    print(f"  {'candidate':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
    for nm, kw in cands.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {nm:<26s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if (dt > 0 and m > 0) else '-':>7s}")


if __name__ == "__main__":
    main()
