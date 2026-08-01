"""The two Meilisearch match levels our position features never test.

Meilisearch built the judged pools, and its docs describe exactness as a graded notion:
a query term can match a word EXACTLY, match as a PREFIX of a longer word, or match with a
TYPO. Its ranking is a bucket sort where each rule is a tiebreaker inside the previous
rule's groups - which is the same architecture our lexicographic key independently arrived
at, so the analogy is structural rather than superficial.

Our match_position and first_token_position both use TOKEN EQUALITY
(`toks[k:k+n] == qq`), so they capture only the exact level and score 0 on the other two.
Every title that matches the query as a prefix-of-word or with a single typo therefore
falls straight through to the weaker levels of the key. Arabic makes this concrete: a query
"حذاء" against a catalog form "حذاءك" is a prefix match, and dialect spelling variation
produces edit-distance-1 pairs routinely.

Three new position features, all the same 3/(1+k) shape as the shipped ones so they inherit
the threshold-free property:
  PREFIX-OF-WORD  - first query token is a prefix of some title token
  CONTAINED       - first query token appears anywhere inside some title token (covers
                    Arabic proclitics attached to the front, which stemming failed to fix
                    destructively but which a POSITION feature can exploit non-destructively)
  TYPO-1          - first query token is within edit distance 1 of some title token

Each is tested as a blend bonus and as a key level, and consistency of sign across both
holdout halves is the retention criterion, since stacking is what recovers magnitude.
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


def ed1(a: str, b: str) -> bool:
    """True if edit distance(a, b) <= 1. Bounded and early-exiting, not a full DP."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return True
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] != b[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


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
        """3/(1+k) for the earliest title token satisfying pred against the query's first
        token; 0 if none. Returns 0 when an EXACT match exists, so the feature is
        orthogonal to first_token_position rather than a superset of it."""
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

    pfx = memo("pfx", lambda q, i: _first_where(q, i, lambda a, t: t.startswith(a) and t != a))
    con = memo("con", lambda q, i: _first_where(q, i, lambda a, t: len(a) >= 3 and a in t))
    typ = memo("typ", lambda q, i: _first_where(q, i, lambda a, t: len(a) >= 4 and ed1(a, t)))
    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    rrf_max = 4.0 / 61.0

    # how often do these fire where the exact test does not?
    for nm, fn in (("prefix-of-word", pfx), ("contained", con), ("typo-1", typ)):
        hits = tot = 0
        for q in hold[:1500]:
            for i in h.pools[q]:
                tot += 1
                if fn(q, i) > 0:
                    hits += 1
        print(f"  {nm:<16s} fires on {hits/tot:.2%} of candidates where exact match fails")
    print()

    def build(qids, bonus=None, sixth=None):
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

    cands = {}
    for nm, fn in (("prefix-of-word", pfx), ("contained", con), ("typo-1", typ)):
        cands[f"KEY 6th = {nm}"] = dict(sixth=fn)
        for b in (0.02, 0.05):
            cands[f"BLEND {nm} {b:.2f}"] = dict(bonus=lambda q, i, f=fn, b=b: b * f(q, i))

    print(f"  {'candidate':<28s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
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
            keep.append((nm, m))
        print(f"  {nm:<28s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if both else '-':>7s}")

    print(f"\n  consistent-sign: {[n for n, _ in keep] if keep else 'none'}")


if __name__ == "__main__":
    main()
