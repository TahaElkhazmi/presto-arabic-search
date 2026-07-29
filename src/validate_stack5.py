"""Full-slice check for the two remaining consistent-sign additions on top of stack3.

  prefix-of-word + typo-1 as a blend bonus at 0.05  (+0.00086 tune / +0.00093 confirm, t 2.50)
  proximity bucket as a blend bonus at 0.04         (+0.00043 tune / +0.00022 confirm)

Both are sub-gate on the confirm half. They are measured here anyway because the decisive
slice is the CLEAN ARBITER - the 2,784 queries no campaign has touched - and on the previous
stack that slice came in HIGHER than the confirm half (+0.00446 against +0.00385). A change
worth +0.0009 on confirm can therefore still land near the +0.0015 gate where it counts, and
guessing either way is worse than measuring.

Ships only if the clean arbiter clears +0.0015 with t > 2. Otherwise it is recorded as
another documented sub-gate positive and the shipped file stays as it is.
"""

import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_combined_key import min_span
from probe_exactness_levels import ed1
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5
B_PT, B_PX = 0.05, 0.04

SPENT = ("dev_rand2000", "dev_unseen2000", "dev_probe2000", "dev_recal2000",
         "dev_ens2000", "dev_final2000", "dev_listwise200", "probe_filler800",
         "probe_band600", "probe_judge150")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
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

    print("building harness over the full benchmark ...", flush=True)
    h = Harness(all_qids, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
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

    def _fw(q, i, pred):
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

    def _px(q, i):
        s = min_span(ntok[i], qt[q])
        if s == 0:
            return 0.0
        sl = s - len(qt[q])
        return 3.0 if sl == 0 else (2.0 if sl == 1 else (1.0 if sl <= 3 else 0.5))

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    pfx = memo("pfx", lambda q, i: _fw(q, i, lambda a, t: t.startswith(a) and t != a))
    typ = memo("typ", lambda q, i: _fw(q, i, lambda a, t: len(a) >= 4 and ed1(a, t)))
    px = memo("px", _px)
    rrf_max = 4.0 / 61.0

    def build(mode):
        out = {}
        for q in h.qids:
            key = lambda qq, i: (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP
                                 + ft(qq, i) * S_FT + lt(qq, i) * S_LT)
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i))
                if mode in ("pt", "both"):
                    v += B_PT * (pfx(q, i) + typ(q, i))
                if mode in ("px", "both"):
                    v += B_PX * px(q, i)
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

    base = build("none")
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs the file on the board (LB 0.69362): {same:,}/{len(live):,}")
    if same != len(live):
        print("  *** baseline does not reproduce - stop ***")
        return

    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")
    shuf = sorted(hold, key=int)
    random.Random(77).shuffle(shuf)
    tune, cf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in SPENT:
        p = ROOT / "splits" / f"{f}.txt"
        if p.exists():
            used |= load_qid_file(p)
    arb = set(all_qids) - used
    slices = [("tune half (fitted)", tune), ("confirm half", cf),
              ("ARBITER (9,675)", arb), ("CLEAN arbiter", arb - set(hold)),
              ("val", val), ("FULL BENCHMARK", set(all_qids))]

    for mode, label in (("pt", "prefix+typo 0.05"), ("both", "prefix+typo + proximity")):
        cand = build(mode)
        print(f"\n=== {label} ===")
        for lab, qs in slices:
            qs = [q for q in sorted(qs, key=int) if q in qrels]
            x = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
            y = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
            d = [b - a for a, b in zip(x, y)]
            m = sum(d) / len(d)
            se = st.pstdev(d) / len(d) ** 0.5
            print(f"  {lab:<22s} n={len(qs):>6,}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")
        full = [q for q in all_qids if q in qrels]
        fm = sum(ndcg_at_k(cand[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
                 for q in full) / len(full)
        print(f"  -> local {0.693827 + fm:.6f}  projected LB ~{0.693827 + fm - 0.001:.4f}")

    print("\n  gate: +0.0015 on the CLEAN arbiter with t > 2, else it does not ship.")


if __name__ == "__main__":
    main()
