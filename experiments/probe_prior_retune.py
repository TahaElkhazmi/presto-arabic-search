"""The RRF prior composition has not been re-tuned since the whole key changed.

k=60 and lexical:dense = 3:1 were fitted on 2026-07-26 and were worth +0.0024 at the time.
Since then the in-band key gained THREE levels (prefix -> match-position -> first-token ->
last-token) and the blend gained THREE additive terms. The prior carries (1-lam) = 0.50 of
the blend score, so it is the largest component never revisited, and the project's most
expensive lesson is precisely that constants fitted in an earlier era encode that era.

Coverage is not in doubt here: the prior enters every candidate's score on every query,
which is the profile the validated screen says to look for. The question is only edge.

Efficiency note: the Harness recomputes priors at construction, so a naive grid would
rebuild the 4.8M-row score cache per cell. It is built ONCE and the priors are recomputed
directly from the stored run files, which makes a 20-cell grid affordable.

Also swept: --lam itself, since re-weighting the prior and re-weighting the blend are the
same decision from two directions.
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
EPS, EPS_HI, CONF_T = 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5
LIVE_K, LIVE_W, LIVE_LAM = 60, 3.0, 0.50

G_K = (30, 60, 100, 160)
G_W = (2.0, 3.0, 4.5, 6.0)
G_LAM = (0.42, 0.46, 0.50, 0.55)


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    ntok = {i: n.split() for i, n in norm.items()}
    import math
    uniq = set(norm.values())
    df = {}
    for n in uniq:
        for t in set(n.split()):
            df[t] = df.get(t, 0) + 1
    N = len(uniq)
    idfv = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

    print("building harness once ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[LIVE_W, 1.0], rrf_k=LIVE_K)
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

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)

    def priors_for(k, w):
        """Recompute the RRF prior from the stored run files - no cache reload."""
        out = {}
        for q in h.qids:
            d = {}
            for run, wt in ((h.run_bm25, w), (h.run_dense, 1.0)):
                for rank, iid in enumerate(run.get(q, [])[:100], start=1):
                    d[iid] = d.get(iid, 0.0) + wt / (k + rank)
            out[q] = d
        return out

    def build(qids, priors, rrf_max, lam):
        out = {}
        for q in qids:
            pr = priors[q]
            key = lambda qq, i: (h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP
                                 + ft(qq, i) * S_FT + lt(qq, i) * S_LT)
            scored = []
            for i in h.pools[q]:
                v = (lam * h.ce[(q, i)] + (1 - lam) * pr.get(i, 0.0) / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i))
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

    live_pr = priors_for(LIVE_K, LIVE_W)
    live_max = (LIVE_W + 1.0) / (LIVE_K + 1)
    bt_r = build(tune, live_pr, live_max, LIVE_LAM)
    bc_r = build(conf_h, live_pr, live_max, LIVE_LAM)
    bt, bc = sc(bt_r, tune), sc(bc_r, conf_h)
    print(f"live prior (k={LIVE_K}, {LIVE_W:.0f}:1, lam {LIVE_LAM}): "
          f"tune {bt:.6f}  confirm {bc:.6f}\n")

    print(f"  {'k':>5}{'ratio':>7}{'lam':>6}{'tune d':>12}{'confirm d':>12}{'t':>7}")
    rows = []
    for k in G_K:
        for w in G_W:
            pr = priors_for(k, w)
            rmax = (w + 1.0) / (k + 1)
            for lam in G_LAM:
                rt = build(tune, pr, rmax, lam)
                dt = sc(rt, tune) - bt
                rows.append((dt, k, w, lam, pr, rmax))
    rows.sort(key=lambda r: -r[0])
    for dt, k, w, lam, pr, rmax in rows[:8]:
        rc = build(conf_h, pr, rmax, lam)
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {k:>5}{w:>7.1f}{lam:>6.2f}{dt:>+12.6f}{m:>+12.6f}"
              f"{m/se if se else 0:>+7.2f}")

    print("\n  only a config positive on BOTH halves is worth a full-benchmark run.")


if __name__ == "__main__":
    main()
