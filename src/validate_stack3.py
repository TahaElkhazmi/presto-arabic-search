"""Full-benchmark and clean-arbiter validation of the three stacked sub-gate changes.

Each was reproducibly positive on both holdout halves and individually short of the
+0.0015 gate; together they measured +0.00385 on the confirm half (t +3.66), close to the
~+0.0035 that perfect additivity would predict. They act on three different parts of the
pipeline, which is the reason to expect additivity rather than redundancy:

  1. idf-token blend bonus   - the BLEND score
  2. wider eps on top-5      - the BAND GEOMETRY
  3. last-token position     - the TIE-BREAK KEY (a 5th level)

The baseline must reproduce the file currently on the board before any delta is believed.
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
from probe_harness import DIALECT_FILLERS, Harness, _id_key

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5
B_IDF, TOP_EPS, TOP_K = 0.05, 0.080, 5

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
    idf = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

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

    def build(on):
        out = {}
        for q in h.qids:
            key = (lambda qq, i: h.cont3(qq, i) * S_CONT + mp(qq, i) * S_MP
                   + ft(qq, i) * S_FT + (lt(qq, i) * S_LT if on else 0.0))
            scored = []
            for i in h.pools[q]:
                v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
                     + BETA_MP * mp(q, i) + BETA_FT * ft(q, i)
                     + (B_IDF * xi(q, i) if on else 0.0))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            gk = lambda x, qq=q: (-key(qq, x[1]), _id_key(x[1]))
            res, grp = [], []
            for posn, (sc_, iid) in enumerate(scored):
                e = TOP_EPS if (on and posn < TOP_K) else e0
                if grp and (grp[-1][0] - sc_) > e:
                    grp.sort(key=gk)
                    res.extend(i for _, i in grp)
                    grp = []
                grp.append((sc_, iid))
            grp.sort(key=gk)
            res.extend(i for _, i in grp)
            out[q] = res[:10]
        return out

    base, cand = build(False), build(True)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_ftbonus.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs submission_v7synth_ftbonus.csv (local 0.689919): "
          f"{same:,}/{len(live):,}")
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

    print()
    for label, qs in (("tune half (fitted)", tune), ("confirm half", cf),
                      ("ARBITER (9,675)", arb), ("CLEAN arbiter", arb - set(hold)),
                      ("val", val), ("FULL BENCHMARK", set(all_qids))):
        qs = [q for q in sorted(qs, key=int) if q in qrels]
        x = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
        y = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
        d = [b - a for a, b in zip(x, y)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<22s} n={len(qs):>6,}  {sum(y)/len(y):.6f}  "
              f"delta {m:+.6f}  t {m/se if se else 0:+.1f}")

    full = [q for q in all_qids if q in qrels]
    fm = sum(ndcg_at_k(cand[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in full) / len(full)
    ch = sum(1 for q in all_qids if base[q][:10] != cand[q][:10])
    print(f"\n  top-10s changed: {ch:,} ({100*ch/len(all_qids):.1f}%)")
    print(f"  local {0.689919 + fm:.6f}   projected LB ~{0.689919 + fm - 0.001:.4f}")


if __name__ == "__main__":
    main()
