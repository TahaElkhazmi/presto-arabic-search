"""LEAD 1: re-test the CE ensembles under the CURRENT key, not the weak-key era one.

All five prior ensemble attempts chose weight 0, but every one was measured before
prefix / match-position / first-token / last-token and the three blend bonuses existed.
The blend's reliance on the cross-encoder changed materially in that time (lam 0.60 ->
0.50, temperature 2.6 -> 3.2), so "weight 0" was measured under conditions that no
longer hold. An ensemble's value is conditional on how much work the CE is doing.

Only three caches cover the holdout split at full-benchmark scale (v2, v6warm, v7synth),
so those are the testable mixtures; the bge / v3 / v3.1 / v8graded caches cover 0% of it
and would need GPU time to revisit.

The mixture is formed AFTER each model's own temperature rescale, in probability space
(what the old ensemble tool did) or in logit space, then the full stack3 transform runs
on top of it unchanged. conf is recomputed from the mixed scores, since the band
definition is a function of the CE distribution the mixture changes.

Baseline must reproduce runs/submission_v7synth_stack3.csv on all 8,012 holdout queries
before any delta is believed.
"""

import argparse
import math
import random
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, DATA, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness, _id_key, rescale

setup_utf8()

STEM = "rerank_qwen3-reranker-arabic-ecom-rerank_{}_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT, S_LT = 1e6, 1e2, 1e-2, 1e-5
B_IDF, TOP_EPS, TOP_K = 0.05, 0.080, 5
RRF_MAX = 4.0 / 61.0


def logit(s: float) -> float:
    s = min(max(s, 1e-12), 1.0 - 1e-12)
    return max(-30.0, min(30.0, math.log(s / (1.0 - s))))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Stack3:
    """The live stack3 transform, with the CE score injectable.

    cache_stem/T select which model's scores seed h.ce; every caller that wants a
    different CE (an ensemble mixture, a newly trained model) passes its own dict to
    build() instead, so the transform itself exists in exactly one place.
    """

    def __init__(self, qids, cache_stem=STEM.format("v7synth"), T=T_LIVE):
        corpus = load_jsonl(BENCH / "corpus.jsonl")
        norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
        self.ntok = {i: n.split() for i, n in norm.items()}
        df = {}
        uniq = set(norm.values())
        for n in uniq:
            for t in set(n.split()):
                df[t] = df.get(t, 0) + 1
        N = len(uniq)
        self.idf = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

        self.h = Harness(qids, T=T, cache_stem=cache_stem,
                         weights=[3.0, 1.0], rrf_k=60)
        h = self.h
        self.qt, self.qbest = {}, {}
        for q in h.qids:
            t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
            t = t or normalize_arabic(h.queries[q]).split()
            self.qt[q] = t
            self.qbest[q] = max(t, key=self.idf) if t else None
        self.mpc, self.ftc, self.xic, self.ltc = {}, {}, {}, {}

    def mp(self, q, i):
        v = self.mpc.get((q, i))
        if v is None:
            toks, qq = self.ntok[i], self.qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            self.mpc[(q, i)] = v
        return v

    def ft(self, q, i):
        v = self.ftc.get((q, i))
        if v is None:
            toks, qq = self.ntok[i], self.qt[q]
            v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
            self.ftc[(q, i)] = v
        return v

    def xi(self, q, i):
        v = self.xic.get((q, i))
        if v is None:
            toks, b = self.ntok[i], self.qbest[q]
            v = 3.0 / (1.0 + toks.index(b)) if (b is not None and b in toks) else 0.0
            self.xic[(q, i)] = v
        return v

    def lt(self, q, i):
        v = self.ltc.get((q, i))
        if v is None:
            toks = self.ntok[i]
            last = self.qt[q][-1] if self.qt[q] else None
            if last is None or last not in toks:
                v = 0.0
            else:
                k = toks.index(last)
                v = 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)
            self.ltc[(q, i)] = v
        return v

    def build(self, ce, lam=LAM, priors=None, prior_max=RRF_MAX):
        """ce: dict[(qid, iid)] -> calibrated score in [0,1].

        priors/prior_max override the RRF prior so alternative fusion functions
        (convex combination, rank-biased centroids) can be swapped in with the rest
        of the transform held fixed.
        """
        h = self.h
        priors = h.priors if priors is None else priors
        conf = {}
        for q in h.qids:
            v = sorted((ce[(q, i)] for i in h.pools[q]), reverse=True)
            conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
        out = {}
        for q in h.qids:
            key = (lambda qq, i: h.cont3(qq, i) * S_CONT + self.mp(qq, i) * S_MP
                   + self.ft(qq, i) * S_FT + self.lt(qq, i) * S_LT)
            scored = []
            for i in h.pools[q]:
                v = (lam * ce[(q, i)] + (1 - lam) * priors[q].get(i, 0.0) / prior_max
                     + BETA_MP * self.mp(q, i) + BETA_FT * self.ft(q, i)
                     + B_IDF * self.xi(q, i))
                scored.append((v, i))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e0 = EPS_HI if conf[q] < CONF_T else EPS
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


def load_raw(stem, pairs):
    """Raw (uncalibrated) cache scores restricted to the pool pairs we need."""
    qids = {q for q, _ in pairs}
    d = pd.read_parquet(DATA / "cache" / f"{stem}.parquet")
    d = d[d["qid"].isin(qids)]
    raw = {(q, i): s for q, i, s in zip(d["qid"], d["iid"], d["score"])}
    miss = sum(1 for p in pairs if p not in raw)
    return raw, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partners", nargs="+", default=["v6warm", "v2"])
    ap.add_argument("--weights", default="0.10,0.20,0.30,0.40,0.50")
    ap.add_argument("--temps", default="2.3,3.2")
    ap.add_argument("--logit-mean", action="store_true")
    ap.add_argument("--three-way", action="store_true")
    args = ap.parse_args()

    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")
    qids = sorted(hold, key=int)
    print(f"building stack3 harness over {len(qids):,} holdout queries ...", flush=True)
    S = Stack3(qids)
    h = S.h
    pairs = [(q, i) for q in h.qids for i in h.pools[q]]
    print(f"  {len(pairs):,} pool pairs", flush=True)

    base = S.build(h.ce)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    same = sum(1 for q in h.qids if live.get(q, [])[:10] == base[q][:10])
    print(f"REGRESSION vs submission_v7synth_stack3.csv: {same:,}/{len(h.qids):,}")
    if same != len(h.qids):
        print("  *** baseline does not reproduce - stop ***")
        return

    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune, cf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])
    qrels = h.qrels

    def slices(run):
        out = []
        for qs in (tune, cf):
            qs = [q for q in h.qids if q in qs and q in qrels]
            out.append([ndcg_at_k(run[q], qrels[q], 10) for q in qs])
        return out

    b_tune, b_cf = slices(base)
    print(f"baseline  tune {sum(b_tune)/len(b_tune):.6f}  confirm {sum(b_cf)/len(b_cf):.6f}\n")

    raws = {}
    for p in args.partners:
        raw, miss = load_raw(STEM.format(p), pairs)
        print(f"partner {p}: {len(raw):,} scores on pool pairs, {miss:,} missing")
        if miss:
            print(f"  *** {p} does not cover the pools - skipping ***")
            continue
        raws[p] = raw

    ws = [float(x) for x in args.weights.split(",")]
    ts = [float(x) for x in args.temps.split(",")]
    mode = "logit" if args.logit_mean else "prob"
    print(f"\nmix space: {mode}   (w = weight on the PARTNER, 0 = live system)")
    print(f"{'partner':<10s}{'T_p':>6s}{'w':>7s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")

    results = []
    for p, raw in raws.items():
        for T in ts:
            cal = {k: rescale(v, T) for k, v in raw.items() if k in h.ce}
            for w in ws:
                if args.logit_mean:
                    ce = {k: sigmoid((1 - w) * logit(h.ce[k]) + w * logit(cal[k]))
                          for k in h.ce}
                else:
                    ce = {k: (1 - w) * h.ce[k] + w * cal[k] for k in h.ce}
                run = S.build(ce)
                c_tune, c_cf = slices(run)
                dt = sum(y - x for x, y in zip(b_tune, c_tune)) / len(b_tune)
                dd = [y - x for x, y in zip(b_cf, c_cf)]
                dc = sum(dd) / len(dd)
                se = st.pstdev(dd) / len(dd) ** 0.5
                t = dc / se if se else 0.0
                print(f"{p:<10s}{T:>6.1f}{w:>7.2f}{dt:>+12.6f}{dc:>+12.6f}{t:>+7.2f}",
                      flush=True)
                results.append((dt, dc, t, p, T, w))

    if args.three_way and len(raws) == 2:
        print("\nthree-way v7synth + v6warm + v2 (equal partner split)")
        print(f"{'T_p':>6s}{'w':>7s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")
        for T in ts:
            cals = [{k: rescale(v, T) for k, v in raw.items() if k in h.ce}
                    for raw in raws.values()]
            for w in ws:
                ce = {k: (1 - w) * h.ce[k] + w * 0.5 * (cals[0][k] + cals[1][k])
                      for k in h.ce}
                run = S.build(ce)
                c_tune, c_cf = slices(run)
                dt = sum(y - x for x, y in zip(b_tune, c_tune)) / len(b_tune)
                dd = [y - x for x, y in zip(b_cf, c_cf)]
                dc = sum(dd) / len(dd)
                se = st.pstdev(dd) / len(dd) ** 0.5
                print(f"{T:>6.1f}{w:>7.2f}{dt:>+12.6f}{dc:>+12.6f}"
                      f"{dc/se if se else 0:>+7.2f}", flush=True)
                results.append((dt, dc, dc / se if se else 0, "3way", T, w))

    print("\nbest by confirm half:")
    for dt, dc, t, p, T, w in sorted(results, key=lambda r: -r[1])[:5]:
        print(f"  {p:<8s} T={T:.1f} w={w:.2f}  tune {dt:+.6f}  confirm {dc:+.6f}  t {t:+.2f}")


if __name__ == "__main__":
    main()
