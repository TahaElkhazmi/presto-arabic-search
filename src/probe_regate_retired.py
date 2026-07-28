"""Re-gate the RETIRED predicates under the corrected estimator, one level lower.

The combined-key fix promoted 4 of 4 features it was tried on and one of them
(first-token position) was worth +0.0030. Five more features were retired by the same
flawed standalone screen in the phase-3 batch, and they have never been measured the
way the cascade uses them. This re-runs all of them.

TWO DIFFERENCES from probe_combined_key.py, both deliberate:

1. The BASELINE is the NEW key (containment, match-position, FIRST-TOKEN position, id),
   not the shipped one, because that is what we intend to ship. The question is what is
   left to separate AFTER first-token position, not before it.
2. The decision cell is therefore conditioned on all THREE existing levels being equal.

Screening runs on holdout_remainder8012 only; the full-benchmark and clean-arbiter
confirmation is a separate, slower pass reserved for whatever survives here.
"""

import bisect
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness
from probe_phase3_predicates import jaro_winkler, per_query_auc

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE, BETA = 3.2, 0.35
# four ordered levels now, so the feature term must stay below the smallest gap of the
# level above it; the regression check against the shipped key is what proves it.
S_CONT, S_MP, S_FT, S_FEAT = 1e6, 1e2, 1e-2, 1e-6


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, qs_ = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
        qs_[q] = " ".join(qt[q])

    mpc, ftc = {}, {}

    def matchpos(q, i):
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

    def firsttok(q, i):
        v = ftc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
            ftc[(q, i)] = v
        return v

    # ---- the five retired phase-3 features, verbatim in behaviour ----
    jwc = {}

    def jw_bucket(q, i):
        v = jwc.get((q, i))
        if v is None:
            s = jaro_winkler(qs_[q][:40], norm[i][:40])
            v = jwc[(q, i)] = 3.0 if s >= 0.90 else (2.0 if s >= 0.80 else
                                                     (1.0 if s >= 0.70 else 0.0))
        return v

    def has_size(q, i):
        return float(any(any(c.isdigit() for c in t) for t in ntok[i]))

    def qcov(q, i):
        s = set(ntok[i])
        return float(sum(1 for t in qt[q] if t in s))

    def latin_match(q, i):
        lat = lambda s: any("a" <= c.lower() <= "z" for c in s)
        return float(lat(qs_[q]) == lat(norm[i]))

    def last_tok_bucket(q, i):
        toks = ntok[i]
        last = qt[q][-1] if qt[q] else None
        if last is None or last not in toks:
            return 0.0
        k = toks.index(last)
        return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

    # ---- the three promoted-but-not-shipped features from the combined-key probe ----
    def all_present(q, i):
        s = set(ntok[i])
        return 1.0 if all(t in s for t in qt[q]) else 0.0

    def ordered_subseq(q, i):
        k = 0
        qq = qt[q]
        for t in ntok[i]:
            if k < len(qq) and t == qq[k]:
                k += 1
        return 1.0 if k == len(qq) else 0.0

    def title_len_bucket(q, i):
        n = len(ntok[i])
        return 3.0 if n <= 4 else (2.0 if n <= 8 else (1.0 if n <= 14 else 0.0))

    feats = {
        "JW prefix bucket": jw_bucket,
        "size-marker present": has_size,
        "query-token coverage": qcov,
        "latin script match": latin_match,
        "last-token position": last_tok_bucket,
        "all query tokens present": all_present,
        "ordered subsequence": ordered_subseq,
        "title length bucket": title_len_bucket,
    }

    def build(fn=None, qids=None):
        def key(q, i):
            v = h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP + firsttok(q, i) * S_FT
            return v + fn(q, i) * S_FEAT if fn else v
        return h.build_run(key=key, bonus_fn=lambda q, i: BETA * matchpos(q, i),
                           qids=qids, **LIVE)

    base = build()

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt, bc = sc(base, tune), sc(base, conf)
    print(f"baseline (the new first-token key): tune {bt:.6f}  confirm {bc:.6f}")

    ndm = {q: ndcg_at_k(base[q], qrels[q], 10) for q in hold if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]
    print(f"cohorts: zero {len(zc):,}  healthy {len(hc):,}\n")

    def cells(q):
        out = {}
        for i in h.pools[q]:
            k = (round(h.cont3(q, i), 12), round(matchpos(q, i), 12),
                 round(firsttok(q, i), 12))
            out.setdefault(k, []).append(i)
        return out

    def auc(fn, qs, combined):
        pairs = []
        for q in qs:
            for sub in cells(q).values():
                if len(sub) < 2:
                    continue
                if combined:
                    order = sorted(sub, key=lambda i: (-fn(q, i), num(i)))
                    val = {i: -r for r, i in enumerate(order)}
                else:
                    val = {i: fn(q, i) for i in sub}
                p = [val[i] for i in sub if qrels[q].get(i, 0) > 0]
                n = [val[i] for i in sub if qrels[q].get(i, 0) <= 0]
                if p and n:
                    pairs.append((p, n))
        return per_query_auc(pairs)

    lowid = lambda q, i: -num(i)
    iz, ih = auc(lowid, zc, False), auc(lowid, hc, False)
    print("AUC in the cell where containment, match-position AND first-token all tie")
    print(f"  {'':<26s}{'standalone':>22s}{'COMBINED with low-id':>24s}")
    print(f"  {'feature':<26s}{'zero':>11s}{'healthy':>11s}{'zero':>12s}{'healthy':>12s}")
    print(f"  {'low-id (incumbent)':<26s}{iz:>11.4f}{ih:>11.4f}{iz:>12.4f}{ih:>12.4f}")
    promoted = []
    for name, fn in feats.items():
        cz, ch_ = auc(fn, zc, True), auc(fn, hc, True)
        sz, sh = auc(fn, zc, False), auc(fn, hc, False)
        ok = cz > iz and ch_ > ih
        print(f"  {name:<26s}{sz:>11.4f}{sh:>11.4f}{cz:>12.4f}{ch_:>12.4f}"
              f"   {'pass' if ok else 'skip'}")
        if ok:
            promoted.append((name, fn))

    if not promoted:
        print("\nNothing separates the residue below first-token position.")
        return

    print(f"\nEND TO END as a FOURTH level, for the {len(promoted)} promoted\n")
    print(f"  {'feature':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>10s}")
    for name, fn in promoted:
        rt, rc = build(fn, tune), build(fn, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0
        print(f"  {name:<26s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}"
              f"{'CARRY' if (m >= 0.0015 and t > 2) else 'drop':>10s}")

    print("\n  CARRY means it earns a full-benchmark and clean-arbiter run, not a ship.")


if __name__ == "__main__":
    main()
