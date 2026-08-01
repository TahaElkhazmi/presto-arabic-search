"""Re-gate the dead predicates under the estimator the cascade ACTUALLY uses, and
test the one Meilisearch ranking dimension never tried: PROXIMITY.

TWO IDEAS, both cheap and both cache-only.

(1) THE ESTIMATOR WAS WRONG FOR PARTITION FEATURES.
`probe_phase3_predicates.py` skips a feature unless its STANDALONE within-group AUC
beats low-id on both cohorts. But the cascade never uses a feature standalone: it sorts
by `(-feature, +id)`, so the feature PARTITIONS and low-id still orders inside each
class. A feature can be weaker than low-id overall and still improve the pair, if the
candidate pairs it separates are ones low-id gets WRONG. That is precisely how the
prefix key won: it is binary, it preserved the incumbent, and it beat GATE-AraBert,
which was continuous and displaced the incumbent (-0.031).
So the right quantity is the AUC of the COMBINED lexicographic key, and it is computed
here by ranking each cell under `(-feature, +id)` and scoring that ranking. Nine
features were retired on the standalone comparison; this re-opens all of them for the
price of one CPU pass.

(2) PROXIMITY IS THE UNTESTED MEILISEARCH DIMENSION.
The judged pools came from Meilisearch, whose ranking rules run words -> typo ->
PROXIMITY -> attribute -> exactness. The prefix key captured exactness and the
match-position key captured attribute position, and those two were the campaign's two
largest wins. Proximity has never been tested.
It matters most exactly where the current key goes blind: `matchpos` only fires when the
query tokens appear CONTIGUOUSLY (`toks[k:k+n] == qq`). Every title where the query
tokens are all present but SCATTERED scores matchpos = 0 and falls straight through to
raw id. Proximity inserts a discrete level in that cohort, which is the same move -
insert a level above the saturated fallback - that paid twice already.

Everything runs at the CURRENT live operating point (T=3.2, lam=0.50, eps .030/.050,
conf_t 0.10 on the top1-top3 margin, beta=0.35 match-position blend bonus). The
feature-off path must reproduce the file on the board exactly, or nothing here is real.
"""

import bisect
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE, BETA = 3.2, 0.35

# Key scaling. cont3 gaps are >= ~0.02 so *1e6 separates by >= 20,000; the smallest
# positive match-position gap on a realistic title is ~1e-4 so *1e2 separates by ~1e-2;
# the feature term is capped at 3e-5 and therefore cannot cross a match-position level.
# The regression check against the live file is what proves this encoding is faithful.
S_CONT, S_MP, S_FEAT = 1e6, 1e2, 1e-5


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def per_query_auc(pairs):
    conc = disc = ties = 0
    for pos, neg in pairs:
        if not pos or not neg:
            continue
        ns = sorted(neg)
        for p in pos:
            lo, hi = bisect.bisect_left(ns, p), bisect.bisect_right(ns, p)
            conc += lo
            ties += hi - lo
            disc += len(ns) - hi
    tot = conc + disc + ties
    return (conc + 0.5 * ties) / tot if tot else float("nan")


def min_span(toks, qq):
    """Smallest window of the title containing every query token; 0 if any is absent."""
    need = set(qq)
    if not need:
        return 0
    pos = {t: [] for t in need}
    for k, t in enumerate(toks):
        if t in pos:
            pos[t].append(k)
    if any(not v for v in pos.values()):
        return 0
    ptr = {t: 0 for t in need}
    best = 10 ** 9
    while True:
        cur = [pos[t][ptr[t]] for t in need]
        lo, hi = min(cur), max(cur)
        best = min(best, hi - lo + 1)
        t_lo = next(t for t in need if pos[t][ptr[t]] == lo)
        ptr[t_lo] += 1
        if ptr[t_lo] >= len(pos[t_lo]):
            return best


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    print("building harness over the full benchmark ...", flush=True)
    h = Harness(all_qids, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

    mpc = {}

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

    spc = {}

    def span(q, i):
        v = spc.get((q, i))
        if v is None:
            v = spc[(q, i)] = min_span(ntok[i], qt[q])
        return v

    # ---- candidate partition features (all DISCRETE, all preserve low-id inside) ----
    def all_tokens_present(q, i):
        return 1.0 if span(q, i) > 0 else 0.0

    def proximity_bucket(q, i):
        s = span(q, i)
        if s == 0:
            return 0.0
        slack = s - len(qt[q])          # 0 = contiguous, 1 = one intruding token, ...
        return 3.0 if slack == 0 else (2.0 if slack == 1 else (1.0 if slack <= 3 else 0.5))

    def ordered_subsequence(q, i):
        toks, qq = ntok[i], qt[q]
        k = 0
        for t in toks:
            if k < len(qq) and t == qq[k]:
                k += 1
        return 1.0 if k == len(qq) else 0.0

    def first_token_bucket(q, i):
        toks, qq = ntok[i], qt[q]
        if not qq or qq[0] not in toks:
            return 0.0
        k = toks.index(qq[0])
        return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

    feats = {
        "all query tokens present": all_tokens_present,
        "PROXIMITY bucket (span)": proximity_bucket,
        "ordered subsequence": ordered_subsequence,
        "first-token position": first_token_bucket,
    }

    # ---- cohorts, defined by the CURRENT shipped run ----
    def build(fn=None, qids=None):
        if fn is None:
            key = lambda q, i: h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
        else:
            key = lambda q, i: (h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
                                + fn(q, i) * S_FEAT)
        return h.build_run(key=key, bonus_fn=lambda q, i: BETA * matchpos(q, i),
                           qids=qids, **LIVE)

    base = build()
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_retune_bonus.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs the file on the board (LB 0.68527): {same:,}/{len(live):,}")
    if same != len(live):
        print("  *** encoding is NOT faithful - stop and diagnose ***")
        return

    ndm = {q: ndcg_at_k(base[q], qrels[q], 10) for q in hold if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]
    print(f"cohorts on the holdout: zero {len(zc):,}   healthy {len(hc):,}\n")

    # ---- how big is the blind spot the proximity idea targets? ----
    tot = sc_ = 0
    for q in hold:
        for i in h.pools[q]:
            if h.cont3(q, i) >= 0.999:
                tot += 1
                if matchpos(q, i) == 0.0 and span(q, i) > 0:
                    sc_ += 1
    print(f"BLIND SPOT: of {tot:,} fully-contained candidates, {sc_:,} ({sc_/tot:.1%}) "
          f"have every query token\n  present but NOT contiguous, so match-position "
          f"scores them 0 and they fall through to raw id.\n")

    # ---- AUC: standalone (the old gate) vs COMBINED with low-id (the real one) ----
    def cells(q):
        out = {}
        for i in h.pools[q]:
            out.setdefault((round(h.cont3(q, i), 12), round(matchpos(q, i), 12)), []).append(i)
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
    print("AUC inside the exact decision cell (containment AND match-position equal)")
    print(f"  {'':<26s}{'standalone':>22s}{'COMBINED with low-id':>24s}")
    print(f"  {'feature':<26s}{'zero':>11s}{'healthy':>11s}{'zero':>12s}{'healthy':>12s}")
    print(f"  {'low-id (incumbent)':<26s}{iz:>11.4f}{ih:>11.4f}{iz:>12.4f}{ih:>12.4f}")
    promoted = []
    for name, fn in feats.items():
        sz, sh = auc(fn, zc, False), auc(fn, hc, False)
        cz, ch_ = auc(fn, zc, True), auc(fn, hc, True)
        old = "pass" if (sz > iz and sh > ih) else "SKIP"
        new = "pass" if (cz > iz and ch_ > ih) else "skip"
        print(f"  {name:<26s}{sz:>11.4f}{sh:>11.4f}{cz:>12.4f}{ch_:>12.4f}"
              f"   old={old} new={new}")
        if cz > iz and ch_ > ih:
            promoted.append((name, fn))

    if not promoted:
        print("\nNothing clears the incumbent even under the combined-key estimator.")
        return

    print(f"\nEND TO END for the {len(promoted)} feature(s) the combined-key gate promotes")
    print("  (tune half then confirm half of holdout_remainder8012)\n")

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt, bc = sc(base, tune), sc(base, conf)
    print(f"  baseline: tune {bt:.6f}   confirm {bc:.6f}\n")
    print(f"  {'feature':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}")
    for name, fn in promoted:
        rt = build(fn, tune)
        rc = build(fn, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {name:<26s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}")

    print("\n  a feature must hold up on the CONFIRM half before it earns a full-benchmark")
    print("  and arbiter run; tune-half movement alone is the 3-4x inflated number.")


if __name__ == "__main__":
    main()
