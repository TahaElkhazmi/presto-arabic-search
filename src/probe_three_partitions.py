"""Three proposed partition refinements, tested on top of the SHIPPED cascade.

The live key is now (containment desc, token match-position desc, id asc). Each
experiment inserts one more level between match-position and id, so the decision point
under test is: among candidates that already agree on containment AND on where the
query matches, does the new feature separate relevant from non-relevant?

  E1 LCP ratio buckets  - ordinal head-match strength. Targets the residual population
                          where the full query does NOT appear contiguously, which
                          currently falls straight through to raw id (match-position
                          returns 0 for all of them, so they are unordered by it).
  E2 TQD buckets        - query-token count over title-token count. Within a query this
                          is a monotone transform of title length, already measured at
                          AUC 0.5606 raw and 0.5420 conditional, i.e. below the low-id
                          incumbent. BUCKETING is the part worth testing: coarse buckets
                          preserve the low-id ordering inside each bucket, which is
                          exactly why binary prefix succeeded where continuous GATE
                          (which displaced id entirely) cost -0.031.
  E3 ID-block topology  - gap-detected catalog sections, ordered by block minimum then
                          id. Related to id contiguity, which measured 0.5811 on the
                          zero cohort but 0.3586 (INVERTED) on healthy queries, so the
                          healthy-cohort column is the one that decides this.

Sorting uses explicit TUPLE keys rather than a packed scalar, because the shipped key
already has two levels and packing a third into one float invites precision collisions
(match-position gaps fall to ~1e-3 at high token offsets).

All AUCs are per query, never pooled. Zero GPU.
"""

import bisect
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.60, 0.010, 0.040, 0.30


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


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

    def matchpos(q, i):
        toks, qq = ntok[i], qt[q]
        n = len(qq)
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    # ---- E1: LCP ratio bucket ----
    def lcp_bucket(q, i):
        a, b = qt[q], ntok[i]
        k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]:
            k += 1
        r = k / len(a) if a else 0.0
        return 2 if r >= 1.0 else (1 if r >= 0.6 else 0)

    # ---- E2: TQD bucket ----
    def tqd_bucket(q, i):
        t = len(ntok[i])
        d = len(qt[q]) / t if t else 0.0
        return 2 if d >= 0.8 else (1 if d >= 0.4 else 0)

    # ---- E3: id-block, computed per tie group ----
    def id_blocks(items):
        ids = sorted(items, key=num)
        if len(ids) < 3:
            return {i: num(i) for i in ids}
        gaps = [num(ids[k + 1]) - num(ids[k]) for k in range(len(ids) - 1)]
        mu = st.mean(gaps)
        sd = st.pstdev(gaps) if len(gaps) > 1 else 0.0
        thr = mu + 3 * sd
        block_min, out = num(ids[0]), {}
        for k, iid in enumerate(ids):
            if k > 0 and (num(iid) - num(ids[k - 1])) > thr:
                block_min = num(iid)
            out[iid] = block_min
        return out

    # ================= conditional AUC =================
    base_run_key = lambda q, i: (-h.cont3(q, i), -matchpos(q, i), num(i))
    live = {}
    for q in h.qids:
        live[q] = sorted(h.pools[q], key=lambda i: base_run_key(q, i))

    nd_live = None

    def cond_auc(fn, qs):
        pairs = []
        for q in qs:
            pool = h.pools[q]
            cells = {}
            for i in pool:
                cells.setdefault((round(h.cont3(q, i), 12),
                                  round(matchpos(q, i), 12)), []).append(i)
            for sub in cells.values():
                if len(sub) < 2:
                    continue
                p = [fn(q, i) for i in sub if qrels[q].get(i, 0) > 0]
                n = [fn(q, i) for i in sub if qrels[q].get(i, 0) <= 0]
                if p and n:
                    pairs.append((p, n))
        return per_query_auc(pairs)

    tmp = h.build_run(lam=LAM, eps=EPS, eps_hi=EPS_HI, conf_t=CONF_T,
                      key=lambda q, i: h.cont3(q, i) * 1000.0 + matchpos(q, i))
    ndm = {q: ndcg_at_k(tmp[q], qrels[q], 10) for q in h.qids if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]

    print("CONDITIONAL per-query AUC, given containment AND match-position are equal")
    print("(the exact decision the shipped cascade currently makes with raw id)\n")
    print(f"  {'feature':<26s}{'zero cohort':>14s}{'healthy':>12s}")
    feats = {
        "low-id (incumbent)": lambda q, i: -num(i),
        "E1 LCP ratio bucket": lcp_bucket,
        "E2 TQD bucket": tqd_bucket,
    }
    for name, fn in feats.items():
        print(f"  {name:<26s}{cond_auc(fn, zc):>14.4f}{cond_auc(fn, hc):>12.4f}")

    def e3(q, i, blocks):
        return -blocks[i]
    pairs_z, pairs_h = [], []
    for qs, acc in ((zc, pairs_z), (hc, pairs_h)):
        for q in qs:
            cells = {}
            for i in h.pools[q]:
                cells.setdefault((round(h.cont3(q, i), 12),
                                  round(matchpos(q, i), 12)), []).append(i)
            for sub in cells.values():
                if len(sub) < 2:
                    continue
                bl = id_blocks(sub)
                p = [-bl[i] for i in sub if qrels[q].get(i, 0) > 0]
                n = [-bl[i] for i in sub if qrels[q].get(i, 0) <= 0]
                if p and n:
                    acc.append((p, n))
    print(f"  {'E3 id-block min':<26s}{per_query_auc(pairs_z):>14.4f}"
          f"{per_query_auc(pairs_h):>12.4f}")

    # ================= end to end =================
    def build(extra, qs):
        out = {}
        for q in qs:
            prior, rrf_max = h.priors[q], (3.0 + 1.0) / 61.0
            scored = [(LAM * h.ce[(q, i)] + (1 - LAM) * prior[i] / rrf_max, i)
                      for i in h.pools[q]]
            scored.sort(key=lambda x: (-x[0], num(x[1])))
            eps = EPS_HI if h.conf[q] < CONF_T else EPS
            blocks = None
            if extra == "e3":
                blocks = id_blocks(h.pools[q])

            def key(i, q=q, blocks=blocks):
                base = (-h.cont3(q, i), -matchpos(q, i))
                if extra == "e1":
                    return base + (-lcp_bucket(q, i), num(i))
                if extra == "e2":
                    return base + (-tqd_bucket(q, i), num(i))
                if extra == "e3":
                    return base + (blocks[i], num(i))
                return base + (num(i),)

            res, grp = [], []
            for s, iid in scored:
                if grp and (grp[-1][0] - s) > eps:
                    grp.sort(key=lambda x: key(x[1]))
                    res.extend(i for _, i in grp)
                    grp = []
                grp.append((s, iid))
            grp.sort(key=lambda x: key(x[1]))
            res.extend(i for _, i in grp)
            out[q] = res[:10]
        return out

    def sc(run, qs):
        v = [ndcg_at_k(run[q], qrels[q], 10) for q in qs if q in run]
        return sum(v) / len(v)

    base_t, base_c = build(None, tune), build(None, conf)
    print(f"\nEND TO END (shipped cascade = baseline)")
    print(f"  baseline: tune {sc(base_t, tune):.6f}   confirm {sc(base_c, conf):.6f}\n")
    print(f"  {'experiment':<22s}{'tune delta':>13s}{'confirm delta':>15s}{'t':>8s}")
    for tag, label in (("e1", "E1 LCP buckets"), ("e2", "E2 TQD buckets"),
                       ("e3", "E3 id-blocks")):
        rt, rc = build(tag, tune), build(tag, conf)
        dt = sc(rt, tune) - sc(base_t, tune)
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base_c[q], qrels[q], 10)
             for q in conf]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<22s}{dt:>+13.6f}{m:>+15.6f}{m / se if se else 0:>+8.2f}")
    print(f"\n  gate: positive, significant, and >= +0.0015 incremental")


if __name__ == "__main__":
    main()
