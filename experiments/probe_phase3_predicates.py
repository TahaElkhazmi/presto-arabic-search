"""PHASE 3: further non-destructive partition predicates above the low-id fallback.

Rule of preservation, learned the hard way: a candidate predicate must PARTITION the
group into discrete classes so the low-id prior still orders each class. Continuous
scores displace it entirely and lose, which is how GATE-AraBert (-0.031) and the TQD
buckets (-0.008) failed. Low-id at this decision point measures AUC 0.6321 on the zero
cohort and 0.6882 on healthy queries and has now beaten four separate challengers.

Candidates here, all discrete and all untested so far:
  JW prefix bucket   - Jaro-Winkler similarity between query and title, bucketed.
                       Rewards shared leading characters with typo tolerance, which
                       plain prefix matching cannot do; Libyan-dialect queries are
                       typo-heavy.
  size-marker        - does the title carry a digit+unit pack marker (500 غ, 2 كغ)?
                       The failure examples are dominated by pack-size variants.
  qtoken coverage    - how many query tokens appear anywhere in the title. May be
                       saturated (containment already implies gram coverage), which the
                       measurement will show directly.
  latin match        - query and title agree on containing Latin-script tokens. Brand
                       queries like "accu chek" and "enzo" mix scripts.
  last-token position - where the LAST query token lands, bucketed. Complements
                       match-position, which only sees the first contiguous occurrence.

Conditional AUC is computed given containment AND match-position already equal, since
that is the exact decision the shipped cascade currently makes with raw id.
"""

import bisect
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def jaro(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    md = max(len(a), len(b)) // 2 - 1
    if md < 0:
        md = 0
    fa, fb = [False] * len(a), [False] * len(b)
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - md), min(len(b), i + md + 1)):
            if not fb[j] and b[j] == ca:
                fa[i] = fb[j] = True
                m += 1
                break
    if m == 0:
        return 0.0
    t = k = 0
    for i, ca in enumerate(a):
        if fa[i]:
            while not fb[k]:
                k += 1
            if ca != b[k]:
                t += 1
            k += 1
    t //= 2
    return (m / len(a) + m / len(b) + (m - t) / m) / 3.0


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    j = jaro(a, b)
    l = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        l += 1
    return j + l * p * (1 - j)


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

    qt, qs_ = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
        qs_[q] = " ".join(qt[q])

    mp_cache = {}

    def matchpos(q, i):
        v = mp_cache.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            mp_cache[(q, i)] = v
        return v

    jw_cache = {}

    def jw_bucket(q, i):
        v = jw_cache.get((q, i))
        if v is None:
            s = jaro_winkler(qs_[q][:40], norm[i][:40])
            v = 3 if s >= 0.90 else (2 if s >= 0.80 else (1 if s >= 0.70 else 0))
            jw_cache[(q, i)] = v
        return v

    def has_size(q, i):
        toks = ntok[i]
        return float(any(any(c.isdigit() for c in t) for t in toks))

    def qcov(q, i):
        s = set(ntok[i])
        return sum(1 for t in qt[q] if t in s)

    def latin_match(q, i):
        def lat(s):
            return any("a" <= c.lower() <= "z" for c in s)
        return float(lat(qs_[q]) == lat(norm[i]))

    def last_tok_bucket(q, i):
        toks = ntok[i]
        last = qt[q][-1] if qt[q] else None
        if last is None or last not in toks:
            return 0
        k = toks.index(last)
        return 3 if k == 0 else (2 if k <= 2 else 1)

    feats = {
        "low-id (incumbent)": lambda q, i: -num(i),
        "JW prefix bucket": jw_bucket,
        "size-marker present": has_size,
        "query-token coverage": qcov,
        "latin script match": latin_match,
        "last-token position": last_tok_bucket,
    }

    tmp = h.build_run(lam=LAM, eps=EPS, eps_hi=EPS_HI, conf_t=CONF_T,
                      key=lambda q, i: h.cont3(q, i) * 1000.0 + matchpos(q, i))
    ndm = {q: ndcg_at_k(tmp[q], qrels[q], 10) for q in h.qids if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]

    def cond_auc(fn, qs):
        pairs = []
        for q in qs:
            cells = {}
            for i in h.pools[q]:
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

    print("CONDITIONAL AUC given containment AND match-position already equal\n")
    print(f"  {'feature':<26s}{'zero cohort':>14s}{'healthy':>12s}")
    for name, fn in feats.items():
        print(f"  {name:<26s}{cond_auc(fn, zc):>14.4f}{cond_auc(fn, hc):>12.4f}")

    # end to end for anything that beats the incumbent on BOTH cohorts
    inc_z, inc_h = cond_auc(feats["low-id (incumbent)"], zc), \
        cond_auc(feats["low-id (incumbent)"], hc)

    def build(fn, qs):
        out = {}
        rrf_max = 4.0 / 61.0
        for q in qs:
            prior = h.priors[q]
            scored = [(LAM * h.ce[(q, i)] + (1 - LAM) * prior[i] / rrf_max, i)
                      for i in h.pools[q]]
            scored.sort(key=lambda x: (-x[0], num(x[1])))
            eps = EPS_HI if h.conf[q] < CONF_T else EPS

            def key(i, q=q):
                base = (-h.cont3(q, i), -matchpos(q, i))
                return base + ((-fn(q, i), num(i)) if fn else (num(i),))

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

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r]
        return sum(v) / len(v)

    bt, bc = build(None, tune), build(None, conf)
    print(f"\nEND TO END (shipped cascade baseline: tune {sc(bt, tune):.6f}, "
          f"confirm {sc(bc, conf):.6f})\n")
    print(f"  {'feature':<26s}{'tune d':>11s}{'confirm d':>12s}{'t':>8s}{'status':>10s}")
    for name, fn in feats.items():
        if name.startswith("low-id"):
            continue
        z, hh = cond_auc(fn, zc), cond_auc(fn, hc)
        if not (z > inc_z and hh > inc_h):
            print(f"  {name:<26s}{'--':>11s}{'--':>12s}{'--':>8s}"
                  f"{'skip (AUC)':>10s}")
            continue
        rt, rc = build(fn, tune), build(fn, conf)
        dt = sc(rt, tune) - sc(bt, tune)
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc[q], qrels[q], 10)
             for q in conf]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0
        st_ = "KEEP" if (m >= 0.0015 and t > 2) else "discard"
        print(f"  {name:<26s}{dt:>+11.6f}{m:>+12.6f}{t:>+8.2f}{st_:>10s}")

    print("\n  a feature that does not beat low-id on BOTH cohorts is skipped without")
    print("  an end-to-end run, since displacing a stronger incumbent cannot help")


if __name__ == "__main__":
    main()
