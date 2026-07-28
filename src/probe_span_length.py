"""Experiments 2 and 3: span-density bucket and title/query length-ratio bucket.

Both features are DISCRETE and inserted as a level ABOVE the low-id fallback so they
partition each tie cell and low-id still orders inside every class (the design rule;
continuous replacements of low-id have lost every time they were tried).

EXPERIMENT 2 - consecutive token match span density.
  max_consecutive = longest run of query tokens appearing contiguously, in order, in
  the title.  Bucket: 2 = full phrase contiguous, 1 = partial contiguous (>=2), 0 =
  scattered.  Two placements:
    (a) (containment, first-token-pos, span-bucket, id)   - DROPS match-position;
    (b) (containment, match-pos, first-token-pos, span-bucket, id) - appended 4th level.
  (b) exists because bucket 2 largely restates what match-position already encodes.

EXPERIMENT 3 - title/query length asymmetry.
  ratio = len(title_tokens) / len(query_tokens); <=2.5 focused, <=5.0 standard,
  >5.0 bloated.  The key sorts the feature DESCENDING, so for the stated intent
  (suppress bloat) focused must get the HIGHEST value: focused=2, standard=1,
  bloated=0.  The inverted orientation is also run; the direction is empirical.
  PRIOR ART: probe_regate_retired.py measured ABSOLUTE title-length bucket and it
  LOST to the low-id incumbent on both cohorts (0.6124/0.6849 vs 0.6374/0.7244).
  Experiment 3 differs only in normalising by query length.

Screening on holdout_remainder8012 only; full-benchmark / arbiter confirmation is a
separate pass reserved for whatever survives the confirm half here.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness
from probe_phase3_predicates import per_query_auc

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE, BETA = 3.2, 0.35
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

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

    mpc, ftc, spc = {}, {}, {}

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

    # ---- experiment 2: longest contiguous in-order run of query tokens in title ----
    def span_bucket(q, i):
        v = spc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n, best = len(qq), 0
            for k in range(len(toks)):
                for j in range(n):
                    if toks[k] == qq[j]:
                        m = 1
                        while k + m < len(toks) and j + m < n and toks[k + m] == qq[j + m]:
                            m += 1
                        if m > best:
                            best = m
                if best == n:
                    break
            v = 2.0 if best == n else (1.0 if best >= 2 else 0.0)
            spc[(q, i)] = v
        return v

    # ---- experiment 3: title/query token-length ratio bucket ----
    def ratio_bucket(q, i):
        r = len(ntok[i]) / max(len(qt[q]), 1)
        return 2.0 if r <= 2.5 else (1.0 if r <= 5.0 else 0.0)

    def ratio_bucket_inv(q, i):
        return 2.0 - ratio_bucket(q, i)

    # ---- key encodings (scalar; each level dominates the one below) ----
    def key_base(q, i):
        return h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP + firsttok(q, i) * S_FT

    def key_a(fn):      # (cont, first-token, feature, id) - match-position DROPPED
        return lambda q, i: (h.cont3(q, i) * S_CONT + firsttok(q, i) * S_FT
                             + fn(q, i) * S_FEAT)

    def key_b(fn):      # (cont, match-pos, first-token, feature, id)
        return lambda q, i: key_base(q, i) + fn(q, i) * S_FEAT

    def cells2(q):      # ties on (cont, first-token) - variant (a) cell
        out = {}
        for i in h.pools[q]:
            k = (round(h.cont3(q, i), 12), round(firsttok(q, i), 12))
            out.setdefault(k, []).append(i)
        return out

    def cells3(q):      # ties on (cont, match-pos, first-token) - full-key cell
        out = {}
        for i in h.pools[q]:
            k = (round(h.cont3(q, i), 12), round(matchpos(q, i), 12),
                 round(firsttok(q, i), 12))
            out.setdefault(k, []).append(i)
        return out

    variants = [
        ("E2a span (cont,ft,span,id)",      key_a(span_bucket),      span_bucket,      cells2),
        ("E2b span appended 4th level",     key_b(span_bucket),      span_bucket,      cells3),
        ("E3 ratio focused=2 (as intended)", key_b(ratio_bucket),     ratio_bucket,     cells3),
        ("E3 ratio inverted (bloated=2)",   key_b(ratio_bucket_inv), ratio_bucket_inv, cells3),
    ]

    def build(keyfn, qids=None):
        return h.build_run(key=keyfn, bonus_fn=lambda q, i: BETA * matchpos(q, i),
                           qids=qids, **LIVE)

    base = build(key_base)

    # regression check: the scalar baseline must reproduce the shipped 4-level key,
    # i.e. sorting a pool by the scalar equals sorting by the explicit tuple.
    mism = 0
    for q in h.qids:
        pool = h.pools[q]
        a = sorted(pool, key=lambda i: (-key_base(q, i), num(i)))
        b = sorted(pool, key=lambda i: (-h.cont3(q, i), -matchpos(q, i),
                                        -firsttok(q, i), num(i)))
        if a != b:
            mism += 1
    print(f"regression check: scalar vs tuple 4-level key order mismatches "
          f"{mism}/{len(h.qids)} queries {'OK' if mism == 0 else 'FAIL'}")

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt, bc = sc(base, tune), sc(base, conf)
    print(f"baseline (current 4-level key): tune {bt:.6f}  confirm {bc:.6f}")

    ndm = {q: ndcg_at_k(base[q], qrels[q], 10) for q in hold if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]
    print(f"cohorts: zero {len(zc):,}  healthy {len(hc):,}\n")

    def auc(fn, qs, cells, combined):
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
    inc = {cells2: (auc(lowid, zc, cells2, False), auc(lowid, hc, cells2, False)),
           cells3: (auc(lowid, zc, cells3, False), auc(lowid, hc, cells3, False))}
    print("within-cell per-query AUC, COMBINED key (-feature, +id) vs low-id incumbent")
    print(f"  {'variant':<34s}{'cell':>6s}{'inc z':>9s}{'inc h':>9s}"
          f"{'comb z':>9s}{'comb h':>9s}")
    for name, _, fn, cells in variants:
        iz, ih = inc[cells]
        cz, ch_ = auc(fn, zc, cells, True), auc(fn, hc, cells, True)
        lbl = "2-lvl" if cells is cells2 else "3-lvl"
        beat = "beats" if (cz > iz and ch_ > ih) else "below"
        print(f"  {name:<34s}{lbl:>6s}{iz:>9.4f}{ih:>9.4f}{cz:>9.4f}{ch_:>9.4f}  {beat}")

    print(f"\nEND TO END, all variants (threshold: confirm >= +0.0015 and t > 2)\n")
    print(f"  {'variant':<34s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>10s}")
    for name, keyfn, _, _ in variants:
        rt, rc = build(keyfn, tune), build(keyfn, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0
        print(f"  {name:<34s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}"
              f"{'CARRY' if (m >= 0.0015 and t > 2) else 'drop':>10s}")

    print("\n  CARRY means it earns a full-benchmark and clean-arbiter run, not a ship.")


if __name__ == "__main__":
    main()
