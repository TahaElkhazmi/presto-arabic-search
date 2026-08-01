"""COVERAGE-WEIGHTED screen: the quantity that actually predicts an in-band feature's worth.

WHY THIS EXISTS. Two screens have now been wrong in opposite directions.
  - The ORIGINAL screen compared a feature's STANDALONE within-cell AUC against the low-id
    prior and skipped anything that lost. That retired nine features, one of which
    (first-token position) was worth +0.0030 once measured as the cascade actually uses it.
  - The COMBINED-KEY screen that replaced it ranks a feature by the AUC of `(-feature, +id)`.
    That is the right ordering quantity, but it is NOT sufficient: in Experiment 1 a stemmed
    match-position scored the best combined AUC of anything tested at this decision cell
    (0.6480 zero-cohort against the incumbent's 0.6374) and delivered +0.000034 end to end.

The missing term is COVERAGE. Combined AUC says how well a feature orders the pairs it
SPLITS; it says nothing about how many pairs it splits at all. A feature that is right
whenever it fires but almost never fires is worth nothing, and the two screens above cannot
tell that case apart from a real signal.

THE DECOMPOSITION. For every within-cell (positive, negative) pair, the combined key
`(-feature, +id)` decides by the feature when the feature differs, and by low-id otherwise.
So the AUC improvement of the combined key over low-id alone is exactly:

    expected lift = coverage * edge

where coverage = share of pairs on which the feature takes different values, and edge =
(feature's accuracy - low-id's accuracy) ON THOSE PAIRS ONLY. Both terms are reported, so a
high-AUC/low-coverage feature is visibly distinguishable from a real one.

VALIDATION. This is checked against features whose end-to-end result is already measured,
at the same decision cell they were measured in, so the screen is falsifiable rather than
merely plausible.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_arabic_stem import stem, stem_text
from probe_harness import DIALECT_FILLERS, Harness
from probe_phase3_predicates import jaro_winkler
from stacker_features import char_ngrams, containment

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
T_LIVE, BETA = 3.2, 0.35
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2

# End-to-end confirm-half deltas already measured, for falsifying the screen.
KNOWN_3LVL = {
    "last-token position": +0.001190,
    "query-token coverage": +0.000718,
    "all query tokens present": +0.000656,
    "ordered subsequence": +0.000651,
    "stemmed match-position": +0.000034,
    "stemmed first-token": -0.000011,
    "title length bucket": None,
    "JW prefix bucket": None,
}
KNOWN_2LVL = {"FIRST-TOKEN POSITION (shipped)": +0.002628}


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}
    snorm = {i: stem_text(n) for i, n in norm.items()}
    stok = {i: n.split() for i, n in snorm.items()}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, sqt, qs_ = {}, {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q], sqt[q], qs_[q] = t, [stem(x) for x in t], " ".join(t)

    def _mp(toks, qq):
        n = len(qq)
        for k in range(len(toks) - n + 1) if n else ():
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    def _ft(toks, qq):
        return 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0

    mp = lambda q, i: _mp(ntok[i], qt[q])
    ft = lambda q, i: _ft(ntok[i], qt[q])

    def qcov(q, i):
        s = set(ntok[i])
        return float(sum(1 for t in qt[q] if t in s))

    def all_present(q, i):
        s = set(ntok[i])
        return 1.0 if all(t in s for t in qt[q]) else 0.0

    def ordered(q, i):
        k, qq = 0, qt[q]
        for t in ntok[i]:
            if k < len(qq) and t == qq[k]:
                k += 1
        return 1.0 if k == len(qq) else 0.0

    def last_tok(q, i):
        toks, last = ntok[i], (qt[q][-1] if qt[q] else None)
        if last is None or last not in toks:
            return 0.0
        k = toks.index(last)
        return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

    def title_len(q, i):
        n = len(ntok[i])
        return 3.0 if n <= 4 else (2.0 if n <= 8 else (1.0 if n <= 14 else 0.0))

    def jw(q, i):
        s = jaro_winkler(qs_[q][:40], norm[i][:40])
        return 3.0 if s >= 0.90 else (2.0 if s >= 0.80 else (1.0 if s >= 0.70 else 0.0))

    feats3 = {
        "last-token position": last_tok,
        "query-token coverage": qcov,
        "all query tokens present": all_present,
        "ordered subsequence": ordered,
        "stemmed match-position": lambda q, i: _mp(stok[i], sqt[q]),
        "stemmed first-token": lambda q, i: _ft(stok[i], sqt[q]),
        "title length bucket": title_len,
        "JW prefix bucket": jw,
    }

    def analyse(fn, cell_levels, qs):
        """coverage, edge and expected lift over within-cell (pos, neg) pairs."""
        n_pairs = n_split = 0
        feat_ok = low_ok = 0.0
        for q in qs:
            cells = {}
            for i in h.pools[q]:
                cells.setdefault(tuple(round(f(q, i), 12) for f in cell_levels), []).append(i)
            for sub in cells.values():
                if len(sub) < 2:
                    continue
                pos = [i for i in sub if qrels[q].get(i, 0) > 0]
                neg = [i for i in sub if qrels[q].get(i, 0) <= 0]
                if not pos or not neg:
                    continue
                for p in pos:
                    fp, ip = fn(q, p), num(p)
                    for ng in neg:
                        n_pairs += 1
                        fn_ = fn(q, ng)
                        if fp != fn_:
                            n_split += 1
                            feat_ok += 1.0 if fp > fn_ else 0.0
                            low_ok += 1.0 if ip < num(ng) else (0.5 if ip == num(ng) else 0.0)
        if not n_pairs:
            return 0.0, 0.0, 0.0
        cov = n_split / n_pairs
        edge = (feat_ok - low_ok) / n_split if n_split else 0.0
        return cov, edge, cov * edge

    lvl2 = [h.cont3, mp]
    lvl3 = [h.cont3, mp, ft]

    print("\nSCREEN at the CURRENT decision cell (containment, match-pos, first-token tie)")
    print("  known deltas are the measured confirm-half end-to-end results\n")
    print(f"  {'feature':<26s}{'coverage':>10s}{'edge':>9s}{'exp lift':>11s}"
          f"{'measured':>12s}")
    rows = []
    for nm, fn in feats3.items():
        cov, edge, lift = analyse(fn, lvl3, conf)
        k = KNOWN_3LVL.get(nm)
        rows.append((lift, nm, cov, edge, k))
        ks = f"{k:+.6f}" if k is not None else "not run"
        print(f"  {nm:<26s}{cov:>10.4f}{edge:>+9.4f}{lift:>+11.5f}{ks:>12s}")

    print("\nSCREEN at the PREVIOUS cell (containment, match-pos tie) for the shipped winner")
    cov, edge, lift = analyse(ft, lvl2, conf)
    print(f"  {'FIRST-TOKEN POSITION':<26s}{cov:>10.4f}{edge:>+9.4f}{lift:>+11.5f}"
          f"{KNOWN_2LVL['FIRST-TOKEN POSITION (shipped)']:>+12.6f}")

    scored = [r for r in rows if r[4] is not None]
    scored.sort(reverse=True)
    print("\nFALSIFICATION: does expected lift rank the features by measured outcome?")
    print(f"  {'rank by exp lift':<28s}{'exp lift':>11s}{'measured':>12s}")
    for i, (lift, nm, _, _, k) in enumerate(scored, 1):
        print(f"  {i}. {nm:<25s}{lift:>+11.5f}{k:>+12.6f}")
    order_pred = [r[1] for r in scored]
    order_true = [nm for _, nm in sorted(((k, nm) for _, nm, _, _, k in scored),
                                         reverse=True)]
    print(f"\n  predicted order: {order_pred}")
    print(f"  measured  order: {order_true}")
    print(f"  top pick correct: {order_pred[0] == order_true[0]}")


if __name__ == "__main__":
    main()
