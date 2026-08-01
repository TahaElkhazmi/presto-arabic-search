"""ASSIGNMENT: does any signal we own separate GRADE 3 from GRADE 2, at the decision cell?

Gain is 2^grade - 1 (7 vs 3), so a 3-vs-2 inversion costs more than half the value of a
slot. probe_headroom_autopsy.py measured +0.0854 of oracle headroom in reordering our own
top-10, 45.5% of it graded, and 86.7% of THAT is specifically grade-3-over-grade-2. Every
prior screen in this project used positive-vs-negative as the target, so a feature that
separates 3 from 2 specifically could have been sitting in plain sight.

Semantically: grade 3 is "the product the query actually names"; grade 2 is "the right
kind of thing, wrong variant". So the hypothesis is that the distinguishing information is
in what the TITLE ADDS beyond the query - a size, a colour, a sub-model, a different
product family - which a canonical grade-3 listing would not carry.

FEATURES (all oriented so higher score = more "grade-3-like", i.e. LESS residual content):
  F1  IDF-weighted count of title tokens absent from the query (continuous). Different
      quantity from the raw title/query LENGTH RATIO on the dead list (-0.008 to -0.018,
      the closed-negatives list): that was unweighted length; this is IDF-weighted
      residual content. NOTE: probe_residual_text.py already computed an IDF residual sum,
      but scoped to the saturated-containment tie group of a zero-cohort sample - a
      different, coarser population than the 4-level decision cell used here.
  F2  Same as F1, restricted to tokens containing a digit (size/quantity markers).
  F3  Count of title tokens (absent from the query) whose catalog document frequency is
      below N - an approximation of "a token from a different product family". Two
      thresholds tested (N=100, N=1000).
  F4  Residual character-3gram mass: |title grams NOT in query grams| / |title grams|.
      The reverse direction of cont3 (which is |query grams IN title| / |query grams| and
      is already the key's top level) - a genuinely different quantity, not a restatement.
  F5  Fraction of title tokens AFTER the last matched query token (is the query a suffix of
      the title, or is there trailing content). Different from the S.lt key level, which
      buckets the last token's position from the title START; this measures distance to the
      title END and is NOT one of the four cell-defining dimensions, so it is NOT constant
      inside the cell by construction.

METHOD: screen at the exact decision cell - pool candidates sharing identical
(cont3, match-position, first-token-position, last-token-position), which is where the
low-id fallback currently decides (see src/probe_grade_separation.py and
src/probe_cell_geometry.py, whose id-neighbourhood screen at this same cell definition,
over the full pool, is the source of the 0.6976 binary anchor quoted in
the closed-negatives list). AUC pairs are formed ONLY within a query's own cell(s)
- never across queries - then pooled into one concordance rate, exactly matching the
established pattern in probe_grade_separation.py / probe_cell_geometry.py (that is what
"AUC within each query, never pooled across queries" means operationally in this codebase:
comparisons never cross a query boundary, and the resulting pair list is then aggregated).

Candidate universe is the FULL POOL (not just the shipped top-10), matching
probe_cell_geometry.py's IDEA 2, since the cell is a pool-wide partition and low-id decides
inside it regardless of which members eventually reach the top-10.

A feature is only interesting if it beats the incumbent AND has coverage (1 - ties%) > 0.05.
Eleven features have already lost to low-id at this cell; the prior is strongly against us.

Screening on splits/holdout_remainder8012.txt only. splits/clean_arbiter2784.txt is not
touched.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from probe_ensemble_current import Stack3

setup_utf8()


def idnum(i):
    d = "".join(c for c in i if c.isdigit())
    return int(d) if d else 0


def auc_and_ties(pairs):
    """pairs: list of (score_hi, score_lo). Matches probe_grade_separation.py's auc()."""
    if not pairs:
        return None, None
    w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs)
    ties = sum(1 for a, b in pairs if a == b)
    return w / len(pairs), 100.0 * ties / len(pairs)


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    print(f"building Stack3 harness over {len(hold):,} holdout queries ...", flush=True)
    S = Stack3(hold)
    h = S.h
    qrels = h.qrels

    # independent token document-frequency table (mirrors Stack3.__init__'s df/idf
    # construction exactly, so S.idf(t) and log(1 + N_total/(df[t]+0.5)) agree - checked
    # below as an internal consistency guard before trusting F3's "fewer than N" claim)
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm_names = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    uniq_names = set(norm_names.values())
    df = Counter()
    for n in uniq_names:
        for t in set(n.split()):
            df[t] += 1
    n_total = len(uniq_names)

    import math
    mism = sum(1 for t in list(df)[:2000]
               if abs(S.idf(t) - math.log(1.0 + n_total / (df[t] + 0.5))) > 1e-9)
    print(f"df/idf consistency check (2000 sample tokens): {mism} mismatches "
          f"{'OK' if mism == 0 else '*** MISMATCH ***'}")

    # ---- feature definitions --------------------------------------------------
    def title_toks(i):
        return S.ntok[i]

    def query_tok_set(q):
        return set(S.qt[q])

    def f1_residual_idf(q, i):
        qs = query_tok_set(q)
        return -sum(S.idf(t) for t in set(title_toks(i)) - qs)

    def f2_residual_idf_numeric(q, i):
        qs = query_tok_set(q)
        return -sum(S.idf(t) for t in set(title_toks(i)) - qs
                    if any(c.isdigit() for c in t))

    def make_f3(N):
        def f(q, i):
            qs = query_tok_set(q)
            return -sum(1 for t in set(title_toks(i)) - qs if df.get(t, 0) < N)
        return f

    def f4_residual_gram_mass(q, i):
        ng = h.name_grams(i)
        if not ng:
            return 0.0
        qg = h.qgrams[q]
        return -(len(ng - qg) / len(ng))

    def f5_trailing_frac(q, i):
        toks = title_toks(i)
        qt = S.qt[q]
        last = qt[-1] if qt else None
        if last is None or last not in toks or not toks:
            return 0.0  # uniform inside cells where lt=0 (last token not found for anyone)
        k = toks.index(last)
        trailing = len(toks) - 1 - k
        return -(trailing / len(toks))

    feats = {
        "F1 residual IDF (title\\query)": f1_residual_idf,
        "F2 residual IDF (numeric only)": f2_residual_idf_numeric,
        "F3 rare-token count (N=100)": make_f3(100),
        "F3 rare-token count (N=1000)": make_f3(1000),
        "F4 residual 3gram mass (neg)": f4_residual_gram_mass,
        "F5 trailing-content frac (neg)": f5_trailing_frac,
        "low-id (incumbent)": lambda q, i: -idnum(i),
        "CE score (reference)": lambda q, i: h.ce[(q, i)],
    }

    # ---- build decision cells over the FULL POOL, once per query --------------
    def cell_key(q, i):
        return (h.cont3(q, i), S.mp(q, i), S.ft(q, i), S.lt(q, i))

    cells_by_q = {}
    for q in hold:
        cells = {}
        for i in h.pools[q]:
            cells.setdefault(cell_key(q, i), []).append(i)
        cells_by_q[q] = cells

    # ---- BINARY target: judged-positive vs judged-non-positive, within cell ----
    bin_pairs = {k: [] for k in feats}
    n_bin_cells = n_bin_q = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        used = False
        for members in cells_by_q[q].values():
            if len(members) < 2:
                continue
            pos = [i for i in members if rel.get(i, 0) > 0]
            neg = [i for i in members if rel.get(i, 0) <= 0 and i in rel]
            if not pos or not neg:
                continue
            used = True
            n_bin_cells += 1
            for k, f in feats.items():
                for a in pos:
                    for b in neg:
                        bin_pairs[k].append((f(q, a), f(q, b)))
        n_bin_q += 1 if used else 0

    # ---- GRADED targets: within cell, both judged positive, grades differ -----
    # (a) general graded (any pair of differing positive grades)
    # (b) the assignment's actual target: grade 3 vs grade 2 specifically
    gen_pairs = {k: [] for k in feats}
    g32_pairs = {k: [] for k in feats}
    n_gen_cells = n_g32_cells = n_gen_q = n_g32_q = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        used_gen = used_g32 = False
        for members in cells_by_q[q].values():
            if len(members) < 2:
                continue
            graded = [(i, rel[i]) for i in members if rel.get(i, 0) > 0]
            if len(graded) < 2:
                continue
            prs = [(a, b) for a, ga in graded for b, gb in graded if ga > gb]
            if prs:
                used_gen = True
                n_gen_cells += 1
                for k, f in feats.items():
                    gen_pairs[k].extend((f(q, a), f(q, b)) for a, b in prs)
            threes = [i for i, g in graded if g == 3]
            twos = [i for i, g in graded if g == 2]
            if threes and twos:
                used_g32 = True
                n_g32_cells += 1
                for k, f in feats.items():
                    for a in threes:
                        for b in twos:
                            g32_pairs[k].append((f(q, a), f(q, b)))
        n_gen_q += 1 if used_gen else 0
        n_g32_q += 1 if used_g32 else 0

    def report(title, pairs, n_cells, n_q, anchor):
        print(f"\n=== {title}")
        print(f"pairs: {sum(len(v) for v in pairs.values()) // max(len(feats), 1):,} "
              f"(per feature)  cells: {n_cells:,}  queries touched: {n_q:,} "
              f"({100 * n_q / len(hold):.1f}% of holdout)")
        if anchor is not None:
            print(f"reference anchor for low-id (incumbent) at this cell: ~{anchor:.4f}")
        print(f"{'feature':<34s}{'AUC':>8s}{'ties %':>9s}{'coverage':>10s}")
        rows = []
        for k in feats:
            a, ties = auc_and_ties(pairs[k])
            rows.append((a if a is not None else float("nan"), ties if ties is not None
                         else 0.0, k))
        base = next((a for a, _, k in rows if k.startswith("low-id")), None)
        for a, ties, k in sorted(rows, key=lambda r: (-(r[0] if r[0] == r[0] else -1))):
            cov = 100.0 - ties
            mark = ""
            if base is not None and a == a:
                if a > base and cov > 5.0:
                    mark = "  <-- BEATS incumbent, coverage OK"
                elif a > base:
                    mark = "  <-- beats incumbent but low coverage"
            print(f"{k:<34s}{a:>8.4f}{ties:>9.1f}{cov:>9.1f}%{mark}")

    report("BINARY target (positive vs judged-non-positive), full-pool decision cell",
           bin_pairs, n_bin_cells, n_bin_q, anchor=0.6976)
    report("GRADED target, GENERAL (any differing positive-grade pair), same cell",
           gen_pairs, n_gen_cells, n_gen_q, anchor=0.6731)
    report("GRADED target, GRADE 3 vs GRADE 2 SPECIFICALLY (the assignment's real target)",
           g32_pairs, n_g32_cells, n_g32_q, anchor=None)

    print("\nreading: AUC 0.5 = chance. coverage = 100 - ties% = share of within-cell "
          "pairs the\nfeature actually reaches (expected lift = coverage x edge). Rule of "
          "thumb from\nsection 7.5 of the study guide: only interesting if coverage > 5% "
          "AND it beats\nthe low-id incumbent's AUC on this exact cell and target.")


if __name__ == "__main__":
    main()
