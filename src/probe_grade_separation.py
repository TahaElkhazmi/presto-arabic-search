"""Does ANY signal we own separate grade 3 from grade 2? Nobody has ever asked.

`probe_headroom_autopsy.py` shows +0.0854 of oracle headroom in reordering our OWN top-10,
and 45.5% of it is graded: it needs grade 3 above grade 2, not relevant above irrelevant.
Every screen this project has run (probe_phase3_predicates, probe_combined_key,
screen_coverage, probe_regate_retired) used positive-versus-negative as the target, so a
feature that ranks grade 3 over grade 2 could have been sitting in plain sight and would
never have shown up.

Gains are 7 vs 3, so a grade3/grade2 inversion costs more than half the value of the slot.

AUC is computed WITHIN each query and then averaged (never pooled - a standing rule here),
over the pairs that actually matter: both items judged positive, different grades, both
already inside our shipped top-10. Coverage is reported alongside, because coverage x edge
is what predicts end-to-end lift and the highest edge ever measured here was worthless at
coverage 0.005.

Measurement/screening on the holdout halves only, gated later on the clean arbiter, which
is the same flagged gray zone every previous screen in this project has operated in.
"""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import load_candidates
from probe_ensemble_current import STEM, Stack3

setup_utf8()


def auc(pairs):
    """pairs: list of (score_hi, score_lo) for items whose TRUE grade is hi > lo."""
    if not pairs:
        return None
    w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs)
    return w / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="submission_v7synth_stack3.csv")
    args = ap.parse_args()

    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / args.run)

    print(f"building harness over {len(hold):,} holdout queries ...", flush=True)
    S = Stack3(hold)
    h = S.h
    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}
    norm = {i: normalize_arabic(n) for i, n in names.items()}

    def idnum(i):
        d = "".join(c for c in i if c.isdigit())
        return int(d) if d else 0

    feats = {
        "low-id (incumbent)": lambda q, i: -idnum(i),
        "cont3 containment": lambda q, i: h.cont3(q, i),
        "match position": lambda q, i: S.mp(q, i),
        "first-token position": lambda q, i: S.ft(q, i),
        "last-token position": lambda q, i: S.lt(q, i),
        "highest-idf position": lambda q, i: S.xi(q, i),
        "CE score": lambda q, i: h.ce[(q, i)],
        "RRF prior": lambda q, i: h.priors[q].get(i, 0.0),
        "name length (shorter)": lambda q, i: -len(norm[i]),
        "name token count (fewer)": lambda q, i: -len(norm[i].split()),
        "exact name == query": lambda q, i: 1.0 if norm[i] == normalize_arabic(
            h.queries[q]) else 0.0,
    }

    # pairs where BOTH are in our shipped top-10, both judged positive, grades differ
    per_feat = {k: [] for k in feats}
    nq = npairs = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        top = [i for i in run.get(q, [])[:10] if rel.get(i, 0) > 0 and (q, i) in h.ce]
        cand = [(i, rel[i]) for i in top]
        prs = [(a, b) for a, ga in cand for b, gb in cand if ga > gb]
        if not prs:
            continue
        nq += 1
        npairs += len(prs)
        for k, f in feats.items():
            per_feat[k].extend((f(q, a), f(q, b)) for a, b in prs)

    print(f"\ngrade-separation pairs: {npairs:,} over {nq:,} queries "
          f"({100*nq/len(hold):.1f}% of holdout)")
    print("both items in our shipped top-10, both judged positive, grades differ\n")
    print(f"{'feature':<28s}{'AUC':>8s}{'ties %':>9s}")
    rows = []
    for k in feats:
        p = per_feat[k]
        a = auc(p)
        ties = 100.0 * sum(1 for x, y in p if x == y) / len(p) if p else 0.0
        rows.append((a if a is not None else 0.0, ties, k))
    for a, ties, k in sorted(rows, reverse=True):
        mark = "  <-- beats chance" if a > 0.55 else ("  <-- INVERTED" if a < 0.45 else "")
        print(f"{k:<28s}{a:>8.4f}{ties:>9.1f}{mark}")

    print("\nreading: AUC 0.5 = no signal. A feature only helps if it beats chance AND "
          "its\nties percentage is low enough to actually reach the pairs (coverage).")

    # ---- the same question AT THE ACTUAL DECISION CELL ----
    # A global AUC is the estimator error this project already made twice. What decides a
    # slot is the cell where the shipped key has run out of levels: cont3, match-position,
    # first-token and last-token all TIE, and low-id breaks it. Only a feature that beats
    # low-id THERE can pay, and only on the pairs it actually splits.
    cell = {k: [] for k in ("CE score", "RRF prior", "low-id (incumbent)",
                            "name length (shorter)")}
    ncell = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        top = [i for i in run.get(q, [])[:10] if rel.get(i, 0) > 0 and (q, i) in h.ce]
        key = lambda i: (h.cont3(q, i), S.mp(q, i), S.ft(q, i), S.lt(q, i))
        for a in top:
            for b in top:
                if rel[a] > rel[b] and key(a) == key(b):
                    ncell += 1
                    for k in cell:
                        cell[k].append((feats[k](q, a), feats[k](q, b)))

    print(f"\n=== AT THE DECISION CELL: cont3, match-position, first-token and "
          f"last-token ALL TIE")
    print(f"grade-differing pairs in that cell: {ncell:,} "
          f"({100*ncell/max(npairs,1):.1f}% of all grade pairs)")
    if ncell:
        print(f"{'feature':<28s}{'AUC':>8s}{'ties %':>9s}")
        for k in sorted(cell, key=lambda x: -(auc(cell[x]) or 0)):
            a = auc(cell[k])
            ties = 100.0 * sum(1 for x, y in cell[k] if x == y) / len(cell[k])
            print(f"{k:<28s}{a:>8.4f}{ties:>9.1f}")
        base = auc(cell["low-id (incumbent)"]) or 0.5
        ce = auc(cell["CE score"]) or 0.5
        print(f"\nCE minus low-id at the cell: {ce - base:+.4f}")
        print("Positive means a CE level inserted BELOW the text key and ABOVE low-id "
              "would\norder these pairs better than the incumbent does. That is the only "
              "placement\nthat does not displace anything, and displacing has lost every "
              "time.")


if __name__ == "__main__":
    main()
