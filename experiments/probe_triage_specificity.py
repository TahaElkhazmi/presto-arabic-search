"""Two questions in one pass: WHERE the reorder headroom lives, and whether CE specificity
is a real signal.

TRIAGE (external Idea 0). The +0.0465 binary reorder headroom is the ceiling for any
post-processing change, but it does not say which component is at fault. Split it:

  intra-chain inversion - the relevant item and the zero it sits below are in the SAME eps
                          chain, so the TIE-BREAK KEY had its chance and got it wrong.
  inter-chain inversion - they are in DIFFERENT chains, so the band never compared them and
                          only the BLEND SCORE could have fixed it.

This matters because it decides which world we are in. If intra dominates, key-side work
(conditional order, level ablation) is the lever. If inter dominates, the key is fine and the
lever is the blend, where every signal we own is already spent - and the honest move is to ship.
The proposer correctly noted my earlier 77.2% figure counts SLOTS, not INVERSIONS, so it
suggests but does not establish intra dominance.

SPECIFICITY (external Idea 3). Aggregate our own cached CE scores per NAME across every
benchmark query whose pool contains that name, then

    specificity(q, name) = CE(q, name) - mean_over_queries CE(., name)

A name the cross-encoder scores highly for MANY queries is generic or keyword-stuffed; one it
scores highly for only this query is specific to it. Since grade 3 is "the product the query
names" and grade 2 is "the right kind of thing", specificity is a plausible 3-vs-2 separator,
which is where 45.5% of the remaining headroom sits.

This is the TRANSPOSE of per-query z-normalization (tested, dead): that normalized across
candidates within a query, this normalizes across queries within a product. Legal by
construction - it reads only our own scores, never a judgment.

Screened where it must be screened: within the final decision cell, against the incumbent
low-id prior, on the graded target. A global AUC here would be the fourth estimator trap again.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, DATA, ROOT, load_jsonl, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_ensemble_current import (CONF_T, EPS, EPS_HI, RRF_MAX, STEM, TOP_EPS, TOP_K,
                                    Stack3)
from probe_harness import _id_key

setup_utf8()


def idnum(i):
    d = "".join(c for c in i if c.isdigit())
    return int(d) if d else 0


def auc(pairs):
    if not pairs:
        return None, 0.0
    w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in pairs) / len(pairs)
    ties = 100.0 * sum(1 for a, b in pairs if a == b) / len(pairs)
    return w, ties


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")

    print("building name-level CE profile from the FULL cache ...", flush=True)
    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}
    df = pd.read_parquet(DATA / "cache" / f"{STEM.format('v7synth')}.parquet")
    df["name"] = df["iid"].map(names)
    prof = df.groupby("name")["score"].agg(["mean", "max", "count"])
    print(f"  {len(prof):,} distinct names, mean pools/name {prof['count'].mean():.1f}")
    nmean = prof["mean"].to_dict()
    ncount = prof["count"].to_dict()
    del df, prof

    print(f"building harness over {len(hold):,} holdout queries ...", flush=True)
    S = Stack3(hold)
    h = S.h

    # ---------- TRIAGE ----------
    intra = inter = 0
    intra_gain = inter_gain = 0.0
    gpair = {}
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        top = [i for i in run.get(q, [])[:10] if (q, i) in h.ce]
        if len(top) < 2:
            continue
        # rebuild the shipped chains so we know which pairs shared one
        scored = []
        for i in h.pools[q]:
            v = (0.50 * h.ce[(q, i)] + 0.50 * h.priors[q][i] / RRF_MAX
                 + 0.35 * S.mp(q, i) + 0.05 * S.ft(q, i) + 0.05 * S.xi(q, i))
            scored.append((v, i))
        scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
        ces = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        conf = ces[0] - ces[2] if len(ces) >= 3 else 1.0
        e0 = EPS_HI if conf < CONF_T else EPS
        chain_of, cid = {}, 0
        prev = None
        for posn, (sc, iid) in enumerate(scored):
            e = TOP_EPS if posn < TOP_K else e0
            if prev is not None and (prev - sc) > e:
                cid += 1
            chain_of[iid] = cid
            prev = sc
        base = ndcg_at_k(top, rel, 10)
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                hi, lo = top[a], top[b]
                if rel.get(lo, 0) > rel.get(hi, 0):      # an oracle inversion
                    sw = list(top)
                    sw[a], sw[b] = sw[b], sw[a]
                    g = ndcg_at_k(sw, rel, 10) - base
                    if chain_of.get(hi) == chain_of.get(lo):
                        intra += 1
                        intra_gain += g
                    else:
                        inter += 1
                        inter_gain += g
                    if rel.get(lo, 0) > 0 and rel.get(hi, 0) > 0:
                        k = f"{rel[lo]}v{rel[hi]}"
                        gpair[k] = gpair.get(k, 0) + 1

    tot = intra + inter
    print(f"\n=== TRIAGE: oracle inversions inside the shipped top-10 ({tot:,} total)")
    print(f"  INTRA-chain (the tie-break KEY had its chance):  {intra:>7,} "
          f"({100*intra/max(tot,1):5.1f}%)  swap-gain sum {intra_gain:+.1f}")
    print(f"  INTER-chain (only the BLEND could have fixed it): {inter:>7,} "
          f"({100*inter/max(tot,1):5.1f}%)  swap-gain sum {inter_gain:+.1f}")
    print(f"  graded-only inversions by pair: "
          f"{dict(sorted(gpair.items(), key=lambda x: -x[1]))}")

    # ---------- SPECIFICITY, screened at the decision cell on the GRADED target ----------
    feats = {
        "low-id (incumbent)": lambda q, i: -idnum(i),
        "specificity (CE - name mean)": lambda q, i: h.ce[(q, i)] - nmean.get(names[i], 0.0),
        "name mean CE (popularity)": lambda q, i: nmean.get(names[i], 0.0),
        "name pool count": lambda q, i: float(ncount.get(names[i], 0)),
        "RRF prior rank-value": lambda q, i: h.priors[q].get(i, 0.0),
    }
    gpairs = {k: [] for k in feats}
    bpairs = {k: [] for k in feats}
    ncell = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        key = lambda i: (h.cont3(q, i), S.mp(q, i), S.ft(q, i), S.lt(q, i))
        cells = {}
        for i in h.pools[q]:
            cells.setdefault(key(i), []).append(i)
        for members in cells.values():
            if len(members) < 2:
                continue
            judged = [i for i in members if i in rel]
            for a in judged:
                for b in judged:
                    if rel[a] > rel[b]:
                        ncell += 1
                        for k, f in feats.items():
                            if rel[b] > 0:
                                gpairs[k].append((f(q, a), f(q, b)))
                            else:
                                bpairs[k].append((f(q, a), f(q, b)))

    print(f"\n=== WITHIN-CELL screen ({ncell:,} ordered judged pairs in shared cells)")
    print(f"{'feature':<32s}{'GRADED auc':>12s}{'ties':>7s}{'BINARY auc':>12s}{'ties':>7s}")
    for k in feats:
        ga, gt = auc(gpairs[k])
        ba, bt = auc(bpairs[k])
        print(f"{k:<32s}{(ga if ga else 0):>12.4f}{gt:>7.1f}"
              f"{(ba if ba else 0):>12.4f}{bt:>7.1f}")
    print("\nGRADED = both items relevant, different grades (the 45.5% pot).")
    print("BINARY = relevant vs non-relevant. Beat low-id in a column to be worth a run.")


if __name__ == "__main__":
    main()
