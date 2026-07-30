"""How much is still reachable, decomposed into the three things that could give it.

The pool oracle (0.9525 at depth 100) has been the standing headroom number, but it mixes
together three very different jobs and so cannot tell us which one to work on:

  1. REORDER what we already show.  Take our own top-10 and sort it perfectly. Needs no
     new candidate and no new model - any post-processing rule is bounded by this.
  2. RESELECT from the pool.  Best 10 of the whole pool, perfectly ordered. The gap from
     (1) to (2) is what better SELECTION buys.
  3. Everything else is unreachable at this pool depth.

Also splits (1) into a binary part and a graded part, which matters because our
cross-encoder is trained on BINARY labels while the metric pays 2^grade - 1. If the
binary-perfect ordering already captures most of (1), grade modelling is worthless here;
if not, it is the one signal the training objective never encoded.

TWO BINARY ORACLES, and the difference is not cosmetic. A binary signal only says
relevant-or-not; it has no opinion INSIDE either class, so the split depends entirely on
what you assume happens there.
  - id-tiebreak: re-sort the relevant class by ascending product id. This throws away the
    ordering our blend produced and replaces it with the fallback prior.
  - PARTITION (order-preserving): keep our existing order inside each class. This is how
    this system actually deploys every discrete feature - see the trap-1 result, "the
    cascade sorts by (-feature, +id): a discrete feature PARTITIONS the cell and the
    incumbent still orders inside each class. Nothing is displaced."
The partition form is the correct upper bound on what a binary model could buy us, because
it is the only one a real feature could be plumbed into. This script originally reported
only the id-tiebreak form, which understated the binary oracle by 0.0258 and inflated the
"graded share" of the reorder headroom from 15.4% to 45.7%. Both are printed now.

Measurement only (rule 2.4.b): qrels are read to score, never to fit.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import _id_key, stream_topk

setup_utf8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="submission_v7synth_stack3.csv")
    ap.add_argument("--depth", type=int, default=100)
    args = ap.parse_args()

    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / args.run)
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    qset = set(all_qids)

    grades = Counter()
    for q in all_qids:
        grades.update(qrels.get(q, {}).values())
    tot = sum(grades.values())
    print(f"=== judged grade distribution ({tot:,} judgements)")
    for g in sorted(grades):
        print(f"  grade {g}: {grades[g]:>8,} ({100*grades[g]/tot:5.2f}%)  gain 2^g-1 = {2**g - 1}")

    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", args.depth, qset)
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", args.depth, qset)

    live = reorder = binonly = binpart = reselect = 0.0
    n = 0
    slots = Counter()
    for q in all_qids:
        rel = qrels.get(q)
        if not rel:
            continue
        n += 1
        top = run[q][:10]
        for i in top:
            g = rel.get(i)
            slots["unjudged" if g is None else (f"grade {g}")] += 1

        live += ndcg_at_k(top, rel, 10)
        # 1. perfect REORDER of the same ten
        reorder += ndcg_at_k(sorted(top, key=lambda i: (-rel.get(i, 0), _id_key(i))), rel, 10)
        # 1b. perfect BINARY reorder: relevant-vs-not only, ties by id (what a perfect
        #     binary-labelled model could do with our ten)
        binonly += ndcg_at_k(
            sorted(top, key=lambda i: (0 if rel.get(i, 0) > 0 else 1, _id_key(i))), rel, 10)
        # 1c. perfect binary PARTITION: same relevant-vs-not split, but our own order is
        #     preserved inside each class (sorted() is stable). This is the shape every
        #     discrete feature in this system is actually deployed in, and therefore the
        #     honest upper bound on what a binary model could buy.
        binpart += ndcg_at_k(
            sorted(top, key=lambda i: 0 if rel.get(i, 0) > 0 else 1), rel, 10)
        # 2. perfect RESELECT from the whole pool
        seen, pool = set(), []
        for r in (bm, dn):
            for i in r.get(q, []):
                if i not in seen:
                    seen.add(i)
                    pool.append(i)
        reselect += ndcg_at_k(sorted(pool, key=lambda i: (-rel.get(i, 0), _id_key(i)))[:10],
                              rel, 10)

    print(f"\n=== what our shipped top-10 is made of ({n:,} judged queries)")
    st = sum(slots.values())
    for k in sorted(slots, key=lambda x: (x == "unjudged", x)):
        print(f"  {k:<12s} {slots[k]:>8,} ({100*slots[k]/st:5.2f}%)")

    print(f"\n=== headroom decomposition")
    print(f"  live shipped system                          {live/n:.6f}")
    print(f"  + binary oracle, relevant class re-sorted by id  {binonly/n:.6f}  "
          f"(+{(binonly-live)/n:.6f})")
    print(f"  + binary oracle as a PARTITION (order kept)      {binpart/n:.6f}  "
          f"(+{(binpart-live)/n:.6f})   <- the honest bound")
    print(f"  + perfect GRADED reorder of our own 10           {reorder/n:.6f}  "
          f"(+{(reorder-live)/n:.6f})")
    print(f"  + perfect reselect from the pool                 {reselect/n:.6f}  "
          f"(+{(reselect-reorder)/n:.6f} beyond reorder)")
    print(f"\n  GRADED share of the reorder headroom:")
    print(f"    against the id-tiebreak binary oracle   "
          f"{100*(reorder-binonly)/max(reorder-live,1e-9):.1f}%   (overstated)")
    print(f"    against the PARTITION binary oracle     "
          f"{100*(reorder-binpart)/max(reorder-live,1e-9):.1f}%   <- correct")
    print(f"  ANY post-processing rule on the current top-10 is capped at "
          f"{reorder/n:.6f}")


if __name__ == "__main__":
    main()
