"""LEAD 3 pricing: what is the dense retrieval channel actually worth?

The dense v2 regression was never cleanly attributed, and the proposed isolated
experiment (keep the pairs_with_negatives queries, swap ONLY provided -> clean-mined
negatives) costs 6.4 h of training plus a corpus re-embed, a pool rebuild and a CE
re-score of every pair the new pools introduce. That is the most expensive lead on the
table, so it is worth pricing its CEILING before spending anything.

The ceiling is set by how much the dense channel contributes that BM25 does not. This
decomposes every judged-relevant item at the live pool depth into found-by-BM25-only,
found-by-dense-only, found-by-both, found-by-neither. The dense-only share is the whole
budget a better retriever competes for; the neither share is what recall work of any
kind could in principle recover.

Measurement only (rule 2.4.b): qrels are read to score, never to fit.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, setup_utf8
from eval_ndcg import load_candidates
from probe_harness import stream_topk

setup_utf8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[100, 150])
    args = ap.parse_args()

    qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    qrels = load_candidates(BENCH / "candidates.jsonl")
    qset = set(qids)

    for depth in args.depths:
        bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", depth, qset)
        dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", depth, qset)

        tot = both = bonly = donly = neither = 0
        # per-query: does the dense channel contribute any relevant item BM25 missed?
        q_dense_helps = q_eval = 0
        for q in qids:
            rel = {i for i, g in qrels.get(q, {}).items() if g > 0}
            if not rel:
                continue
            q_eval += 1
            b, d = set(bm.get(q, [])), set(dn.get(q, []))
            uniq_d = rel & d - b
            if uniq_d:
                q_dense_helps += 1
            tot += len(rel)
            both += len(rel & b & d)
            bonly += len(rel & b - d)
            donly += len(uniq_d)
            neither += len(rel - b - d)

        print(f"\n=== depth {depth} per channel ({q_eval:,} judged queries, "
              f"{tot:,} relevant items)")
        for label, n in (("found by BOTH", both), ("BM25 only", bonly),
                         ("DENSE only", donly), ("NEITHER (unreachable)", neither)):
            print(f"  {label:<24s} {n:>8,}  {100 * n / tot:6.2f}%")
        print(f"  pool recall (both+bm25+dense)   {100 * (tot - neither) / tot:6.2f}%")
        print(f"  queries where dense adds >=1 relevant item BM25 missed: "
              f"{q_dense_helps:,} ({100 * q_dense_helps / q_eval:.1f}%)")


if __name__ == "__main__":
    main()
