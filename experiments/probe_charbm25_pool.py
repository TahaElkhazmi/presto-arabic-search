"""TASK 1 stage 0: is the char-ngram voter worth ANY GPU? Pure pool arithmetic.

The expensive half of a third voter is that every candidate it ADDS to the pool has
no cached cross-encoder score, so admitting new candidates costs a GPU pass. Before
paying that, measure what the new candidates are actually worth as an upper bound:
the pool ORACLE, i.e. what a perfect re-ranker would score on each pool.

This is the cheap kill. The cont3idf voter converted a +0.035 prior gain into +0.0001
end to end, so a third voter has to clear a real oracle bar to justify the pass.

Everything here reads candidates.jsonl, which is MEASUREMENT-ONLY under rule 2.4.b.
No number computed here may feed retrieval, training or tuning.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, setup_utf8
from eval_ndcg import load_candidates
from probe_harness import stream_topk

setup_utf8()

DEPTH = 100


def oracle(pool: set[str], qrel: dict[str, int]) -> float:
    """nDCG@10 of a PERFECT re-ranker over this pool.

    DCG comes from the best 10 grades PRESENT IN THE POOL; the ideal DCG comes from
    the TRUE qrel, so a pool that misses relevant items is correctly penalised. Taking
    the ideal from the pool instead is the obvious trap - it makes every oracle ~1.0.
    """
    got = sorted((max(g, 0) for g in (qrel.get(i, 0) for i in pool)), reverse=True)[:10]
    dcg = sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(got))
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:10]
    idcg = sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    qid_set = set(all_qids)
    qrels = load_candidates(BENCH / "candidates.jsonl")

    print("loading runs ...", flush=True)
    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", DEPTH, qid_set)
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", DEPTH, qid_set)
    ch = stream_topk(ROOT / "runs" / "charbm25_34gram_all.jsonl", DEPTH, qid_set)

    qs = [q for q in all_qids if q in qrels]
    rows = []
    for q in qs:
        rel = {i for i, g in qrels[q].items() if g > 0}
        base = set(bm.get(q, [])) | set(dn.get(q, []))
        new = set(ch.get(q, [])) - base
        rows.append(dict(
            q=q, n_rel=len(rel),
            base_hit=len(base & rel), both_hit=len((base | new) & rel),
            new_items=len(new), new_rel=len(new & rel),
            ch_only_hit=len(set(ch.get(q, [])) & rel),
            o_base=oracle(base, qrels[q]), o_both=oracle(base | new, qrels[q]),
        ))

    n = len(rows)
    tot_rel = sum(r["n_rel"] for r in rows)
    print(f"\nqueries with judgments: {n:,}   judged-relevant items: {tot_rel:,}\n")

    print("RECALL of judged-relevant items inside the candidate pool")
    print(f"  live pool (bm25 100 u dense 100)   {sum(r['base_hit'] for r in rows)/tot_rel:.4f}")
    print(f"  + char-ngram 100                   {sum(r['both_hit'] for r in rows)/tot_rel:.4f}")
    print(f"  char-ngram voter ALONE (100)       {sum(r['ch_only_hit'] for r in rows)/tot_rel:.4f}")

    print("\nPOOL SIZE and what the third voter adds")
    print(f"  mean NEW items admitted per query   {sum(r['new_items'] for r in rows)/n:.1f}")
    print(f"  mean NEW *relevant* per query       {sum(r['new_rel'] for r in rows)/n:.3f}")
    print(f"  queries gaining >=1 new relevant    {sum(1 for r in rows if r['new_rel'])/n:.1%}")

    ob = sum(r["o_base"] for r in rows) / n
    oo = sum(r["o_both"] for r in rows) / n
    print("\nPOOL ORACLE nDCG@10 (upper bound on any re-ranker over the pool)")
    print(f"  live pool          {ob:.4f}")
    print(f"  + char-ngram       {oo:.4f}")
    print(f"  ORACLE HEADROOM ADDED BY THE THIRD VOTER: {oo - ob:+.4f}")

    norec = [r for r in rows if r["base_hit"] == 0]
    print(f"\n  queries the live pool MISSES entirely (recall 0): {len(norec):,} "
          f"({len(norec)/n:.1%})")
    if norec:
        rec = sum(1 for r in norec if r["new_rel"]) / len(norec)
        print(f"    of those, char-ngram rescues >=1 relevant: {rec:.1%}")

    print("\nINTERPRETATION GATE: the end-to-end gain cannot exceed the oracle headroom,")
    print("and historically converts at well under 10% of it (cont3idf: +0.035 prior -> +0.0001).")


if __name__ == "__main__":
    main()
