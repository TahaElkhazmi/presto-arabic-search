"""Price deeper pools BEFORE spending GPU on them, the same way the 4B cascade was priced.

Deeper pools were measured dead long ago, but that verdict was reached when the in-band key
fell through to raw ascending id. The key now carries four text levels and three blend
bonuses, and the whole recent campaign showed that the value of ADMITTING a candidate
depends on how well the system can order it once admitted. So the old verdict is worth
re-pricing rather than re-assuming.

This costs nothing: `runs/bm25_all.jsonl` and `runs/dense_lora_v1_all.jsonl` are DEPTH-1000
files, so the pool at any depth is a slice of what is already on disk, and the pool oracle at
that depth is computable directly.

The number that matters is the ORACLE HEADROOM added by going deeper, because the realistic
gain is a fraction of it - historically well under 10% (the char-ngram voter added +0.0059 of
headroom and converted to +0.0002). Against that, the GPU cost is real: every admitted
candidate needs a cross-encoder score it does not have.

candidates.jsonl is MEASUREMENT-ONLY under rule 2.4.b; this only prices a decision.
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

DEPTHS = (100, 150, 200, 300, 500)
RATE = 137.0   # pairs/s, 0.6B cross-encoder at batch 128 on this GPU


def oracle(pool, qrel):
    got = sorted((max(qrel.get(i, 0), 0) for i in pool), reverse=True)[:10]
    dcg = sum((2.0 ** g - 1.0) / math.log2(k + 2) for k, g in enumerate(got))
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:10]
    idcg = sum((2.0 ** g - 1.0) / math.log2(k + 2) for k, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    qid_set = set(all_qids)
    qrels = load_candidates(BENCH / "candidates.jsonl")
    qs = [q for q in all_qids if q in qrels]

    print("loading depth-1000 runs ...", flush=True)
    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", max(DEPTHS), qid_set)
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", max(DEPTHS), qid_set)

    base_pairs = None
    print(f"\n  {'depth':>6}{'mean pool':>11}{'new pairs':>13}{'GPU h':>8}"
          f"{'recall':>9}{'ORACLE':>10}{'headroom':>11}")
    ref = None
    for d in DEPTHS:
        tot_pool = tot_rel = hit = 0
        osum = 0.0
        for q in qs:
            pool = set(bm.get(q, [])[:d]) | set(dn.get(q, [])[:d])
            tot_pool += len(pool)
            rel = {i for i, g in qrels[q].items() if g > 0}
            tot_rel += len(rel)
            hit += len(pool & rel)
            osum += oracle(pool, qrels[q])
        mean_pool = tot_pool / len(qs)
        o = osum / len(qs)
        if d == 100:
            base_pairs, ref = tot_pool, o
        new = tot_pool - base_pairs
        print(f"  {d:>6}{mean_pool:>11.1f}{new:>13,}{new/RATE/3600:>8.1f}"
              f"{hit/tot_rel:>9.4f}{o:>10.6f}{o - ref:>+11.6f}")

    print("\n  headroom is the CEILING for a perfect re-ranker over the added candidates;")
    print("  the realistic conversion has been under 10% every time it was measured.")


if __name__ == "__main__":
    main()
