"""Independent verification: what does the PROVIDED candidates file score by itself?

`data/benchmark/data/candidates.jsonl` is the judged pool: per query a short list of
items WITH their relevance grades. It ships in the public HuggingFace benchmark repo
(`prestoai/arabic-ecom-search-bench`), which is why this competition has already had
two answer-key entries invalidated.

The question this answers: if someone used that file as their retrieval source and
ordered it with a model carrying little or no signal, what would the leaderboard
show? Because those pools are ~62% positive by construction, even a RANDOM ordering
scores far above anything a real full-catalog system achieves.

Reproduced here from scratch rather than taken on trust, because the conclusion
drives whether we spend the remaining days chasing 0.71412 or defending rank 2.
Measurement only; nothing here is used to build or tune a submission.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_run_csv, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k

setup_utf8()

TARGET = 0.71412  # the entry we are trying to explain
OURS = 0.66861


def _id_key(iid):
    try:
        return int(iid)
    except ValueError:
        return iid


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    qids = sorted(qrels)
    n = len(qids)

    sizes = [len(qrels[q]) for q in qids]
    npos = [sum(1 for g in qrels[q].values() if g > 0) for q in qids]
    print(f"candidates.jsonl: {n:,} queries")
    print(f"  candidates per query : mean {st.mean(sizes):.2f}  median {st.median(sizes):.0f}")
    print(f"  judged POSITIVE      : mean {st.mean(npos):.2f}  median {st.median(npos):.0f}")
    print(f"  positive rate        : {100 * sum(npos) / sum(sizes):.1f}% of each pool\n")

    def mean_ndcg(order_fn):
        return sum(ndcg_at_k(order_fn(q), qrels[q], 10) for q in qids) / n

    # 1. the file's own order
    file_order = mean_ndcg(lambda q: list(qrels[q].keys()))
    # 2. sorted by grade (the true oracle)
    oracle = mean_ndcg(lambda q: [i for i, _ in sorted(qrels[q].items(),
                                                       key=lambda kv: -kv[1])])
    # 3. ascending product id, no model at all
    lowid = mean_ndcg(lambda q: sorted(qrels[q], key=_id_key))
    # 4. random shuffles
    rands = []
    for seed in range(5):
        rng = random.Random(seed)

        def shuf(q, rng=rng):
            v = list(qrels[q])
            rng.shuffle(v)
            return v

        rands.append(mean_ndcg(shuf))

    print("nDCG@10 from ordering ONLY the provided candidate pools (no catalog search)\n")
    print(f"  {'strategy':<34s}{'nDCG@10':>10s}")
    print(f"  {'-' * 44}")
    print(f"  {'sorted by grade (true oracle)':<34s}{oracle:>10.5f}")
    print(f"  {'the file order as shipped':<34s}{file_order:>10.5f}")
    print(f"  {'ascending product id (no model)':<34s}{lowid:>10.5f}")
    print(f"  {'RANDOM shuffle (mean of 5)':<34s}{st.mean(rands):>10.5f}"
          f"   sd {st.pstdev(rands):.5f}")
    print(f"  {'-' * 44}")
    print(f"  {'>> the 0.71412 entry':<34s}{TARGET:>10.5f}")
    print(f"  {'>> us (full catalog, 107k items)':<34s}{OURS:>10.5f}")

    r = st.mean(rands)
    print(f"\nINTERPRETATION")
    print(f"  random-on-candidates      = {r:.5f}")
    print(f"  target minus random       = {TARGET - r:+.5f}")
    print(f"  target minus low-id       = {TARGET - lowid:+.5f}")
    if oracle > r:
        p = (TARGET - r) / (oracle - r)
        print(f"  the target sits {100 * p:.1f}% of the way from RANDOM to ORACLE "
              f"on these pools")
    print(f"\n  For contrast, our full-catalog system searches all 107,041 products "
          f"and scores {OURS:.5f}.")
    print(f"  Ordering the answer file at random beats it by {r - OURS:+.5f} "
          f"while using no model whatsoever.")


if __name__ == "__main__":
    main()
