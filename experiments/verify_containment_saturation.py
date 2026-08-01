"""Verify the proposed mechanism for the zero cohort: CONTAINMENT SATURATION.

The claim to check: on zero-scoring queries a large fraction of the candidate pool
sits at the MAXIMUM char-3gram containment value, so cont3 is constant across the
whole near-tie head and our in-band re-sort degenerates to ascending product id
among dozens of textually identical matches. If true, no reshuffling of signals we
already own can help, because the items are genuinely exchangeable under every
text-based score, and only injected world knowledge can separate them.

This is the linchpin of the strategic conclusion, so it is reproduced here
independently rather than taken on trust. Also checks the competing hypothesis
(duplicate-name flooding of the top-10), which predicts FEW distinct names in the
head of zero queries.

Zero GPU.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    scored = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}

    import random
    zeros = sorted((q for q, v in scored.items() if v == 0.0), key=int)
    nonzeros = sorted(random.Random(5).sample(
        [q for q, v in scored.items() if v > 0.0], 1500), key=int)
    zeros = zeros[:1500]

    qids = sorted(set(zeros) | set(nonzeros), key=int)
    print(f"building pools for {len(qids)} queries (0 GPU)...", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    live = h.build_run(topn=10, **LIVE_STACK)
    print("done\n")

    def stats(qs, label):
        sat_frac, n_at_max, distinct_head, margins = [], [], [], []
        for q in qs:
            pool = h.pools[q]
            cs = [h.cont3(q, i) for i in pool]
            mx = max(cs)
            k = sum(1 for c in cs if c >= mx - 1e-12)
            sat_frac.append(k / len(pool))
            n_at_max.append(k)
            distinct_head.append(len({h.norm_name(i) for i in live[q][:10]}))
            margins.append(h.conf[q])
        print(f"  --- {label} (n={len(qs)}) ---")
        print(f"    pool fraction at MAX cont3 : mean {100 * st.mean(sat_frac):5.1f}%  "
              f"median {100 * st.median(sat_frac):5.1f}%")
        print(f"    items tied at MAX cont3    : mean {st.mean(n_at_max):5.1f}  "
              f"median {st.median(n_at_max):5.0f}")
        print(f"    distinct names in top-10   : mean {st.mean(distinct_head):5.2f}  "
              f"median {st.median(distinct_head):5.0f}")
        print(f"    CE top1-top3 margin        : mean {st.mean(margins):.4f}  "
              f"median {st.median(margins):.4f}")

    print("CONTAINMENT SATURATION AND HEAD STRUCTURE\n")
    stats(zeros, "ZERO cohort")
    stats(nonzeros, "non-zero comparison")

    print("\nINTERPRETATION")
    print("  On zero queries dozens of pool items share the TOP containment")
    print("  value (see the table above), so the in-band key is constant across")
    print("  the head and the re-sort")
    print("  falls back to ascending id among textually identical candidates.")
    print("  Duplicate-name flooding is the competing explanation: it predicts FEW")
    print("  distinct names in the top-10 of zero queries. Compare the two")
    print("  'distinct names in top-10' rows above to see whether that holds.")


if __name__ == "__main__":
    main()
