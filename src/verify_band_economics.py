"""Verify the band sizing that every remaining EV estimate depends on.

The claim: the low-confidence band concentrates most of the remaining loss, so a
judge that only ever sees band queries can still reach the whole gap. Specifically
conf3 < 0.02 is said to be ~41% of queries holding ~45% of loss, and conf3 < 0.05
~55% of queries holding ~63% of loss.

Everything downstream is priced off those fractions (capturing 15% of the conf<0.05
headroom = +0.027, 25% = +0.045), so they are reproduced here independently on the
clean holdout before any GPU time is spent against them.

"Loss" is 1 - nDCG@10 summed over queries. "Headroom" is the pool oracle minus what
we actually score, i.e. what a perfect re-ranker on our EXISTING pools would add.
Zero GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_qid_file, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.30)


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    print(f"holdout: {len(qids)} queries (disjoint from the detect-and-treat split)\n")

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    run = h.build_run(**LIVE_STACK)

    qrels = h.qrels
    nd = {q: ndcg_at_k(run[q], qrels[q], 10) for q in qids if q in qrels}
    qids = [q for q in qids if q in nd]
    n = len(qids)
    base = sum(nd.values()) / n
    print(f"live baseline on holdout: {base:.6f}")

    # pool oracle per query: best achievable from the pool we already have
    def oracle(q):
        rel = sorted(((g, i) for i, g in qrels[q].items()
                      if g > 0 and i in set(h.pools[q])), key=lambda x: -x[0])
        return ndcg_at_k([i for _, i in rel[:10]], qrels[q], 10)

    orc = {q: oracle(q) for q in qids}
    total_loss = sum(1.0 - nd[q] for q in qids)
    total_head = sum(orc[q] - nd[q] for q in qids)
    print(f"pool oracle on holdout   : {sum(orc.values()) / n:.6f}")
    print(f"total recoverable headroom (oracle - live): {total_head / n:+.4f} per query\n")

    print(f"{'conf3 <':>9s}{'queries':>10s}{'% of q':>9s}{'% of loss':>11s}"
          f"{'% of headroom':>15s}{'headroom/all-q':>16s}")
    print("-" * 70)
    for t in THRESHOLDS:
        band = [q for q in qids if h.conf[q] < t]
        if not band:
            continue
        loss_b = sum(1.0 - nd[q] for q in band)
        head_b = sum(orc[q] - nd[q] for q in band)
        print(f"{t:>9.3f}{len(band):>10d}{100 * len(band) / n:>8.1f}%"
              f"{100 * loss_b / total_loss:>10.1f}%{100 * head_b / total_head:>14.1f}%"
              f"{head_b / n:>+16.4f}")

    print("\nWHAT CAPTURE RATES ARE WORTH (full-benchmark nDCG, band = conf3 < 0.05)")
    band05 = [q for q in qids if h.conf[q] < 0.05]
    head05 = sum(orc[q] - nd[q] for q in band05) / n
    for frac in (0.05, 0.10, 0.15, 0.25, 0.40):
        print(f"  capture {100 * frac:>3.0f}% of the band headroom -> {frac * head05:+.4f}"
              f"   (reaches {base + frac * head05:.4f})")
    print(f"\n  the gap to beat is +0.0455 (0.66861 -> 0.71412)")


if __name__ == "__main__":
    main()
