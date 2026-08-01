"""LEAD 1 verdict: does a filler-stripped CROSS-ENCODER input beat the raw one?

Compares two CE score caches over the SAME 800 filler-bearing queries, the same
pools, the same priors and the same post-processing. The only thing that differs is
whether the cross-encoder saw 'محتاج كفر شفاف ماك' or 'كفر شفاف ماك'.

Protocol rule 2 says re-tune before judging a swap, because the temperature and
lambda were fitted to the live score distribution and a changed CE input can shift
it. So both variants are reported twice: at the live constants (like-for-like) and
each at its own best over a small T x lam grid. The re-tuned column is INFLATED
because it is tuned and read on the same 800 queries; it is shown to check that the
like-for-like result is not an artifact of constants that happen to suit the
incumbent, not as a shippable number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

LIVE_CACHE = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
STRIP_CACHE = "rerank_v7synth_fillerstripped_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
QIDS = ROOT / "splits" / "probe_filler800.txt"


def build(qids, stem, T):
    h = Harness(qids, T=T, cache_stem=stem, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    return h


def score(h, qids, **cfg):
    run = h.build_run(**cfg)
    v = [ndcg_at_k(run[q], h.qrels[q], 10) for q in qids if q in run and q in h.qrels]
    return sum(v) / len(v)


def main():
    qids = sorted(load_qid_file(QIDS), key=int)
    print(f"{len(qids)} filler-bearing queries from the untouched remainder\n")

    hl = build(qids, LIVE_CACHE, 2.6)
    hs = build(qids, STRIP_CACHE, 2.6)
    if hs.missing_ce:
        print(f"!! stripped cache missing {hs.missing_ce} pairs; scoring incomplete")
        return

    base = score(hl, qids, **LIVE_STACK)
    cand = score(hs, qids, **LIVE_STACK)
    print("AT LIVE CONSTANTS (T=2.6, lam=0.60) - the like-for-like comparison")
    print(f"  raw-query CE      : {base:.6f}")
    print(f"  stripped-query CE : {cand:.6f}   ({cand - base:+.6f})\n")

    # per-query paired stats
    rl = hl.build_run(**LIVE_STACK)
    rs = hs.build_run(**LIVE_STACK)
    d = [ndcg_at_k(rs[q], hl.qrels[q], 10) - ndcg_at_k(rl[q], hl.qrels[q], 10)
         for q in qids]
    import statistics as st
    mean = sum(d) / len(d)
    se = st.pstdev(d) / len(d) ** 0.5
    win = sum(1 for x in d if x > 1e-9)
    lose = sum(1 for x in d if x < -1e-9)
    print(f"  paired: mean {mean:+.6f}  se {se:.6f}  t {mean / se:+.2f}")
    print(f"  {win} better, {lose} worse, {len(d) - win - lose} unchanged\n")

    print("EACH AT ITS OWN BEST (inflated: tuned and read on these same 800)")
    grid_T = [1.6, 2.0, 2.6, 3.2]
    grid_lam = [0.45, 0.55, 0.60, 0.70]
    for label, stem in (("raw-query CE", LIVE_CACHE), ("stripped-query CE", STRIP_CACHE)):
        best, arg = -1.0, None
        for T in grid_T:
            h = build(qids, stem, T)
            for lam in grid_lam:
                cfg = dict(LIVE_STACK)
                cfg["lam"] = lam
                s = score(h, qids, **cfg)
                if s > best:
                    best, arg = s, (T, lam)
        print(f"  {label:<20s} {best:.6f}  at T={arg[0]}, lam={arg[1]}")


if __name__ == "__main__":
    main()
