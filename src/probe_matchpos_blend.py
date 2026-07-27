"""PHASE 2: match-position as an additive term in the BLEND, not just a tie-break.

Today match-position only orders candidates INSIDE an eps near-tie chain. Anything
that is not in such a chain never sees it, so a strong signal is confined to a narrow
decision. This tests giving it broader influence by adding beta * matchpos directly to
the blend score, which lets it reorder candidates whose blend scores are close but not
inside the same eps group.

The risk is precisely the failure mode that killed GATE-AraBert: a continuous term
applied globally displaces orderings that are already good, and at high beta it will
start overriding the cross-encoder itself. So beta is swept from far below the eps
scale (0.001, i.e. a tenth of eps=0.010) up to where it must dominate (0.10), and the
expected shape is a rise then a collapse. A monotone decline from beta=0 would mean
the signal has no headroom outside the band at all.

Baseline is the SHIPPED cascade, so every delta is incremental over what is on the
leaderboard. Tune half selects beta, confirm half reports it, and neither is the
arbiter, which is reserved for the final gate.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
SHIPPED = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30)
# Pass 2 of the sweep. The first pass rose monotonically to the grid maximum (0.100
# gave +0.001012 tune, +0.001953 confirm, t=+2.83), so the optimum lies beyond it and
# the value cannot be trusted yet. Extended until the expected collapse appears: at
# beta approaching 1.0 the position term rivals the entire blend, which must eventually
# override the cross-encoder and degrade.
BETAS = (0.0, 0.100, 0.200, 0.350, 0.600, 1.000, 2.000)


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

    cache: dict[tuple[str, str], float] = {}

    def matchpos(q, i):
        v = cache.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            cache[(q, i)] = v
        return v

    keyf = lambda q, i: h.cont3(q, i) * 1000.0 + matchpos(q, i)

    def run(beta, qs):
        bf = None if beta == 0 else (lambda q, i: beta * matchpos(q, i))
        return h.build_run(qids=qs, key=keyf, bonus_fn=bf, **SHIPPED)

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r]
        return sum(v) / len(v)

    base_t = run(0.0, tune)
    base_c = run(0.0, conf)
    b_t, b_c = sc(base_t, tune), sc(base_c, conf)
    print(f"shipped cascade baseline: tune {b_t:.6f}   confirm {b_c:.6f}\n")

    print(f"  {'beta':>8s}{'tune':>12s}{'delta':>12s}")
    best, best_beta = None, None
    for beta in BETAS:
        s = sc(run(beta, tune), tune)
        print(f"  {beta:>8.3f}{s:>12.6f}{s - b_t:>+12.6f}")
        if beta > 0 and (best is None or s > best):
            best, best_beta = s, beta

    print(f"\nCONFIRM at the tune-selected beta={best_beta} (not fitted on this half):")
    rc = run(best_beta, conf)
    d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base_c[q], qrels[q], 10)
         for q in conf]
    m = sum(d) / len(d)
    se = st.pstdev(d) / len(d) ** 0.5
    print(f"  {b_c:.6f} -> {sc(rc, conf):.6f}   delta {m:+.6f}   se {se:.6f}   "
          f"t {m / se if se else 0:+.2f}")
    print(f"  {sum(1 for x in d if x > 1e-9)} better, "
          f"{sum(1 for x in d if x < -1e-9)} worse, "
          f"{sum(1 for x in d if abs(x) <= 1e-9)} unchanged")
    print(f"\n  gate: +0.0015 incremental on the arbiter (checked only if this passes)")


if __name__ == "__main__":
    main()
