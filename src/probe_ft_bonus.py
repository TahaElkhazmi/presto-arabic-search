"""First-token position as an ADDITIVE BLEND term, not just a tie-break level.

The shipped stack already does this for match-position: `--pos-bonus 0.35` adds
0.35*matchpos directly to the blend score, which was worth +0.0017 on top of the same
signal's tie-break role. The reason it helped is that a tie-break level only orders
candidates INSIDE an eps near-tie chain, so a strong signal is confined to a narrow
decision; a bounded additive term also lets it reorder candidates whose blend scores are
close but not in the same chain.

First-token position is now a proven signal (+0.0030 as a tie-break level) and has NEVER
been tried as a blend term. This is the exact same move that already paid once, applied
to the newer signal.

Scale note: matchpos is 1/(1+k) in [0,1] while firsttok is 3/(1+k) in [0,3], so a given
beta has three times the effect here; the grid is set accordingly and is centred well
below 0.35.

Screening on the holdout halves; full benchmark and clean arbiter only if it survives.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
T_LIVE, BETA_MP = 3.2, 0.35
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2
GRID = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35)


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
    mpc, ftc = {}, {}

    def mp(q, i):
        v = mpc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            mpc[(q, i)] = v
        return v

    def ft(q, i):
        v = ftc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
            ftc[(q, i)] = v
        return v

    key = lambda q, i: h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT

    def build(beta_ft, qids=None):
        bf = (lambda q, i: BETA_MP * mp(q, i) + beta_ft * ft(q, i))
        return h.build_run(key=key, bonus_fn=bf, qids=qids, **LIVE)

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    base = build(0.0)
    bt, bc = sc(base, tune), sc(base, conf)
    print(f"baseline (matchpos bonus only): tune {bt:.6f}  confirm {bc:.6f}\n")

    print(f"  {'beta_ft':>8}{'tune d':>12}{'confirm d':>12}{'t':>8}")
    best = None
    for b in GRID:
        if b == 0.0:
            print(f"  {b:>8.2f}{0.0:>+12.6f}{0.0:>+12.6f}{0.0:>+8.2f}")
            continue
        rt, rc = build(b, tune), build(b, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {b:>8.2f}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}")
        if best is None or dt > best[1]:
            best = (b, dt, m)

    print(f"\n  best on tune: beta_ft={best[0]:.2f} (tune {best[1]:+.6f}, "
          f"confirm {best[2]:+.6f})")
    if best[0] in (GRID[1], GRID[-1]):
        print("  NOTE: optimum on a grid edge - unbracketed")
    print("  gate: confirm >= +0.0015 before it earns a full-benchmark run")


if __name__ == "__main__":
    main()
