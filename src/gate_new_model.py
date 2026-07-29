"""Gate a newly trained cross-encoder against the live v7synth, end to end.

Protocol rule 3 (2026-07-27, the v8graded lesson): pure-CE ordering is NO LONGER a
sufficient gate. v8graded ordered BETTER than v7synth (0.6254 vs 0.6204) and made the
system WORSE (-0.0017), because the blend trusts a better orderer less. So pure-CE is
used here only as a cheap KILL screen (v3 collapsed to 0.4647 and could have been
killed in minutes), and the decision is taken end to end at the candidate's OWN
re-tuned operating point.

Two modes, matching the two slices they are allowed to touch:

  --mode retune   on a tuning slice: pure-CE screen, then a (temperature, lam) grid.
                  Fitting happens here and only here.
  --mode gate     on the CLEAN arbiter with the constants chosen by the retune pass,
                  frozen and passed in. +0.0015 ships, anything less does not.

Both modes hold the rest of the stack3 transform fixed, so the only thing that moves
is the cross-encoder. The baseline is always the live v7synth cache on the same
queries, so the comparison is paired and the t-statistic is a paired one.
"""

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ROOT, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_ensemble_current import STEM, T_LIVE, Stack3, load_raw
from probe_harness import _id_key, rescale

setup_utf8()


def pure_ce_ndcg(h, ce, qrels):
    """nDCG@10 of the pool ordered by the cross-encoder alone (low-id ties)."""
    out = []
    for q in h.qids:
        if q not in qrels:
            continue
        order = sorted(h.pools[q], key=lambda i: (-ce[(q, i)], _id_key(i)))
        out.append(ndcg_at_k(order[:10], qrels[q], 10))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True,
                    help="cache stem of the candidate, e.g. "
                         "rerank_qwen3-reranker-arabic-ecom-rerank_v9listnet_len160_bfloat16")
    ap.add_argument("--qids", type=Path, required=True, help="slice to evaluate on")
    ap.add_argument("--mode", choices=["retune", "gate"], default="retune")
    # WIDE by default. A new loss produces a different logit scale, so the incumbent
    # T=3.2 / lam=0.50 is the wrong prior for it: ListNet shapes a softmax over the group
    # while LambdaLoss shapes pairwise margins. Cost of a wide grid is dev-argmax
    # inflation (more cells, higher max by luck), which is why the retune pass only
    # PROPOSES constants and the verdict is taken on the clean arbiter at frozen values.
    ap.add_argument("--temps", default="1.0,1.6,2.2,2.6,3.2,4.0,5.0")
    ap.add_argument("--lams", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--temp", type=float, default=None, help="gate mode: frozen T")
    ap.add_argument("--lam", type=float, default=None, help="gate mode: frozen lam")
    args = ap.parse_args()

    qids = sorted(load_qid_file(args.qids), key=int)
    print(f"building stack3 harness over {len(qids):,} queries from {args.qids.name} ...",
          flush=True)
    S = Stack3(qids)
    h = S.h
    qrels = h.qrels
    pairs = [(q, i) for q in h.qids for i in h.pools[q]]

    raw, miss = load_raw(args.stem, pairs)
    print(f"candidate cache: {len(raw):,} scores on {len(pairs):,} pool pairs, "
          f"{miss:,} missing")
    if miss:
        print("  *** candidate does not cover the pools - score them first, stopping ***")
        return

    # PURE-CE KILL SCREEN. Not a gate (rule 3), just a floor: a collapse like v3's
    # 0.5931 -> 0.4647 is visible here for the price of the scoring pass already paid.
    # Temperature is a strictly monotone map, so it cannot reorder a pool: pure-CE is
    # computed once, on the raw scores, and is temperature-free by construction.
    base_pure = pure_ce_ndcg(h, h.ce, qrels)
    cand_pure = pure_ce_ndcg(h, raw, qrels)
    print("\npure-CE ordering (kill screen only, NOT the gate - rule 3)")
    print(f"  v7synth    {base_pure:.4f}")
    print(f"  candidate  {cand_pure:.4f}   ({cand_pure - base_pure:+.4f})"
          f"{'   <-- COLLAPSE, kill it' if cand_pure < base_pure - 0.05 else ''}")

    base_run = S.build(h.ce)
    qs = [q for q in h.qids if q in qrels]
    b = [ndcg_at_k(base_run[q], qrels[q], 10) for q in qs]
    print(f"\nbaseline stack3 with v7synth: {sum(b) / len(b):.6f}  (n={len(qs):,})")

    def measure(ce, lam):
        run = S.build(ce, lam=lam)
        c = [ndcg_at_k(run[q], qrels[q], 10) for q in qs]
        d = [y - x for x, y in zip(b, c)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        return sum(c) / len(c), m, (m / se if se else 0.0)

    if args.mode == "retune":
        print("\nre-tune grid (protocol rule 2: judge the swap at its OWN optimum)")
        print(f"{'T':>6s}{'lam':>7s}{'nDCG':>12s}{'delta':>12s}{'t':>7s}")
        results = []
        for T in [float(x) for x in args.temps.split(",")]:
            cand = {k: rescale(raw[k], T) for k in h.ce}
            for lam in [float(x) for x in args.lams.split(",")]:
                sc, m, t = measure(cand, lam)
                print(f"{T:>6.1f}{lam:>7.2f}{sc:>12.6f}{m:>+12.6f}{t:>+7.2f}", flush=True)
                results.append((m, t, T, lam))
        results.sort(key=lambda r: -r[0])
        print("\ntop 5 by delta:")
        for m, t, T, lam in results[:5]:
            print(f"  T={T:<5.1f} lam={lam:<5.2f}  {m:+.6f}  t {t:+.2f}")
        m, t, T, lam = results[0]
        # An argmax on the EDGE of the grid is not an optimum, it is a grid that was too
        # narrow; the project requires optima bracketed on both sides before believing them.
        Ts = sorted({r[2] for r in results})
        Ls = sorted({r[3] for r in results})
        edge = T in (Ts[0], Ts[-1]) or lam in (Ls[0], Ls[-1])
        if edge:
            print(f"\n  *** WARNING: argmax sits on the grid EDGE (T={T}, lam={lam}). "
                  f"Widen the grid;\n      an unbracketed optimum has not been shown to be "
                  f"one.")
        interior = any(abs(r[2] - T) > 1e-9 for r in results[:5]) and \
            any(abs(r[3] - lam) > 1e-9 for r in results[:5])
        print(f"\nARGMAX T={T} lam={lam} delta {m:+.6f}")
        print(f"  dev-argmax inflation is measured at 3-4x on this project, so treat "
              f"this as an upper bound;\n  the gate is the CLEAN arbiter, not this slice.")
        print(f"  top-5 spans more than one (T,lam): {interior}")
        print(f"\nnext: python src/gate_new_model.py --stem {args.stem} "
              f"--qids splits/clean_arbiter2784.txt --mode gate --temp {T} --lam {lam}")
    else:
        assert args.temp is not None and args.lam is not None, \
            "gate mode needs the frozen --temp and --lam from the retune pass"
        cand = {k: rescale(raw[k], args.temp) for k in h.ce}
        sc, m, t = measure(cand, args.lam)
        print(f"\nGATE on {args.qids.name} at frozen T={args.temp} lam={args.lam}")
        print(f"  candidate {sc:.6f}   delta {m:+.6f}   t {t:+.2f}")
        print(f"  VERDICT: {'PASS' if m >= 0.0015 else 'FAIL'} against the +0.0015 gate")


if __name__ == "__main__":
    main()
