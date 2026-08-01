"""Protocol rule 2 for the depth-150 pools: re-tune before judging the change.

At the LIVE constants depth 150 measures +0.000278 full benchmark and -0.000371 on the
clean arbiter, i.e. null. That is not yet a verdict, because the constants were all fitted
against depth-100 pools and the pools materially changed: mean pool size went 166.4 ->
249.7. The near-tie chains are therefore denser, so the eps band geometry in particular is
mis-calibrated in a specific, predictable direction rather than a hypothetical one - with
50% more candidates packed into the same score range, a fixed eps chains together more
documents than it did at depth 100.

Fitted on the TUNE half only, read on the confirm half, exactly as every previous re-tune.
Two cautions carried from the log: dev-argmax inflation runs 3-4x here, and the last two
re-tunes (the first-token key, then the ft-bonus) were both NULL, so rule 2 says re-tune
after a change, not that the re-tune must win.
"""

import argparse
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_ensemble_current import Stack3
import probe_ensemble_current as P

setup_utf8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="0.020,0.030,0.045,0.060")
    ap.add_argument("--eps-hi", default="0.035,0.050,0.070,0.090")
    ap.add_argument("--top-eps", default="0.060,0.080,0.110")
    ap.add_argument("--lams", default="0.45,0.50,0.55")
    args = ap.parse_args()

    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")
    qids = sorted(hold, key=int)
    print(f"building DEPTH-150 harness over {len(qids):,} holdout queries ...", flush=True)
    S = Stack3(qids, depth=150)
    h = S.h
    qrels = h.qrels
    mean_pool = sum(len(h.pools[q]) for q in h.qids) / len(h.qids)
    print(f"  mean pool {mean_pool:.1f} (depth-100 live pools are 166.4)")

    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune, cf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])

    def halves(run):
        out = []
        for qs in (tune, cf):
            sel = [q for q in h.qids if q in qs and q in qrels]
            out.append([ndcg_at_k(run[q], qrels[q], 10) for q in sel])
        return out

    # reference: the SHIPPED depth-100 file on these same queries
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    b_t, b_c = [], []
    for qs, acc in ((tune, b_t), (cf, b_c)):
        for q in h.qids:
            if q in qs and q in qrels:
                acc.append(ndcg_at_k(live[q], qrels[q], 10))
    print(f"shipped depth-100 reference  tune {sum(b_t)/len(b_t):.6f}  "
          f"confirm {sum(b_c)/len(b_c):.6f}\n")

    def score(run):
        c_t, c_c = halves(run)
        dt = sum(y - x for x, y in zip(b_t, c_t)) / len(b_t)
        dd = [y - x for x, y in zip(b_c, c_c)]
        dc = sum(dd) / len(dd)
        se = st.pstdev(dd) / len(dd) ** 0.5
        return dt, dc, (dc / se if se else 0.0)

    dt, dc, t = score(S.build(h.ce))
    print(f"depth 150 at LIVE constants   tune {dt:+.6f}  confirm {dc:+.6f}  t {t:+.2f}")
    print(f"{'eps':>7s}{'eps_hi':>8s}{'top_eps':>9s}{'lam':>6s}"
          f"{'tune':>12s}{'confirm':>12s}{'t':>7s}")

    results = []
    for eps in [float(x) for x in args.eps.split(",")]:
        for hi in [float(x) for x in args.eps_hi.split(",")]:
            if hi < eps:
                continue
            for te in [float(x) for x in args.top_eps.split(",")]:
                for lam in [float(x) for x in args.lams.split(",")]:
                    P.EPS, P.EPS_HI, P.TOP_EPS = eps, hi, te
                    dt, dc, t = score(S.build(h.ce, lam=lam))
                    print(f"{eps:>7.3f}{hi:>8.3f}{te:>9.3f}{lam:>6.2f}"
                          f"{dt:>+12.6f}{dc:>+12.6f}{t:>+7.2f}", flush=True)
                    results.append((dt, dc, t, eps, hi, te, lam))
    P.EPS, P.EPS_HI, P.TOP_EPS = 0.030, 0.050, 0.080

    results.sort(key=lambda r: -r[0])
    print("\ntop 5 by TUNE half (the fitted slice), with their confirm-half readings:")
    for dt, dc, t, eps, hi, te, lam in results[:5]:
        print(f"  eps={eps:.3f} hi={hi:.3f} top={te:.3f} lam={lam:.2f}  "
              f"tune {dt:+.6f}  confirm {dc:+.6f}  t {t:+.2f}")
    dt, dc, t, eps, hi, te, lam = results[0]
    print(f"\nARGMAX on tune: eps={eps} eps_hi={hi} top_eps={te} lam={lam}")
    print(f"  tune {dt:+.6f} -> confirm {dc:+.6f} (t {t:+.2f})")
    print("  A re-tune that improves tune and not confirm is dev-argmax inflation, which "
          "is\n  measured at 3-4x on this project. Judge on the confirm column.")


if __name__ == "__main__":
    main()
