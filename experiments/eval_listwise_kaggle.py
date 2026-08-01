"""Score the Kaggle listwise judge, replaying every APPLICATION POLICY in post.

The GPU pass only RECORDS a permutation and its stepwise confidences; how much of
that permutation we actually use is a CPU decision, so all of it is replayed here for
free rather than burning a GPU run per policy.

Policies, in increasing order of how much they trust the judge:
  tie-group  - reorder ONLY inside contiguous equal-cont3 groups of the live top-20.
               This is the surgical injection point: it is exactly where our in-band
               key carries zero information and degenerates to ascending product id,
               so the bar the judge must clear is a coin flip. Measured ceiling for
               this policy is ~88% of the full top-20 re-rank ceiling.
  prefix-m   - move the judge's first m picks to the front, keep live order after.
               A replay of the earlier 200-query probe showed the entire measured
               negative came from picks at positions 4-10: full 10-pick -0.00156
               versus prefix-3 +0.00247.
  full       - replace the whole top-10 with the judge's order. Measured fatal as a
               cascade (-0.0105) and reported only as the pessimistic bound.

Capture rate is reported against the reachable top-20 re-rank ceiling, not the full
pool oracle, because the judge only ever sees the top 20. On that scale +0.02
all-query needs ~18% capture and the full +0.0455 gap needs ~40%.

POSITION BIAS is the confound that matters: a judge which simply echoes the order it
was shown scores about zero delta while looking cooperative. Pass --shuffle-file with
a shuffled-presentation run to measure order-independent signal.
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import ROOT, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
BAND_FRAC = 3308 / 8012   # conf3<0.02 share of the benchmark, measured on the holdout


def tie_groups(cont: list[float], tol: float = 1e-9) -> list[list[int]]:
    """Contiguous runs of equal containment in the shown order."""
    groups, cur = [], [0]
    for i in range(1, len(cont)):
        if abs(cont[i] - cont[i - 1]) <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orders", type=Path,
                   default=ROOT / "data" / "kaggle_listwise" / "out" / "listwise_orders.parquet")
    p.add_argument("--shuffle-file", type=Path, default=None,
                   help="a shuffled-presentation run, for the position-bias control")
    p.add_argument("--band-file", type=Path,
                   default=ROOT / "splits" / "probe_band600.txt")
    args = p.parse_args()

    if not args.orders.exists():
        raise SystemExit(f"missing {args.orders}")
    df = pd.read_parquet(args.orders)
    qids = sorted(load_qid_file(args.band_file), key=int)
    print(f"{len(qids)} band queries; judges: {sorted(df.model.unique())}\n")

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    shown = h.build_run(topn=20, **LIVE_STACK)   # exactly what the judge was shown
    base_run = h.build_run(**LIVE_STACK)
    qrels = h.qrels

    base = {q: ndcg_at_k(base_run[q], qrels[q], 10) for q in qids if q in qrels}

    # reachable ceiling: best top-10 obtainable by re-ordering the shown top-20
    def ceil20(q):
        rel = sorted(((qrels[q].get(i, 0), i) for i in shown[q]),
                     key=lambda x: -x[0])
        return ndcg_at_k([i for g, i in rel[:10] if g > 0], qrels[q], 10)

    ceil = {q: ceil20(q) for q in base}
    head = sum(ceil[q] - base[q] for q in base) / len(base)
    print(f"live baseline on band     : {sum(base.values()) / len(base):.6f}")
    print(f"top-20 re-rank ceiling    : {sum(ceil.values()) / len(ceil):.6f}")
    print(f"reachable headroom        : {head:+.4f} per band query")
    print(f"  (+0.02 all-query needs ~{0.02 / (head * BAND_FRAC) * 100:.0f}% capture; "
          f"+0.0455 needs ~{0.0455 / (head * BAND_FRAC) * 100:.0f}%)\n")

    def report(model, sub, label=""):
        judge, conf1 = {}, {}
        for r in sub.itertuples():
            judge[str(r.qid)] = json.loads(r.items)
            c = json.loads(r.conf)
            conf1[str(r.qid)] = c[0] if c else 0.0
        common = [q for q in base if q in judge]

        def score(fn):
            d = [ndcg_at_k(fn(q), qrels[q], 10) - base[q] for q in common]
            m = sum(d) / len(d)
            se = st.pstdev(d) / len(d) ** 0.5
            nz = sum(1 for x in d if abs(x) > 1e-9)
            return m, se, nz, sum(1 for x in d if x > 1e-9), sum(1 for x in d if x < -1e-9)

        def full(q):
            return judge[q][:10]

        def prefix(m):
            def f(q):
                pre = judge[q][:m]
                rest = [i for i in shown[q] if i not in set(pre)]
                return (pre + rest)[:10]
            return f

        def tiegroup(q):
            sh = shown[q]
            cont = [h.cont3(q, i) for i in sh]
            rank = {iid: r for r, iid in enumerate(judge[q])}
            out = list(sh)
            for g in tie_groups(cont):
                if len(g) < 2:
                    continue
                items = [sh[i] for i in g]
                items.sort(key=lambda x: rank.get(x, 10**6))
                for pos, i in zip(g, items):
                    out[pos] = i
            return out[:10]

        print("=" * 74)
        print(f"{model}{label}   n={len(common)}")
        print("=" * 74)
        print(f"  {'policy':<16s}{'delta':>11s}{'se':>9s}{'t':>7s}"
              f"{'capture':>9s}{'proj':>10s}{'w/l':>10s}")
        for name, fn in (("tie-group", tiegroup), ("prefix-1", prefix(1)),
                         ("prefix-3", prefix(3)), ("prefix-5", prefix(5)),
                         ("full-10", full)):
            m, se, nz, w, l = score(fn)
            print(f"  {name:<16s}{m:>+11.6f}{se:>9.6f}{m / se if se else 0:>+7.2f}"
                  f"{100 * m / head if head else 0:>8.1f}%{m * BAND_FRAC:>+10.6f}"
                  f"{f'{w}/{l}':>10s}")
        return judge, conf1, common

    for model in sorted(df.model.unique()):
        judge, conf1, common = report(model, df[df.model == model])

        # position-bias diagnostics computable from the live-order run alone
        echo = sum(1 for q in common if judge[q][0] == shown[q][0])
        ident = sum(1 for q in common if judge[q][:10] == shown[q][:10])
        print(f"  echoed the shown first item: {echo}/{len(common)} "
              f"({100 * echo / len(common):.0f}%)")
        print(f"  returned the identity order: {ident}/{len(common)} "
              f"({100 * ident / len(common):.0f}%)")
        print("  -> high values mean position bias, not agreement; the shuffle "
              "control is what separates them\n")

    if args.shuffle_file and args.shuffle_file.exists():
        sdf = pd.read_parquet(args.shuffle_file)
        print("\n########## POSITION-BIAS CONTROL (shuffled presentation) ##########")
        for model in sorted(sdf.model.unique()):
            sub = sdf[sdf.model == model]
            judge = {str(r.qid): json.loads(r.items) for r in sub.itertuples()}
            common = [q for q in base if q in judge]
            refound = sum(1 for q in common if judge[q][0] == shown[q][0])
            print(f"  {model}: shuffled first pick re-finds the live top-1 "
                  f"{refound}/{len(common)} ({100 * refound / len(common):.0f}%), "
                  f"chance = {100 / 20:.0f}%")
            print("    a judge with real order-independent signal should be well "
                  "above chance; near chance means it was only echoing position")
            report(model, sub, label=" [shuffled]")


if __name__ == "__main__":
    main()
