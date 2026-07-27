"""Score the Kaggle listwise judge's orderings against the live stack.

Reports, per judge model:
  - the paired delta on the band queries, with a t statistic (most queries move, so
    the noise band is the paired standard error, not the spread of nDCG itself);
  - the CAPTURE RATE against the pool oracle, which is the number that maps onto the
    leaderboard. Capturing ~26% of the band headroom is the win line against 0.71412;
    10-15% is the realistic good outcome;
  - confidence-gated variants, since the judge's own first-pick softmax is free and
    an earlier weak signal (n=4) suggested it separates good interventions from bad;
  - the projected full-benchmark effect, which is what a submission would actually
    move, since the band is only part of the benchmark.

Replacement versus blending: a REPLACEMENT cascade measured -0.0105 end to end even
using a better orderer, so a positive number here is necessary but not sufficient.
Deployment would blend and re-tune the stack; this script measures the ceiling of the
judge's ordering on the band, not the shippable number.
"""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from common import ROOT, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
ORDERS = ROOT / "data" / "kaggle_listwise" / "out" / "listwise_orders.parquet"
BAND_FRAC = 3308 / 8012   # conf<0.02 share of the benchmark, measured on the holdout


def main():
    if not ORDERS.exists():
        raise SystemExit(f"missing {ORDERS}; the kernel output has not been downloaded")
    df = pd.read_parquet(ORDERS)
    qids = sorted(load_qid_file(ROOT / "splits" / "probe_band600.txt"), key=int)
    print(f"{len(qids)} band queries, judges: {sorted(df.model.unique())}\n")

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    base_run = h.build_run(**LIVE_STACK)
    qrels = h.qrels

    base = {q: ndcg_at_k(base_run[q], qrels[q], 10) for q in qids if q in qrels}

    def oracle(q):
        rel = sorted(((g, i) for i, g in qrels[q].items()
                      if g > 0 and i in set(h.pools[q])), key=lambda x: -x[0])
        return ndcg_at_k([i for _, i in rel[:10]], qrels[q], 10)

    orc = {q: oracle(q) for q in base}
    head = sum(orc[q] - base[q] for q in base) / len(base)
    print(f"live baseline on band : {sum(base.values()) / len(base):.6f}")
    print(f"pool oracle on band   : {sum(orc.values()) / len(orc):.6f}")
    print(f"headroom on band      : {head:+.4f} per band query\n")

    for model in sorted(df.model.unique()):
        sub = df[df.model == model]
        ls, conf1 = {}, {}
        for r in sub.itertuples():
            items = json.loads(r.items)
            ls[str(r.qid)] = items[:10]
            c = json.loads(r.conf)
            conf1[str(r.qid)] = c[0] if c else 0.0

        common = [q for q in base if q in ls]
        d = [ndcg_at_k(ls[q], qrels[q], 10) - base[q] for q in common]
        mean = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        win = sum(1 for x in d if x > 1e-9)
        lose = sum(1 for x in d if x < -1e-9)
        cap = mean / head if head else 0.0

        print("=" * 66)
        print(f"{model}   n={len(common)}")
        print("=" * 66)
        print(f"  delta on band     : {mean:+.6f}   se {se:.6f}   t {mean / se:+.2f}")
        print(f"  {win} better, {lose} worse, {len(d) - win - lose} unchanged")
        print(f"  CAPTURE RATE      : {100 * cap:+.1f}% of band headroom")
        print(f"  projected full-benchmark: {mean * BAND_FRAC:+.6f}  "
              f"(band is {100 * BAND_FRAC:.1f}% of queries)")
        print(f"  -> would land near {0.66861 + mean * BAND_FRAC:.5f} "
              f"(win line 0.71412)")

        print("  confidence gating (apply the judge only above a first-pick threshold):")
        for t in (0.0, 0.5, 0.7, 0.9, 0.95):
            sel = [q for q in common if conf1[q] >= t]
            if not sel:
                continue
            dd = [ndcg_at_k(ls[q], qrels[q], 10) - base[q] for q in sel]
            m = sum(dd) / len(dd)
            gated = sum(dd) / len(common)  # others keep the live order
            print(f"    conf>={t:.2f}: {len(sel):>4d} q, delta-on-treated {m:+.5f}, "
                  f"band-wide {gated:+.5f}, projected {gated * BAND_FRAC:+.6f}")


if __name__ == "__main__":
    main()
