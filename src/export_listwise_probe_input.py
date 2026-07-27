"""Export the band probe input for the Kaggle listwise judge.

Ships everything the kernel needs and nothing it does not: for each band query, the
query text and the top-K candidates in LIVE ORDER with their product names. The
kernel returns only a permutation, so no labels ever leave this machine and nothing
qrels-derived is uploaded (rule 2.4.b stays clean by construction, and the kernel
literally cannot evaluate itself).

Band = queries whose CE top1-top3 margin is below --conf-max, drawn from the clean
holdout that the detect-and-treat sweep never touched. That band holds 46% of all
remaining loss at conf<0.02 and 62% at conf<0.05.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from common import ROOT, load_qid_file, setup_utf8
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
OUT_DIR = ROOT / "data" / "kaggle_listwise"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf-max", type=float, default=0.02)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=51)
    args = p.parse_args()

    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0

    band = [q for q in qids if h.conf[q] < args.conf_max]
    print(f"band conf < {args.conf_max}: {len(band)}/{len(qids)} queries")

    import random
    picked = sorted(random.Random(args.seed).sample(band, min(args.n, len(band))),
                    key=int)
    print(f"sampled {len(picked)}")

    cand = h.build_run(topn=args.k, qids=picked, **LIVE_STACK)

    rows = []
    for qid in picked:
        for rank, iid in enumerate(cand[qid][: args.k], start=1):
            rows.append({"qid": qid, "query": h.queries[qid], "rank": rank,
                         "item_id": iid, "name": h.names[iid]})
    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "listwise_probe_input.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(df)} rows, "
          f"{df.qid.nunique()} queries, K={args.k}")

    # the qid list is what the local evaluator scores against
    (ROOT / "splits" / "probe_band600.txt").write_text(
        "\n".join(picked) + "\n", encoding="utf-8")
    print(f"wrote splits/probe_band600.txt")
    print(f"\nsanity: mean pool {df.groupby('qid').size().mean():.1f} candidates/query")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
