"""Evaluate an ensemble of cross-encoder score caches blended with the RRF prior.

final = lam * mean_k(CE_model_k) + (1 - lam) * RRF_prior / rrf_max, low-id ties.
Each --caches parquet is a per-(model,seq,dtype) score cache written by rerank.py
(columns qid, iid, score in [0,1]); the ensemble averages the models' scores on
each pool pair before blending, so it reuses all existing caches at zero GPU cost.
With --logit-mean the average runs in logit space (clipped to [-12, 12]) and is
mapped back through sigmoid, so saturated sigmoids do not compress the margins.
Reports nDCG@10 per lambda on the selected slice (dev to tune, val once).
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import math
from pathlib import Path

import pandas as pd

from common import BENCH, load_jsonl, select_queries, setup_utf8
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from rerank import _id_key, build_pools, load_cache


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--caches", type=Path, nargs="+", required=True, help="rerank score caches to average")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--lam-grid", default="0,0.2,0.25,0.3,0.35,0.4,0.5")
    p.add_argument("--logit-mean", action="store_true", help="average in logit space, sigmoid back")
    args = p.parse_args()
    assert len(args.runs) == len(args.depths) == len(args.weights)

    caches = [load_cache(path) for path in args.caches]
    for path, c in zip(args.caches, caches):
        print(f"cache {path.name}: {len(c)} scores")

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, 0)
    qids = [str(q["query_id"]) for q in queries]
    pools, priors = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)

    missing = sum(1 for qid in qids for iid in pools[qid]
                  for c in caches if (qid, iid) not in c)
    assert missing == 0, f"{missing} pool pairs missing from some cache; score them first"

    rrf_max = sum(args.weights) / (args.rrf_k + 1)

    def logit(s):
        s = min(max(s, 1e-12), 1.0 - 1e-12)
        return max(-12.0, min(12.0, math.log(s / (1.0 - s))))

    if args.logit_mean:
        def ce_mean(qid, iid):
            m = sum(logit(c[(qid, iid)]) for c in caches) / len(caches)
            return 1.0 / (1.0 + math.exp(-m))
    else:
        def ce_mean(qid, iid):
            return sum(c[(qid, iid)] for c in caches) / len(caches)

    qrels = load_candidates(BENCH / "candidates.jsonl")
    mode = "logit-mean" if args.logit_mean else "sigmoid-mean"
    print(f"{len(qids)} queries, {len(caches)} models, {mode}\n{'lam':>6s}  {'nDCG@10':>8s}")
    for lam in [float(x) for x in args.lam_grid.split(",")]:
        run = {}
        for qid in qids:
            scored = [
                (lam * ce_mean(qid, iid) + (1 - lam) * priors[qid][iid] / rrf_max, iid)
                for iid in pools[qid]
            ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            run[qid] = [iid for _, iid in scored][:10]
        res = evaluate_run(run, qrels)
        print(f"{lam:6.2f}  {res['nDCG@10']:8.4f}   ({res['queries_evaluated']} queries)")


if __name__ == "__main__":
    main()
