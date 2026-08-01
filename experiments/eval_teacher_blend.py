"""Blend externally-scored teacher logits with the RRF prior; report nDCG@10 per lambda.

Same pools, prior, blend formula, and tie-break as src/rerank.py; the only
difference is that pair scores come from a parquet produced off-machine
(columns qid, iid, logit — sigmoid applied here in fp32 — or a ready 'score').
Optionally writes a run at a fixed --lam like rerank.py does.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import BENCH, RUNS, load_jsonl, select_queries, setup_utf8, write_run_csv
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from rerank import _id_key, build_pools


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--lam-grid", default=None, help="comma list; evaluate against qrels")
    p.add_argument("--lam", type=float, default=None, help="write a run at this lambda")
    p.add_argument("--name", default=None, help="output run name for --lam mode")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()
    assert args.lam_grid or (args.lam is not None and args.name), "need --lam-grid, or --lam + --name"

    df = pd.read_parquet(args.scores)
    if "score" in df.columns:
        sc = df["score"].to_numpy(dtype=np.float32)
    else:
        sc = 1.0 / (1.0 + np.exp(-df["logit"].to_numpy(dtype=np.float32)))
    cache = {(q, i): float(s) for q, i, s in zip(df["qid"], df["iid"], sc)}
    print(f"loaded {len(cache)} teacher scores from {args.scores.name}")

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, 0)
    qids = [str(q["query_id"]) for q in queries]
    pools, priors = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)
    missing = sum(1 for qid in qids for iid in pools[qid] if (qid, iid) not in cache)
    print(f"{len(qids)} queries, {sum(len(v) for v in pools.values())} pool pairs, {missing} missing scores")
    assert missing == 0, "teacher scores do not cover the pools"

    rrf_max = sum(args.weights) / (args.rrf_k + 1)

    def blended(lam: float) -> dict[str, list[str]]:
        out = {}
        for qid in qids:
            scored = [
                (lam * cache[(qid, iid)] + (1 - lam) * priors[qid][iid] / rrf_max, iid)
                for iid in pools[qid]
            ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            out[qid] = [iid for _, iid in scored]
        return out

    if args.lam_grid:
        qrels = load_candidates(BENCH / "candidates.jsonl")
        print(f"{'lam':>6s}  {'nDCG@10':>8s}")
        for lam in [float(x) for x in args.lam_grid.split(",")]:
            run = {qid: ids[:10] for qid, ids in blended(lam).items()}
            res = evaluate_run(run, qrels)
            print(f"{lam:6.2f}  {res['nDCG@10']:8.4f}   ({res['queries_evaluated']} queries)")
        return

    run = blended(args.lam)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in run.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{args.name}.csv", {q: ids[: args.top_k] for q, ids in run.items()})
    print(f"wrote runs/{args.name}.jsonl and runs/{args.name}.csv")


if __name__ == "__main__":
    main()
