"""Weighted reciprocal-rank fusion of two or more run files (build order step 5).

score(doc) = sum over runs i of  weight_i / (k + rank_i(doc)),  rank 1-based.
Docs absent from a run contribute nothing for that run. Ties break by doc id
(as int when possible) for determinism.
"""

import argparse
import json
from pathlib import Path

from common import RUNS, setup_utf8, write_run_csv
from eval_ndcg import load_run_jsonl


def rrf_fuse(
    runs: list[dict[str, list[str]]],
    weights: list[float],
    k: int = 60,
    top_k: int = 10,
    qids: set[str] | None = None,
) -> dict[str, list[str]]:
    all_qids = set().union(*(set(r) for r in runs))
    if qids is not None:
        all_qids &= qids
    fused: dict[str, list[str]] = {}
    for qid in all_qids:
        scores: dict[str, float] = {}
        for run, w in zip(runs, weights):
            for rank, iid in enumerate(run.get(qid, []), start=1):
                scores[iid] = scores.get(iid, 0.0) + w / (k + rank)
        ordered = sorted(scores.items(), key=lambda x: (-x[1], _id_key(x[0])))
        fused[qid] = [iid for iid, _ in ordered[:top_k]]
    return fused


def _id_key(iid: str):
    try:
        return int(iid)
    except ValueError:
        return iid


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--k", type=int, default=60)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    assert len(args.runs) == len(args.weights), "one weight per run"

    runs = [load_run_jsonl(path) for path in args.runs]
    fused = rrf_fuse(runs, args.weights, k=args.k, top_k=args.top_k)

    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in fused.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{args.name}.csv", fused)
    print(f"wrote runs/{args.name}.jsonl and runs/{args.name}.csv")


if __name__ == "__main__":
    main()
