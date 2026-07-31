"""Own nDCG@10 evaluator, implemented from the competition metric specification.

Cross-checked against the official data/benchmark/evaluate.py; both must agree to 4
decimals on an identical run file before any number from here is trusted.

Accepts two run formats:
  .csv    submission format: header query_id,product_id; ids space-separated, best first
          (duplicates dropped after first occurrence, truncated to 10 - Kaggle semantics)
  .jsonl  official format: {"query_id": ..., "retrieved": [{"item_id": ...}, ...]}
          (loaded verbatim like the official scorer: no dedupe, no truncation)

Scoring follows the official scorer: mean over queries present in BOTH run and qrels.
"""

import argparse
import json
import math
from pathlib import Path

from common import BENCH, load_qid_file, load_run_csv, setup_utf8


def load_candidates(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qrels[str(obj["query_id"])] = {
                str(c["item_id"]): int(c["relevance"]) for c in obj["candidates"]
            }
    return qrels


def load_run_jsonl(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            retrieved = obj.get("retrieved", obj.get("results", []))
            ids = []
            for entry in retrieved:
                iid = str(entry.get("item_id", entry.get("id", ""))) if isinstance(entry, dict) else str(entry)
                if iid:
                    ids.append(iid)
            run[str(obj["query_id"])] = ids
    return run


def ndcg_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, iid in enumerate(retrieved[:k], start=1):
        grade = max(qrel.get(iid, 0), 0)
        if grade > 0:
            dcg += (2.0 ** grade - 1.0) / math.log2(i + 1)
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)
    idcg = 0.0
    for i, grade in enumerate(ideal[:k], start=1):
        idcg += (2.0 ** grade - 1.0) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    relevant = {iid for iid, g in qrel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    recall_ks: list[int] | None = None,
) -> dict:
    qids = sorted(set(run) & set(qrels))
    ndcgs = [ndcg_at_k(run[qid], qrels[qid], k) for qid in qids]
    out = {
        "queries_evaluated": len(qids),
        "queries_missing_from_run": len(set(qrels) - set(run)),
        f"nDCG@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }
    for rk in recall_ks or []:
        recalls = [recall_at_k(run[qid], qrels[qid], rk) for qid in qids]
        out[f"Recall@{rk}"] = sum(recalls) / len(recalls) if recalls else 0.0
    return out


def main():
    setup_utf8()
    p = argparse.ArgumentParser(description="Own nDCG@10 evaluator")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--recall", type=int, nargs="*", default=[])
    p.add_argument("--candidates", type=Path, default=BENCH / "candidates.jsonl")
    p.add_argument("--qids", type=Path, default=None, help="restrict scoring to these query ids")
    args = p.parse_args()

    qrels = load_candidates(args.candidates)
    if args.run.suffix == ".csv":
        run = load_run_csv(args.run)
    else:
        run = load_run_jsonl(args.run)
    if args.qids is not None:
        keep = load_qid_file(args.qids)
        run = {qid: ids for qid, ids in run.items() if qid in keep}

    result = evaluate_run(run, qrels, k=args.k, recall_ks=args.recall)
    for key, val in result.items():
        if isinstance(val, float):
            print(f"{key:28s} = {val:.6f}")
        else:
            print(f"{key:28s} = {val}")


if __name__ == "__main__":
    main()
