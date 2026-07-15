"""Cross-encoder reranking of retrieval candidate pools with an RRF prior blend.

Pipeline per PHASE2_PLAN.md section 2:
  1. candidate pool per query = union of the top --depth ids from each input run
     (default intent: BM25 top-100 union dense top-100, ~200 dedup candidates).
  2. every (query, candidate) pair is scored by the seq-cls Qwen3 reranker with the
     official template + Arabic e-commerce instruction; scores are sigmoid in [0,1].
  3. final score = lam * ce + (1 - lam) * rrf / rrf_max, where the RRF prior is
     computed over the same runs at the same depths (default weights 2:1 and k=10,
     the tuned fusion settings) and rrf_max = sum(weights)/(k+1) scales it to ~[0,1].
     The prior guards against reranker blind spots; lam=1 is pure reranking.
  4. ties break by ascending integer id (the low-id prior from phase 1).

Scores are cached in data/cache/rerank_<model>_len<seq>_<dtype>.parquet keyed by
(query_id, item_id); lambda sweeps and repeat runs only pay for unseen pairs.

--lam-grid "0,0.2,..." skips run writing and instead reports nDCG@10 per lambda on
the selected query slice (tune on dev qids, verify once on val qids).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from common import BENCH, DATA, MODELS, RUNS, load_jsonl, select_queries, setup_utf8, write_run_csv
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from reranker_common import format_document, format_query


def _id_key(iid: str):
    try:
        return int(iid)
    except ValueError:
        return iid


def build_pools(
    runs: list[dict[str, list[str]]],
    depths: list[int],
    weights: list[float],
    rrf_k: int,
    qids: list[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Per query: deduped candidate union and the RRF prior over the same slices."""
    pools: dict[str, list[str]] = {}
    priors: dict[str, dict[str, float]] = {}
    for qid in qids:
        seen: dict[str, float] = {}
        pool: list[str] = []
        for run, depth, w in zip(runs, depths, weights):
            for rank, iid in enumerate(run.get(qid, [])[:depth], start=1):
                if iid not in seen:
                    seen[iid] = 0.0
                    pool.append(iid)
                seen[iid] += w / (rrf_k + rank)
        pools[qid] = pool
        priors[qid] = seen
    return pools, priors


def load_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {(q, i): s for q, i, s in zip(df["qid"], df["iid"], df["score"])}


def save_cache(path: Path, cache: dict[tuple[str, str], float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "qid": [q for q, _ in cache],
            "iid": [i for _, i in cache],
            "score": pd.array([cache[k] for k in cache], dtype="float32"),
        }
    )
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="candidate run jsonl files")
    p.add_argument("--depths", type=int, nargs="+", required=True, help="pool depth per run")
    p.add_argument("--weights", type=float, nargs="+", required=True, help="RRF prior weight per run")
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--model", type=Path, required=True, help="seq-cls CrossEncoder model dir")
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--n-queries", type=int, default=0, help="0 = all (after qid filter)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=160)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--lam", type=float, default=0.5, help="blend: lam*ce + (1-lam)*rrf/rrf_max")
    p.add_argument("--lam-grid", default=None, help="comma list; evaluate instead of writing runs")
    p.add_argument("--name", default=None, help="output run name (required unless --lam-grid)")
    p.add_argument("--top-k", type=int, default=10, help="csv truncation depth")
    args = p.parse_args()
    assert len(args.runs) == len(args.depths) == len(args.weights), "one depth+weight per run"
    assert args.name or args.lam_grid, "need --name to write runs, or --lam-grid to evaluate"

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, args.n_queries)
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    pools, priors = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)
    n_pairs = sum(len(v) for v in pools.values())
    print(f"{len(qids)} queries, {n_pairs} candidate pairs "
          f"(mean pool {n_pairs / max(len(qids), 1):.1f})")

    cache_path = DATA / "cache" / f"rerank_{args.model.name}_len{args.max_seq}_{args.dtype}.parquet"
    cache = load_cache(cache_path)
    todo = [(qid, iid) for qid in qids for iid in pools[qid] if (qid, iid) not in cache]
    print(f"cache {cache_path.name}: {len(cache)} scores, {len(todo)} to compute")

    if todo:
        from sentence_transformers.cross_encoder import CrossEncoder

        names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
        model = CrossEncoder(
            str(args.model),
            device="cuda",
            model_kwargs={"torch_dtype": getattr(torch, args.dtype)},
        )
        model[0].max_seq_length = args.max_seq
        pairs = [[format_query(qtext[qid]), format_document(names[iid])] for qid, iid in todo]
        t0 = time.time()
        scores = model.predict(pairs, batch_size=args.batch, show_progress_bar=True)
        print(f"scored {len(todo)} pairs in {(time.time() - t0) / 60:.1f} min")
        for (qid, iid), s in zip(todo, scores):
            cache[(qid, iid)] = float(s)
        save_cache(cache_path, cache)

    rrf_max = sum(args.weights) / (args.rrf_k + 1)

    def blended(lam: float) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
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
    print(f"wrote runs/{args.name}.jsonl (full pool) and runs/{args.name}.csv (top{args.top_k})")


if __name__ == "__main__":
    main()
