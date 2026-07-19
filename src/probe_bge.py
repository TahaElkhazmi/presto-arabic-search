"""Zero-shot BGE-reranker-v2-m3 probe (Phase 6 Lever 4).

Encoder-only (XLM-RoBERTa) multilingual cross-encoder. Tests whether its error
distribution is orthogonal to the Qwen3 v2 reranker (i.e. an ensemble lift).

CRITICAL: bge takes RAW [query, passage] pairs, NOT the Qwen3 chat template in
reranker_common. It does not go through rerank.py's format_query/format_document.
Writes the SAME (qid, iid, score) parquet cache as rerank.py so eval_ensemble.py
can average the bge and v2 caches with no new code.
"""
import argparse
import time
from pathlib import Path

import torch

from common import BENCH, DATA, load_jsonl, select_queries, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import build_pools, load_cache, save_cache


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)  # for identical pools
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--qids", type=Path, required=True)                 # val_rand500 for the probe
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--cache-name", default="bge-reranker-v2-m3")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=160)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = p.parse_args()

    run_files = [load_run_jsonl(x) for x in args.runs]
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, 0)
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    pools, _ = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)

    cache_path = DATA / "cache" / f"rerank_{args.cache_name}_len{args.max_seq}_{args.dtype}.parquet"
    cache = load_cache(cache_path)
    todo = [(q, i) for q in qids for i in pools[q] if (q, i) not in cache]
    n_pairs = sum(len(v) for v in pools.values())
    print(f"{len(qids)} queries, {n_pairs} pairs, {len(todo)} to score")
    if not todo:
        print("nothing to score")
        return

    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}

    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(args.model, device="cuda",
                         model_kwargs={"torch_dtype": getattr(torch, args.dtype)})
    model[0].max_seq_length = args.max_seq

    pairs = [[qtext[q], names[i]] for q, i in todo]     # RAW pairs, no template
    t0 = time.time()
    scores = model.predict(pairs, batch_size=args.batch, show_progress_bar=True)
    lo, hi = float(min(scores)), float(max(scores))
    print(f"scored {len(todo)} in {(time.time()-t0)/60:.1f} min; score range [{lo:.4f}, {hi:.4f}]")
    # SANITY: expect [0,1] (ST applies sigmoid for a 1-label head, as with Qwen v2).
    # If the range shows raw logits instead, sigmoid before caching to match the v2 space.
    if lo < 0.0 or hi > 1.0:
        print("range outside [0,1]: applying sigmoid to match the v2 cache space")
        scores = [1.0 / (1.0 + torch.exp(-torch.tensor(float(s))).item()) for s in scores]
    for (q, i), s in zip(todo, scores):
        cache[(q, i)] = float(s)
    save_cache(cache_path, cache)
    print(f"cache written: {cache_path.name} ({len(cache)} rows)")


if __name__ == "__main__":
    main()
