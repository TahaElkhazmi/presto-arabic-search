"""Resumable full-benchmark CE scoring, checkpointed per chunk (power-cut safe).

Unlike rerank.py (saves the cache once at the very end, so a power cut on a
multi-hour pass loses everything), this scores the 29,014-query benchmark pools in
qid chunks and SAVES the cache after each chunk. save_cache is atomic (tmp +
replace), so an interrupt mid-save never corrupts the existing cache. On restart,
already-cached pairs are skipped, so re-running the same command resumes with zero
rescoring (the CE-pass pattern that survived two app restarts cleanly).

Scores land in the SAME per-(model,seq,dtype) cache rerank.py reads, so after this
completes the submission file is generated cache-only with any operating point.
"""

import argparse
import time
from pathlib import Path

import torch

from common import BENCH, DATA, load_jsonl, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import build_pools, load_cache, save_cache
from reranker_common import format_document, format_query


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--chunk", type=int, default=1500, help="queries per checkpoint (~32 min each)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=160)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = p.parse_args()

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = load_jsonl(BENCH / "queries.jsonl")
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}
    pools, _ = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)

    cache_stem = f"rerank_{args.model.name}_len{args.max_seq}_{args.dtype}"
    cache_path = DATA / "cache" / f"{cache_stem}.parquet"
    cache = load_cache(cache_path)
    print(f"cache {cache_path.name}: {len(cache)} scores loaded", flush=True)

    # queries with any uncached pair, chunked
    pending = [qid for qid in qids if any((qid, iid) not in cache for iid in pools[qid])]
    total_todo = sum(1 for qid in pending for iid in pools[qid] if (qid, iid) not in cache)
    print(f"{len(pending)} queries need scoring, {total_todo} pairs; "
          f"chunk={args.chunk} queries", flush=True)
    if not pending:
        print("nothing to score; cache already complete.")
        return

    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(str(args.model), device="cuda",
                         model_kwargs={"torch_dtype": getattr(torch, args.dtype)})
    model[0].max_seq_length = args.max_seq

    t0 = time.time()
    done_pairs = 0
    for ci in range(0, len(pending), args.chunk):
        chunk_qids = pending[ci : ci + args.chunk]
        todo = [(qid, iid) for qid in chunk_qids for iid in pools[qid]
                if (qid, iid) not in cache]
        if not todo:
            continue
        pairs = [[format_query(qtext[q]), format_document(names[i])] for q, i in todo]
        scores = model.predict(pairs, batch_size=args.batch, show_progress_bar=False)
        for (q, i), s in zip(todo, scores):
            cache[(q, i)] = float(s)
        save_cache(cache_path, cache)  # CHECKPOINT (atomic)
        done_pairs += len(todo)
        el = (time.time() - t0) / 60
        rate = done_pairs / (el * 60) if el else 0
        eta = (total_todo - done_pairs) / rate / 60 if rate else 0
        print(f"chunk {ci // args.chunk + 1}/{-(-len(pending) // args.chunk)}: "
              f"+{len(todo)} pairs, cache {len(cache)}, {done_pairs}/{total_todo} "
              f"({rate:.0f} pairs/s, {el:.0f} min elapsed, ~{eta:.0f} min left) [saved]",
              flush=True)

    print(f"DONE: {len(cache)} total scores, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
