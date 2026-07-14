"""Dense retrieval with the (fine-tuned) Qwen3 embedding model.

Traps handled per COMPETITION_GUIDE.md section 6:
  - queries are embedded with the stored instruction prompt (prompt_name="query")
  - documents are embedded raw (the stored "document" prompt is the empty string)
  - pooling/attention come from the checkpoint; do not override

Corpus embeddings are cached under data/cache/ keyed by model dir name and seq len.
Writes runs/<name>.jsonl at full --top-k depth and runs/<name>.csv truncated to 10.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import BENCH, DATA, MODELS, RUNS, load_jsonl, setup_utf8, write_run_csv


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=MODELS / "qwen3-embedding-0.6b-arabic-ecom")
    p.add_argument("--n-queries", type=int, default=1000, help="0 = all queries")
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=96)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = load_jsonl(BENCH / "queries.jsonl")
    if args.n_queries:
        queries = queries[: args.n_queries]
    name = args.name or f"dense_first{len(queries)}"
    item_ids = [str(row["item_id"]) for row in corpus]

    print(f"loading model from {args.model}")
    model = SentenceTransformer(
        str(args.model),
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = args.max_seq
    print(f"prompts={model.prompts} max_seq={model.max_seq_length}")

    cache_dir = DATA / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{args.model.name}_len{args.max_seq}_corpus.fp16.npy"
    if cache.exists():
        print(f"corpus embeddings from cache: {cache}")
        doc_emb = np.load(cache)
    else:
        t0 = time.time()
        doc_emb = model.encode(
            [row["name"] for row in corpus],
            batch_size=args.batch,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        doc_emb = doc_emb.astype(np.float16)
        np.save(cache, doc_emb)
        print(f"encoded corpus in {time.time() - t0:.0f}s -> {cache}")

    t0 = time.time()
    q_emb = model.encode(
        [row["query"] for row in queries],
        prompt_name="query",
        batch_size=args.batch,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float16)
    print(f"encoded {len(queries)} queries in {time.time() - t0:.0f}s")

    docs_t = torch.from_numpy(doc_emb).cuda()
    qs_t = torch.from_numpy(q_emb).cuda()
    k = min(args.top_k, docs_t.shape[0])
    run: dict[str, list[str]] = {}
    chunk = 256
    for i in range(0, qs_t.shape[0], chunk):
        sims = qs_t[i : i + chunk] @ docs_t.T
        top = torch.topk(sims, k, dim=1).indices.cpu().numpy()
        for j, row_idx in enumerate(range(i, min(i + chunk, qs_t.shape[0]))):
            run[str(queries[row_idx]["query_id"])] = [item_ids[t] for t in top[j]]

    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in run.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{name}.csv", {q: ids[:10] for q, ids in run.items()})
    print(f"wrote runs/{name}.jsonl (top{k}) and runs/{name}.csv (top10)")


if __name__ == "__main__":
    main()
