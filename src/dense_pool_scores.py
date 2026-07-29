"""Dense retrieval SCORES for pool members, which nothing has ever cached.

The run files carry item ids only, so the dense voter's score magnitudes have never been
available to any experiment. That is why `probe_score_prior.py` could only test the
lexical half of the score-versus-rank question, as its own docstring records. The convex
combination fusion of Bruch et al. (TOIS 2023) needs BOTH voters' scores, so it needs this.

Cheap by construction: the corpus embeddings are already cached from the live retrieval
run, so this only embeds the slice's queries and dot-products them against the pool
members' rows. Embeddings are stored L2-normalized, so the dot product IS cosine, and the
same instruction prompt the retriever trains with is applied to queries (prompt_name
"query"; getting this wrong scores 0.0000, see trap 3.3).

GPU cost is a couple of minutes, but it MUST NOT overlap a training run: on 2026-07-29 a
second GPU process with 4.3 GB apparently free made Windows spill to system RAM instead of
raising OOM, costing 26x on the training step. Run it when the card is idle.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, DATA, MODELS, ROOT, load_jsonl, load_qid_file, setup_utf8
from probe_harness import stream_topk

setup_utf8()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", type=Path,
                    default=ROOT / "splits" / "holdout_remainder8012.txt")
    ap.add_argument("--model", type=Path, default=MODELS / "qwen3-arabic-ecom-lora_v1")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--max-seq", type=int, default=96)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=DATA / "cache" / "dense_pool_scores_lora_v1.parquet")
    args = ap.parse_args()

    cache = DATA / "cache" / f"{args.model.name}_len{args.max_seq}_{args.dtype}_corpus.fp16.npy"
    assert cache.exists(), f"corpus embedding cache missing: {cache}"

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    item_ids = [str(r["item_id"]) for r in corpus]
    row_of = {iid: n for n, iid in enumerate(item_ids)}
    doc_emb = np.load(cache)
    assert len(doc_emb) == len(item_ids), \
        f"corpus cache has {len(doc_emb)} rows for {len(item_ids)} items"
    print(f"corpus embeddings {doc_emb.shape} from {cache.name}")

    qids = sorted(load_qid_file(args.qids), key=int)
    qset = set(qids)
    qtext = {str(q["query_id"]): q["query"]
             for q in load_jsonl(BENCH / "queries.jsonl") if str(q["query_id"]) in qset}

    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", args.depth, qset)
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", args.depth, qset)
    pools = {}
    for q in qids:
        seen, pool = set(), []
        for run in (bm, dn):
            for iid in run.get(q, []):
                if iid not in seen:
                    seen.add(iid)
                    pool.append(iid)
        pools[q] = pool
    print(f"{len(qids):,} queries, {sum(len(p) for p in pools.values()):,} pool pairs")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.model), device="cuda",
                                model_kwargs={"torch_dtype": getattr(torch, args.dtype)})
    model.max_seq_length = args.max_seq
    assert "query" in (model.prompts or {}), \
        "checkpoint has no 'query' prompt; the asymmetric model scores 0.0000 without it"

    t0 = time.time()
    q_emb = model.encode([qtext[q] for q in qids], prompt_name="query",
                         batch_size=args.batch, normalize_embeddings=True,
                         show_progress_bar=True).astype(np.float32)
    print(f"encoded {len(qids):,} queries in {time.time() - t0:.0f}s")

    docs = torch.from_numpy(doc_emb.astype(np.float32)).cuda()
    rows = []
    for n, q in enumerate(qids):
        idx = torch.tensor([row_of[i] for i in pools[q]], device="cuda")
        qv = torch.from_numpy(q_emb[n]).cuda()
        sc = (docs.index_select(0, idx) @ qv).cpu().numpy()
        rows.extend((q, i, float(s)) for i, s in zip(pools[q], sc))

    df = pd.DataFrame(rows, columns=["qid", "iid", "score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df):,} dense pool scores -> {args.out}")
    print(f"  score range [{df['score'].min():.4f}, {df['score'].max():.4f}]")


if __name__ == "__main__":
    main()
