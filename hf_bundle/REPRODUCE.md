# Reproducing the final submission (no GPU required)

Final public leaderboard: **0.69413 nDCG@10** (local full-benchmark **0.694591**).

This bundle contains the cached cross-encoder scores and the retrieval runs TRUNCATED TO
TOP-100, which is everything the post-retrieval pipeline consumes at the shipped depth.
The pipeline unions the top 100 of each run, so the truncation is lossless for the command
below. Note the consequence: this bundle supports `--depths 100 100` only. A deeper setting
would silently build a SMALLER pool rather than error, so do not change those numbers.
Model weights are NOT needed to regenerate the submission, only to re-score from scratch.

## Steps

1. Clone the code: https://github.com/TahaElkhazmi/presto-arabic-search
2. Copy this bundle's `runs/` and `data/cache/` into the repository root.
3. Place the competition benchmark data at `data/benchmark/data/`
   (`queries.jsonl`, `corpus.jsonl`, `candidates.jsonl`).
4. Run:

```
python src/rerank.py \
  --runs runs/bm25_all.jsonl runs/dense_lora_v1_all.jsonl \
  --depths 100 100 --weights 3 1 --rrf-k 60 \
  --model models/qwen3-reranker-arabic-ecom-rerank_v7synth \
  --temperature 3.2 --lam 0.50 \
  --eps 0.030 --eps-hi 0.050 --conf-t 0.10 --conf-topk 3 \
  --tie-key cont3posftlt \
  --pos-bonus 0.35 --ft-bonus 0.05 --idf-bonus 0.05 \
  --pt-bonus 0.05 --px-bonus 0.04 \
  --top-eps 0.080 --top-k-eps 5 \
  --strip-fillers \
  --name reproduction
```

The `--model` path is only used to name the score cache.
Because every pair the command needs is already cached (4,827,501 pairs, exactly the depth-100
pool), no weights are loaded and no GPU is touched.

5. Verify:

```
python src/eval_ndcg.py --run runs/reproduction.csv
```

Expected: `nDCG@10 = 0.694591`, `queries_missing_from_run = 0`.

`runs/reproduction.csv` is byte-identical to the submitted `runs/submission_v7synth_stack5.csv`,
whose public leaderboard score is 0.69413. Check it yourself rather than taking it on trust:

```
python -c "import hashlib;print(hashlib.sha256(open('runs/reproduction.csv','rb').read()).hexdigest())"
```

Expected: `42b5ded221786d6cbcb8696f8d775a92ece3d82e38cc9c65cd9981a3860767c3`

The local-to-leaderboard offset is -0.00046, consistent with the -0.001 +- 0.0014 calibration
that held across all eighteen submissions.

## Reproducing earlier points on the leaderboard ladder

Every shipped submission is the same command with fewer post-processing options enabled, so any
point on the ladder can be re-derived independently from this one bundle.
These five are the standing regression checks for the codebase: any change that breaks one of
them is rejected regardless of how it scores.

| Expected local nDCG@10 | Public LB | Change from the command above |
|---|---|---|
| 0.694591 | 0.69413 | (as written above) |
| 0.693827 | 0.69362 | drop `--pt-bonus --px-bonus` |
| 0.689919 | 0.68946 | additionally drop `--idf-bonus --top-eps --top-k-eps`, use `--tie-key cont3posft` |
| 0.684938 | 0.68527 | additionally drop `--ft-bonus`, use `--tie-key cont3pos` |
| 0.667745 | 0.66738 | `--temperature 2.6 --lam 0.60 --eps 0.010 --eps-hi 0.040 --conf-t 0.15 --tie-key cont3` |

All five were re-verified from a clean state on 2026-07-30.

## Re-scoring from scratch (GPU)

Delete `data/cache/rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16.parquet`
and supply the model weights from `reranker-v7synth/` in this repository.
The full benchmark is 4,827,501 pairs at roughly 139 pairs/second on an RTX 4060 (bf16,
`max_seq 160`), so allow about 10 GPU-hours.
The scoring pass is chunked and checkpointed, so re-running the same command resumes from
whatever is already cached.

To rebuild the retrieval runs as well, see section 4 of the repository README; BM25 is CPU-only
and the dense pass needs the `retriever-lora-v1/` weights plus about one GPU-hour to embed the
catalog.

## Bundle contents

| path | what it is |
|---|---|
| `runs/bm25_all.jsonl` | BM25 retrieval, truncated to top-100, all 29,014 queries |
| `runs/dense_lora_v1_all.jsonl` | dense retrieval, truncated to top-100, all 29,014 queries |
| `data/cache/rerank_...v7synth...parquet` | cross-encoder scores for all 4,827,501 depth-100 pool pairs |
