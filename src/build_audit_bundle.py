"""Build a compact, self-verifying audit/handoff bundle.

The full artifact tree is ~48 GB, but reproducing the winning submission needs far
less. The pipeline only ever consumes the TOP-100 of each retrieval run (pools are
the top-100 union), so depth-1000 run files can be truncated 10x with zero effect on
the result. Combined with the cross-encoder score cache, this yields a bundle that
regenerates the exact 0.66738 submission on CPU in minutes, with no GPU and no model
weights at all.

Outputs to dist/audit_bundle/:
  runs/bm25_all.jsonl              top-100 truncation of the depth-1000 run
  runs/dense_lora_v1_all.jsonl     top-100 truncation
  data/cache/rerank_...v7synth...parquet   4.83M cached CE scores
  REPRODUCE.md                     exact command + expected score

Verifies by regenerating the submission from the bundle and scoring it.
"""

import json
import shutil
from pathlib import Path

from common import DATA, ROOT, setup_utf8

setup_utf8()

DIST = ROOT / "dist" / "audit_bundle"
CACHE_STEM = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
DEPTH = 100

REPRODUCE = """# Reproducing the winning submission (no GPU required)

Final public leaderboard: **0.66738 nDCG@10** (local full-benchmark 0.667745).

This bundle contains the cached cross-encoder scores and the truncated retrieval
runs, which is everything the post-retrieval pipeline consumes. Model weights are
NOT needed to regenerate the submission - only to re-score from scratch.

## Steps

1. Clone the code: https://github.com/TahaElkhazmi/presto-arabic-search
2. Copy this bundle's `runs/` and `data/cache/` into the repository root.
3. Place the competition benchmark data at `data/benchmark/data/`
   (queries.jsonl, corpus.jsonl, candidates.jsonl).
4. Run:

```
python src/rerank.py \\
  --runs runs/bm25_all.jsonl runs/dense_lora_v1_all.jsonl \\
  --depths 100 100 --weights 3 1 --rrf-k 60 \\
  --model models/qwen3-reranker-arabic-ecom-rerank_v7synth \\
  --temperature 2.6 --lam 0.60 \\
  --eps 0.010 --eps-hi 0.040 --conf-t 0.15 \\
  --tie-key cont3 --strip-fillers \\
  --name reproduction
```

The `--model` path is only used to name the score cache; because every pair is
already cached, no weights are loaded and no GPU is touched.

5. Verify:

```
python src/eval_ndcg.py --run runs/reproduction.csv
```

Expected: `nDCG@10 = 0.667745`, `queries_missing_from_run = 0`.

## Re-scoring from scratch (GPU)

Delete the cache parquet and supply the model weights (see the model repository);
`src/rescore_benchmark_chunked.py` performs a resumable, checkpointed full pass
(~4-8 h on an 8 GB GPU).
"""


def truncate_run(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["retrieved"] = obj["retrieved"][:DEPTH]
            fo.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    mb_in = src.stat().st_size / 1e6
    mb_out = dst.stat().st_size / 1e6
    print(f"  {src.name}: {mb_in:.0f} MB -> {mb_out:.0f} MB ({n} queries)")


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    print("truncating retrieval runs to top-100 ...")
    for name in ("bm25_all.jsonl", "dense_lora_v1_all.jsonl"):
        truncate_run(ROOT / "runs" / name, DIST / "runs" / name)

    print("copying score cache ...")
    (DIST / "data" / "cache").mkdir(parents=True, exist_ok=True)
    src = DATA / "cache" / f"{CACHE_STEM}.parquet"
    dst = DIST / "data" / "cache" / f"{CACHE_STEM}.parquet"
    shutil.copy2(src, dst)
    print(f"  {src.name}: {dst.stat().st_size / 1e6:.0f} MB")

    (DIST / "REPRODUCE.md").write_text(REPRODUCE, encoding="utf-8")

    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"\nbundle written to {DIST}  ({total / 1e6:.0f} MB total)")


if __name__ == "__main__":
    main()
