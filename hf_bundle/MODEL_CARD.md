---
license: apache-2.0
language:
- ar
tags:
- information-retrieval
- reranking
- arabic
- e-commerce
---

# PrestoAI Arabic E-Commerce Search - system artifacts

Model and reproduction artifacts for team **Looooop Gain03**, final public leaderboard
**0.69413 nDCG@10** on the PrestoAI Arabic E-Commerce Search Retrieval Challenge, across
eighteen submissions each predicted in advance to within +-0.0014 by a local harness.

Code: https://github.com/TahaElkhazmi/presto-arabic-search
**Setup and step-by-step run instructions: `README.md`** in that repository (four
paths: verify the score with no GPU, run inference, rebuild the submission from these
weights, or retrain from scratch).
Technical report: `TECHNICAL_REPORT.md`, which carries the full experimental record
including nineteen diagnosed negative results.

When downloading, keep these local paths - the scripts expect them:

| This repo | Local path |
|---|---|
| `reranker-v7synth/` | `models/qwen3-reranker-arabic-ecom-rerank_v7synth/` |
| `retriever-lora-v1/` | `models/qwen3-arabic-ecom-lora_v1/` |
| `audit-bundle/runs/`, `audit-bundle/data/` | merge into the repository root |

## Contents

| Path | Size | Description |
|---|---|---|
| `audit-bundle/` | 162 MB | **Start here.** Regenerates the exact winning submission on CPU in ~2 minutes, no GPU and no weights required. See its `REPRODUCE.md`. |
| `reranker-v7synth/` | 2.3 GB | Final cross-encoder. Qwen3-Reranker-0.6B-seq-cls, three-stage LoRA fine-tune, merged. |
| `retriever-lora-v1/` | 2.3 GB | Dense retriever. Qwen3-Embedding-0.6B + LoRA, merged. |
| `adapters/` | 60 MB | LoRA adapters for provenance (mmarco_warm Stage-1, v7synth Stage-3, retriever). |

## Pipeline

BM25 (normalized Arabic, ascending-id tie-break) and the LoRA dense retriever each
supply top-100 candidates; these are fused with weighted reciprocal-rank fusion
(k=60, lexical:dense 3:1) into a pool and a prior. The cross-encoder scores every
pool pair and its sigmoid outputs are temperature-recalibrated at T=3.2. The final score is

```
0.50*CE + 0.50*prior + 0.35*match-position + 0.05*first-token-position
        + 0.05*highest-IDF-token-position + 0.05*prefix/typo-position
        + 0.04*proximity-bucket
```

A confidence-adaptive near-tie band (eps 0.030, widening to 0.050 when the CE top1-top3
margin < 0.10, and 0.080 over ranks 1-5) then re-sorts low-information regions by a
five-level lexicographic key: char-3gram query-in-name containment computed after
stripping eleven Libyan-dialect filler tokens, then match position, then first-token
position, then last-token position, then ascending product id.

## Reranker training stages

1. **Arabic IR warm-up** - BCE on 40k Arabic MS MARCO pairs.
2. **Synthetic corpus-grounded curriculum** - LambdaLoss on 60k groups generated from
   the competition catalog itself (query = size/unit-stripped name prefix; positive =
   the source product; hard negatives = char-3gram neighbour products the query
   lexically discriminates against). Labels are grounded by construction, addressing a
   measured 8-17% grounding rate in the provided labels.
3. **Provided-pairs listwise fine-tune** - LambdaLoss over per-query groups.

Pure-ordering quality by stage (identical pools/slice): pointwise BCE 0.524 <
zero-shot 0.561 < listwise 0.593 < +warm-up 0.603 < **+synthetic curriculum 0.621**.
A zero-shot 4B model of the same family scores 0.554.

## Provenance note

`synth_stage15_adapter` (the Stage-2 LoRA) was deleted during cleanup after its merge
completed. The merged Stage-2 checkpoint is the ground-truth artifact and the training
script regenerates the stage from scratch; the adapter-by-adapter chain from the stock
base is therefore not replayable for that stage. All other merges are verified at the
state-dict level in fp32 against base + delta.

## Licensing

Base checkpoints (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) are Apache-2.0; these
derivatives inherit that license. Training used the competition-provided data, the
public Arabic MS MARCO corpus, and the competition catalog. No benchmark relevance
judgment contributed a gradient to any weight in these artifacts.
