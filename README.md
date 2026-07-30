# PrestoAI Arabic E-Commerce Search

Hybrid retrieval + calibrated cross-encoder re-ranking for Arabic (Libyan-dialect)
e-commerce product search.
Final public leaderboard: **0.69362 nDCG@10** (team Looooop Gain03), across seventeen
submissions each predicted in advance to within +-0.0014 by the local harness.

| Document | Contents |
|---|---|
| [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) | The full writeup: architecture, evaluation methodology, the positive path, nineteen diagnosed negative results, design rationale and compliance disclosures. **Read this first.** |
| This file | How to set up, run and reproduce the system. |

Model weights and a zero-GPU reproduction bundle live in a companion Hugging Face
repository, `TahaElkhazmi/presto-arabic-search-models`, because they are too large for
git. Its `audit-bundle/REPRODUCE.md` reproduces the submitted file without a GPU and
without model weights; the same instructions are mirrored at
[`hf_bundle/REPRODUCE.md`](hf_bundle/REPRODUCE.md).

---

## 0. Which path do you want?

| Goal | Section | Time | GPU |
|---|---|---|---|
| **Verify the winning score** without downloading weights | [1](#1-verify-the-winning-submission-fastest-path) | ~2 min | no |
| **Score a query** with the trained models (inference) | [2](#2-running-the-models-inference) | ~5 min | helps |
| **Rebuild the submission** from the model weights | [3](#3-rebuild-the-submission-from-weights) | 4-8 h | yes |
| **Retrain everything** from scratch | [4](#4-full-training-pipeline-from-scratch) | ~20 h | yes |

---

## Environment setup (needed for every path)

Python 3.12. From the repository root:

```bash
python -m venv .venv
```

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
```

On Linux/macOS the interpreter is `.venv/bin/python`; substitute it everywhere below.

Two things that will bite you otherwise:

- **`sentence-transformers>=5.6` is mandatory.** Earlier versions cannot load these
  checkpoints. `requirements-lock.txt` pins the exact verified versions.
- **Arabic text crashes the Windows console** unless stdout is UTF-8. Every entry
  point calls `common.setup_utf8()` already; if you write your own script, call it
  first or you will get `UnicodeEncodeError` on any print.

Place the competition data at `data/benchmark/data/` (`queries.jsonl`,
`corpus.jsonl`, `candidates.jsonl`). Download model folders from the HF repo into
`models/` keeping these names:

| HF folder | Local path |
|---|---|
| `reranker-v7synth/` | `models/qwen3-reranker-arabic-ecom-rerank_v7synth/` |
| `retriever-lora-v1/` | `models/qwen3-arabic-ecom-lora_v1/` |

---

## 1. Verify the winning submission (fastest path)

No weights, no GPU. Download `audit-bundle/` from the HF repo, copy its `runs/` and
`data/cache/` into the repository root, then:

```bash
./.venv/Scripts/python.exe src/rerank.py \
  --runs runs/bm25_all.jsonl runs/dense_lora_v1_all.jsonl --depths 100 100 \
  --weights 3 1 --rrf-k 60 \
  --model models/qwen3-reranker-arabic-ecom-rerank_v7synth \
  --temperature 3.2 --lam 0.50 --eps 0.030 --eps-hi 0.050 --conf-t 0.10 --conf-topk 3 \
  --tie-key cont3posftlt --pos-bonus 0.35 --ft-bonus 0.05 --idf-bonus 0.05 \
  --top-eps 0.080 --top-k-eps 5 --strip-fillers --name reproduction
```

Every cross-encoder score is already cached, so `--model` is used only to name the
cache; no weights are loaded and the GPU is never touched. Then verify:

```bash
./.venv/Scripts/python.exe src/eval_ndcg.py --run runs/reproduction.csv
```

Expected: `nDCG@10 = 0.693827`, `queries_missing_from_run = 0`.
That file is byte-identical to the submitted `runs/submission_v7synth_stack3.csv`,
whose public leaderboard score is **0.69362** (local minus 0.00021).

### Reproducing the earlier submissions too

Each shipped submission is the same command with fewer post-processing options, so any
point on the leaderboard ladder can be re-derived and checked independently:

| Expected nDCG@10 | Public LB | Change from the command above |
|---|---|---|
| 0.693827 | 0.69362 | (as written above) |
| 0.689919 | 0.68946 | drop `--idf-bonus --top-eps --top-k-eps`, use `--tie-key cont3posft` |
| 0.684938 | 0.68527 | additionally drop `--ft-bonus`, use `--tie-key cont3pos` |
| 0.667745 | 0.66738 | `--temperature 2.6 --lam 0.60 --eps 0.010 --eps-hi 0.040 --conf-t 0.15 --tie-key cont3` |

These four are the standing regression checks for the codebase: any change that breaks
one of them is rejected regardless of how it scores.

---

## 2. Running the models (inference)

### Score a (query, product) pair with the re-ranker

The cross-encoder is a `CrossEncoder` whose inputs **must** go through this project's
chat template, or scores are meaningless:

```python
import sys; sys.path.insert(0, "src")
from sentence_transformers.cross_encoder import CrossEncoder
from reranker_common import format_query, format_document
from common import MODELS
import torch, math

model = CrossEncoder(str(MODELS / "qwen3-reranker-arabic-ecom-rerank_v7synth"),
                     device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
model[0].max_seq_length = 160

query, product = "نبي حليب اطفال", "حليب نيدو للاطفال - 400 غ"
score = model.predict([[format_query(query), format_document(product)]])[0]

# apply the calibration used in production
T = 3.2
logit = math.log(score / (1 - score))
calibrated = 1 / (1 + math.exp(-logit / T))
print(score, calibrated)
```

Queries and product names go in **raw** (unnormalized) - the model was trained that
way. Normalization is only used for BM25 and the containment key.

### Embed with the retriever

The embedding model is **asymmetric**: queries need the stored instruction prompt,
documents must not have it. Getting this wrong scores 0.0000.

```python
from sentence_transformers import SentenceTransformer
from common import MODELS

m = SentenceTransformer(str(MODELS / "qwen3-arabic-ecom-lora_v1"))
q = m.encode(["حليب اطفال"], prompt_name="query")   # queries: prompt_name REQUIRED
d = m.encode(["حليب نيدو للاطفال - 400 غ"])          # documents: no prompt
```

### End-to-end on your own queries

Put them in a jsonl with `query_id` and `query` fields, then pass `--queries` to the
retrieval scripts and `rerank.py` (all three accept it), following section 3.

---

## 3. Rebuild the submission from weights

Run the two retrievers over the full catalog at depth 1000, then re-score and blend.

```bash
./.venv/Scripts/python.exe src/run_bm25.py --top-k 1000 --name bm25_all
```

```bash
./.venv/Scripts/python.exe src/run_dense.py --model models/qwen3-arabic-ecom-lora_v1 --dtype float32 --top-k 1000 --name dense_lora_v1_all
```

BM25 is CPU-only (~20 min). The dense pass builds a corpus embedding cache on first
use (~2 min on GPU, cached thereafter).

Now score all ~4.83 M candidate pairs. Use the **checkpointed** scorer, which saves
after every chunk and resumes on re-run - a plain `rerank.py` pass saves only at the
end and loses everything to a power cut:

```bash
./.venv/Scripts/python.exe src/rescore_benchmark_chunked.py --model models/qwen3-reranker-arabic-ecom-rerank_v7synth --runs runs/bm25_all.jsonl runs/dense_lora_v1_all.jsonl --depths 100 100 --weights 3 1 --chunk 1500
```

4-8 h on an 8 GB GPU. If interrupted, re-run the identical command; completed chunks
are skipped instantly. Then generate the submission with the section 1 command
(the cache is now populated) and verify with `eval_ndcg.py`.

---

## 4. Full training pipeline from scratch

Roughly 20 GPU-hours total on an 8 GB card. **Reranker training is not resumable**
(`--resume` restores the LoRA adapter but transformers v5 refuses to `torch.load` the
optimizer state below torch 2.6, and this project is pinned to 2.5.1), so treat
each training command as non-interruptible.

### 4a. Dense retriever

```bash
./.venv/Scripts/python.exe src/train_lora.py --run-name lora_v1
```

~4.5 h. Ends by merging the adapter into `models/qwen3-arabic-ecom-lora_v1` with
from-disk numerical verification.

### 4b. Re-ranker stage 1 - Arabic IR warm-up

```bash
./.venv/Scripts/python.exe src/build_mmarco_warmup.py --n-triples 20000
```

```bash
./.venv/Scripts/python.exe src/train_reranker.py --loss bce --pairs-parquet data/teacher/mmarco_ar_warmup_pairs.parquet --epochs 2 --run-name mmarco_warm
```

~1.5 h. Produces `models/qwen3-reranker-arabic-ecom-mmarco_warm`.

### 4c. Re-ranker stage 2 - synthetic corpus-grounded curriculum

```bash
./.venv/Scripts/python.exe src/build_synthetic_curriculum.py --max-groups 60000
```

```bash
./.venv/Scripts/python.exe src/validate_synthetic.py
```

The validation step is not optional in spirit: it checks the generated queries match
benchmark statistics and measures label contamination before you spend GPU hours.
Then train (~2.8 h):

```bash
./.venv/Scripts/python.exe src/train_reranker.py --loss lambda --groups-parquet data/teacher/synthetic_curriculum_groups.parquet --base models/qwen3-reranker-arabic-ecom-mmarco_warm --run-name synth_stage15 --mini-batch 2 --accum 32
```

### 4d. Re-ranker stage 3 - provided-pairs listwise fine-tune

```bash
./.venv/Scripts/python.exe src/train_reranker.py --loss lambda --base models/qwen3-reranker-arabic-ecom-synth_stage15 --run-name rerank_v7synth --mini-batch 2 --accum 32
```

~3.4 h, producing the final re-ranker. Then follow section 3.

**Do not raise `--mini-batch` above 2.** Beyond ~20 pair-sequences per step this
spills into system RAM and step time degrades from ~9.5 s to ~30 s with no error
message.

---

## 5. Reproducing the tuning campaigns (optional)

The post-processing constants were fitted with these, each on its own single-use
pristine split; all are cache-only and CPU-only:

| Script | Fits |
|---|---|
| `grid_search_v7synth{,_ext,_ext2}.py` | temperature, blend weight, band widths, confidence threshold |
| `grid_search_prior{,_ext}.py` | RRF decay `k` and lexical:dense ratio |
| `validate_v7synth_recal.py`, `validate_prior_recal.py` | val + full-benchmark + untouched-remainder confirmation |
| `loss_autopsy.py` | per-query error decomposition against the pool oracle |
| `grid_search_ensemble.py`, `mini_probes_v7.py` | the closing negative results |

Splits are seeded and committed under `splits/`, so re-running reproduces the numbers in
the technical report exactly.

---

## Final configuration reference

```
retrieval   BM25 top-100 (normalized Arabic, ascending-id tie-break)
          + Qwen3-Embedding-0.6B + LoRA, top-100 (instruction prompt on queries only)
fusion      weighted RRF, k=60, lexical:dense = 3:1
reranker    Qwen3-Reranker-0.6B-seq-cls, 3-stage LoRA fine-tune, seq len 160
calibration temperature T=2.6 (delogit / T / re-sigmoid)
blend       0.60 * CE + 0.40 * normalized RRF prior
bands       eps=0.010, widening to 0.040 when CE top1-top5 margin < 0.15
in-band key char-3gram query-in-name containment (11 dialect fillers stripped),
            then ascending product id
```
