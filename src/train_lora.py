"""LoRA contrastive fine-tune of the official checkpoint (build order step 4).

Spec per COMPETITION_GUIDE.md:
  - CachedMultipleNegativesRankingLoss (GradCache), effective batch 64, mini-batch 8
  - triplets from pairs_with_negatives train split, false negatives filtered
    (normalized negative name == normalized positive name)
  - the stored instruction prompt is applied to the query column during training
  - NO_DUPLICATES batch sampler (36% duplicate catalog names poison in-batch negatives)
  - bf16, LoRA r=16 alpha=32 on all attention + MLP projections, lr 1e-4 cosine, 1 epoch
  - Windows: dataloader_num_workers=0; load_best_model_at_end stays OFF (trap)

After training, the adapter is saved, then merged into a fresh base transformer and
saved as a standalone SentenceTransformer (model[0].auto_model is a plain Qwen3Model
after transformers-native adapter injection, so merge_and_unload() must go through
PeftModel.from_pretrained on a freshly loaded base).

Smoke mode (--smoke) trains on 2,000 triplets (~31 steps, ~5 min) to validate the
full train->merge->save loop before committing to the ~4.7h full epoch.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

from common import MODELS, ROOT, TRAIN, normalize_arabic, setup_utf8

BASE = MODELS / "qwen3-embedding-0.6b-arabic-ecom"


def load_triplets(smoke: bool) -> Dataset:
    df = pd.read_parquet(TRAIN / "pairs_with_negatives" / "train.parquet")
    n0 = len(df)
    keep = df["positive_product_name"].map(normalize_arabic) != df["negative_product_name"].map(
        normalize_arabic
    )
    df = df[keep].reset_index(drop=True)
    print(f"triplets: {n0} -> {len(df)} after false-negative filter ({n0 - len(df)} dropped)")
    if smoke:
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
        print(f"smoke mode: {len(df)} triplets")
    df = df.rename(
        columns={
            "user_query": "anchor",
            "positive_product_name": "positive",
            "negative_product_name": "negative",
        }
    )
    return Dataset.from_pandas(df[["anchor", "positive", "negative"]])


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-name", default=None, help="default: lora_v1 or lora_smoke")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--mini-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq", type=int, default=96)
    args = p.parse_args()
    run_name = args.run_name or ("lora_smoke" if args.smoke else "lora_v1")
    out_dir = ROOT / "outputs" / run_name
    adapter_dir = MODELS / f"{run_name}_adapter"
    merged_dir = MODELS / f"qwen3-arabic-ecom-{run_name}"

    from peft import LoraConfig, PeftModel, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import AutoModel

    dataset = load_triplets(args.smoke)

    model = SentenceTransformer(
        str(BASE), device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}
    )
    model.max_seq_length = args.max_seq
    query_prompt = model.prompts["query"]

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.add_adapter(peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)

    train_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers v5: float in (0,1) means warmup ratio
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=0,
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=[],
        seed=42,
        prompts={"anchor": query_prompt},
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=train_args, train_dataset=dataset, loss=loss
    )
    t0 = time.time()
    trainer.train()
    print(f"training done in {(time.time() - t0) / 60:.1f} min")

    # Save adapter only (auto_model has the transformers-native peft integration).
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(adapter_dir))
    print(f"adapter saved to {adapter_dir}")

    # Merge into a fresh base and save a standalone SentenceTransformer.
    # Merge in float32: early-training LoRA deltas (~1e-5 per element) sit below
    # bf16 representational precision and silently round away in a bf16 merge.
    base_transformer = AutoModel.from_pretrained(str(BASE), torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(base_transformer, str(adapter_dir)).merge_and_unload()
    st = SentenceTransformer(str(BASE), device="cpu", model_kwargs={"torch_dtype": torch.float32})
    st[0].auto_model = merged
    st.save_pretrained(str(merged_dir))
    print(f"merged standalone model saved to {merged_dir}")


if __name__ == "__main__":
    main()
