"""Cross-encoder reranker fine-tune: Qwen3-Reranker-0.6B-seq-cls + LoRA (phase 2 step A).

Design per PHASE2_PLAN.md section 3.A and the model card:
  - pairs are (formatted query, formatted document) with BinaryCrossEntropyLoss;
    label 1 = provided positive, label 0 = provided hard negative plus one random
    catalog negative per query (training only on hard negatives is known to break
    easy-case behavior, so randoms stay in the mix).
  - false-negative hygiene extends the retriever's within-row filter: a per-query
    set of normalized positive names, and any negative (hard or random) whose
    normalized name is in that set is dropped.
  - LoRA r=16 alpha=32 on all attention + MLP projections via transformers-native
    injection, exactly like train_lora.py. The score head stays frozen: native
    injection does not support modules_to_save, and the head is a fixed 1x1024
    readout that the adapted layers below can rotate hidden states into.
  - bf16 training, fp32 state-dict merge with from-disk verification (trap 3.1/3.2);
    after the merge the score head is asserted byte-identical to the base.
  - Windows: dataloader_num_workers=0; no NO_DUPLICATES sampler needed here because
    BCE uses explicit labels, not in-batch negatives.

Smoke mode (--smoke) trains on 2,000 pairs (~30 steps) to validate the full
train->adapter save->merge->reload loop before the full run.
"""

import argparse
import time

import pandas as pd
import torch
from datasets import Dataset

from common import BENCH, MODELS, ROOT, TRAIN, load_jsonl, normalize_arabic, setup_utf8
from reranker_common import BASE_CE, format_document, format_query


def build_pairs(smoke: bool, max_pairs: int, seed: int = 42) -> Dataset:
    df = pd.read_parquet(TRAIN / "pairs_with_negatives" / "train.parquet")
    df["pos_norm"] = df["positive_product_name"].map(normalize_arabic)
    df["neg_norm"] = df["negative_product_name"].map(normalize_arabic)
    pos_names = df.groupby("user_query")["pos_norm"].agg(set)

    positives = df[["user_query", "positive_product_name"]].drop_duplicates()
    pos_pairs = pd.DataFrame(
        {"q": positives["user_query"], "d": positives["positive_product_name"], "label": 1.0}
    )

    hard = df[[not (n in pos_names[q]) for q, n in zip(df["user_query"], df["neg_norm"])]]
    hard = hard[["user_query", "negative_product_name"]].drop_duplicates()
    hard_pairs = pd.DataFrame(
        {"q": hard["user_query"], "d": hard["negative_product_name"], "label": 0.0}
    )

    corpus_names = [row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")]
    rng = pd.Series(corpus_names).sample(frac=1.0, random_state=seed).tolist()
    rand_rows = []
    i = 0
    for q, names in pos_names.items():
        while normalize_arabic(rng[i % len(rng)]) in names:
            i += 1
        rand_rows.append((q, rng[i % len(rng)], 0.0))
        i += 1
    rand_pairs = pd.DataFrame(rand_rows, columns=["q", "d", "label"])

    data = pd.concat([pos_pairs, hard_pairs, rand_pairs], ignore_index=True)
    print(
        f"pairs: {len(pos_pairs)} pos + {len(hard_pairs)} hard neg "
        f"({len(df) - len(hard)} dropped as cross-row false negatives or dupes) "
        f"+ {len(rand_pairs)} random neg = {len(data)}"
    )
    n = 2000 if smoke else max_pairs
    if n and n < len(data):
        data = data.sample(n=n, random_state=seed)
        print(f"subsampled to {len(data)} (label mean {data['label'].mean():.3f})")
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return Dataset.from_pandas(
        pd.DataFrame(
            {
                "query": data["q"].map(format_query),
                "document": data["d"].map(format_document),
                "label": data["label"].astype("float32"),
            }
        )
    )


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-name", default=None, help="default: rerank_v1 or rerank_smoke")
    p.add_argument("--max-pairs", type=int, default=200_000, help="0 = all pairs")
    p.add_argument("--mini-batch", type=int, default=16, help="per-device batch size")
    p.add_argument("--accum", type=int, default=4, help="grad accumulation (effective 64)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq", type=int, default=160)
    args = p.parse_args()
    run_name = args.run_name or ("rerank_smoke" if args.smoke else "rerank_v1")
    out_dir = ROOT / "outputs" / run_name
    adapter_dir = MODELS / f"{run_name}_adapter"
    merged_dir = MODELS / f"qwen3-reranker-arabic-ecom-{run_name}"

    from peft import LoraConfig
    from safetensors.torch import load_file
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    from merge_adapter import merge

    dataset = build_pairs(args.smoke, args.max_pairs)

    model = CrossEncoder(
        str(BASE_CE), device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}
    )
    model[0].max_seq_length = args.max_seq

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.add_adapter(peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    assert not model[0].auto_model.score.weight.requires_grad, "score head must stay frozen"

    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=args.mini_batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers v5: float in (0,1) means warmup ratio
        bf16=True,
        dataloader_num_workers=0,
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=[],
        seed=42,
    )

    trainer = CrossEncoderTrainer(model=model, args=train_args, train_dataset=dataset, loss=loss)
    t0 = time.time()
    trainer.train()
    print(f"training done in {(time.time() - t0) / 60:.1f} min")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(adapter_dir))
    ad_file = adapter_dir / "adapter_model.safetensors"
    assert ad_file.exists(), "adapter save produced no adapter_model.safetensors"
    ad_keys = list(load_file(str(ad_file)))
    assert any(k.endswith(".lora_A.weight") for k in ad_keys), "no lora tensors in adapter file"
    print(f"adapter saved to {adapter_dir} ({len(ad_keys)} tensors)")

    merge(BASE_CE, adapter_dir, merged_dir)

    base_score = load_file(str(BASE_CE / "model.safetensors"))["score.weight"]
    merged_score = load_file(str(merged_dir / "model.safetensors"))["score.weight"]
    assert torch.equal(base_score, merged_score), "frozen score head changed during merge"
    print("score head verified identical to base")


if __name__ == "__main__":
    main()
