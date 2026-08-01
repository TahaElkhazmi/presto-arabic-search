"""Fine-tune bge-reranker-v2-m3 (XLM-RoBERTa encoder) with LambdaLoss.

Cross-architecture reranker probe for ensemble diversity with v6warm (Qwen3
decoder).  Uses our proven LambdaLoss group recipe with RAW [query, name]
pairs (no Qwen chat template - bge takes raw text, matching probe_bge.py).

Engineering differences from train_reranker.py (Qwen):
  - Input: raw text pairs, no format_query/format_document.
  - LoRA targets: XLM-RoBERTa attention projections (query, key, value).
  - Score head: classifier (dense + out_proj), asserted frozen + merge-verified.
  - Base model resolved via huggingface_hub snapshot_download (HF cache or fresh).

Gated probe protocol: smoke -> full train -> pure-CE gate -> ensemble gate.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import MODELS, ROOT, setup_utf8
from train_reranker import build_groups


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-name", default=None,
                   help="default: bge_smoke or bge_v1")
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                   help="HF model name or local path")
    p.add_argument("--mini-batch", type=int, default=2,
                   help="per-device batch size (groups per step before accum)")
    p.add_argument("--accum", type=int, default=32,
                   help="gradient accumulation steps (effective batch = mini-batch * accum)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-seq", type=int, default=160)
    p.add_argument("--n-random", type=int, default=2,
                   help="random catalog negatives per group")
    p.add_argument("--hard-cap", type=int, default=6,
                   help="provided hard negatives per group")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    run_name = args.run_name or ("bge_smoke" if args.smoke else "bge_v1")
    out_dir = ROOT / "outputs" / run_name
    adapter_dir = MODELS / f"{run_name}_adapter"
    merged_dir = MODELS / f"bge-reranker-v2-m3-arabic-ecom-{run_name}"

    # -- data: reuse the proven group builder with raw text (no Qwen template) --
    dataset = build_groups(
        args.smoke, n_random=args.n_random, hard_cap=args.hard_cap, raw=True
    )

    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import LambdaLoss

    from merge_adapter import merge

    # -- resolve base model to a local directory for merge_adapter --
    model_arg = args.model
    base_local = Path(model_arg)
    if not (base_local / "model.safetensors").exists():
        # Not a local dir with safetensors; resolve via HF hub
        base_local = Path(snapshot_download(model_arg))
        print(f"base model resolved from HF cache: {base_local}")
    else:
        print(f"base model from local path: {base_local}")
    assert (base_local / "model.safetensors").exists(), \
        f"model.safetensors not found in {base_local}"

    model = CrossEncoder(
        str(base_local), device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model[0].max_seq_length = args.max_seq

    # Print model architecture for verification
    auto_model = model[0].auto_model
    print(f"model type: {type(auto_model).__name__}")
    print(f"num labels: {auto_model.config.num_labels}")

    # -- LoRA on XLM-RoBERTa attention projections only --
    # "query", "key", "value" match self-attention linear layers;
    # the classifier head (dense + out_proj) is NOT matched, stays frozen.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
    )
    model.add_adapter(peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Verify classifier head is frozen
    classifier = auto_model.classifier
    for name, param in classifier.named_parameters():
        assert not param.requires_grad, f"classifier param {name} is trainable!"
    print("classifier head verified frozen")

    # -- training --
    loss = LambdaLoss(model, mini_batch_size=16)

    train_args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.mini_batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # transformers v5: float in (0,1) = warmup ratio
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

    trainer = CrossEncoderTrainer(
        model=model, args=train_args, train_dataset=dataset, loss=loss,
    )
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    print(f"training done in {elapsed:.1f} min")

    # -- save adapter --
    adapter_dir.mkdir(parents=True, exist_ok=True)
    auto_model.save_pretrained(str(adapter_dir))
    ad_file = adapter_dir / "adapter_model.safetensors"
    assert ad_file.exists(), "adapter save produced no adapter_model.safetensors"
    ad_keys = list(load_file(str(ad_file)))
    assert any(k.endswith(".lora_A.weight") for k in ad_keys), \
        "no lora tensors in adapter file"
    print(f"adapter saved to {adapter_dir} ({len(ad_keys)} tensors)")

    # Free GPU memory before the CPU-intensive fp32 merge (568M params x2 loads)
    del trainer, model, loss
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory released for merge")

    # -- fp32 state-dict merge (proven merge_adapter.py) --
    merge(base_local, adapter_dir, merged_dir)

    # -- verify classifier head unchanged after merge --
    base_sd = load_file(str(base_local / "model.safetensors"))
    merged_sd = load_file(str(merged_dir / "model.safetensors"))
    classifier_keys = [k for k in base_sd if k.startswith("classifier.")]
    assert classifier_keys, "no classifier keys found in base model"
    for key in classifier_keys:
        assert torch.equal(base_sd[key], merged_sd[key]), \
            f"classifier key {key} changed during merge!"
    print(f"classifier head verified identical to base ({len(classifier_keys)} tensors)")
    print(f"\nmerged model saved to {merged_dir}")
    print(f"score with: python src/probe_bge.py --model {merged_dir} --cache-name bge_ft_v1 ...")


if __name__ == "__main__":
    main()
