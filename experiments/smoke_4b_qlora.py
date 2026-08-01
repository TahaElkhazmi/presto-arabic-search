"""Feasibility gate for PLAY A: can the banked 4B re-ranker train and infer in 8 GB?

Answers three questions in one run, in the order that kills the play fastest:
  1. does the 4B load in 4-bit NF4 within our VRAM budget?
  2. can we run a real training step (forward + backward) with LoRA + gradient
     checkpointing, and at what seconds/step?
  3. what is inference throughput in pairs/s - the constraint that decides whether a
     full-benchmark pass is possible (it is not: 4.83M pairs) and how deep a
     confidence-gated cascade can afford to go.

Uses the project's real template via reranker_common, so the numbers transfer.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from common import MODELS, setup_utf8
from reranker_common import format_document, format_query

setup_utf8()

MODEL = MODELS / "qwen3-reranker-4b-seq-cls"
SEQ = 160


def vram():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 1e9, tot / 1e9


def main():
    print(f"model: {MODEL.name}\n")
    t0 = time.time()

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL), quantization_config=qc, num_labels=1, dtype=torch.bfloat16)
    used, tot = vram()
    print(f"1. LOADED in 4-bit: {used:.2f}/{tot:.2f} GB used  [{time.time() - t0:.0f}s]")

    # ---- inference throughput (question 3, measured before training bloats VRAM) ----
    model.eval()
    pairs = [[format_query("حليب اطفال"), format_document("حليب نيدو للاطفال - 400 غ")]] * 32
    texts = [q + d for q, d in pairs]
    enc = tok(texts, padding=True, truncation=True, max_length=SEQ, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**enc)  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            model(**enc)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 3
    rate = len(texts) / dt
    print(f"3. INFERENCE: {rate:.0f} pairs/s at batch 32, seq {SEQ}")
    print(f"   -> full benchmark (4.83M pairs) = {4_827_501 / rate / 3600:.0f} h  [infeasible]")
    print(f"   -> cascade, wide-band 53% of queries at depth 50 (~0.77M pairs) = "
          f"{770_000 / rate / 3600:.1f} h")

    # ---- training step (question 2) ----
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   trainable params: {trainable:,}")

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    micro = 2
    enc2 = tok(texts[:micro], padding=True, truncation=True,
               max_length=SEQ, return_tensors="pt").to("cuda")
    labels = torch.tensor([[1.0], [0.0]], device="cuda", dtype=torch.bfloat16)

    times = []
    for i in range(4):
        t = time.time()
        out = model(**enc2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.logits.float(), labels.float())
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    used, tot = vram()
    step = sum(times[1:]) / 3
    print(f"2. TRAINING STEP OK: {step:.2f}s per micro-batch of {micro}, "
          f"VRAM {used:.2f}/{tot:.2f} GB")
    # v7synth stage 3 = 1284 optimiser steps at accum 32 -> 41k micro-batches of 2
    print(f"   -> a v7synth-sized stage (~41k micro-batches) = {41_088 * step / 3600:.1f} h")
    print("\nVERDICT: 4B QLoRA training is FEASIBLE on this GPU.")


if __name__ == "__main__":
    main()
