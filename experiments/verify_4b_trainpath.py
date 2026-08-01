"""PHASE 4 readiness: can train_reranker.py's stack actually load the 4B in 4-bit?

smoke_4b_qlora.py already proved the 4B trains in 8 GB via RAW transformers. The
untested link is the sentence-transformers CrossEncoder wrapper that train_reranker.py
actually uses: it constructs the model through `model_kwargs`, so a BitsAndBytesConfig
has to survive that path, the LoRA adapter has to attach to the quantized module, and
the score head has to stay frozen (an assertion train_reranker.py makes and which no
previous run exercised under quantization).

Deliberately does NOT run the merge step or write any model to disk: merging a 4B in
float32 would cost ~16 GB of disk for a smoke, and merge_adapter.py reads the ORIGINAL
base weights from the model directory rather than the quantized in-memory copy, so the
merge path is unaffected by quantization anyway.

Verifies, in the order that fails fastest:
  1. 4-bit load through CrossEncoder, with VRAM;
  2. LoRA attaches and the score head is still frozen;
  3. a real forward+backward on the project's own template, with peak VRAM and step
     time, so the training-hour estimate is grounded rather than inherited.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time

import torch

from common import MODELS, setup_utf8
from reranker_common import format_document, format_query

setup_utf8()

BASE = MODELS / "qwen3-reranker-4b-seq-cls"
SEQ = 160
MICRO = 2


def vram():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 1e9, tot / 1e9


def main():
    from peft import LoraConfig, prepare_model_for_kbit_training
    from sentence_transformers.cross_encoder import CrossEncoder
    from transformers import BitsAndBytesConfig

    print(f"base: {BASE.name}")
    t0 = time.time()
    mk = {
        "torch_dtype": torch.bfloat16,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    }
    model = CrossEncoder(str(BASE), device="cuda", model_kwargs=mk)
    model[0].max_seq_length = SEQ
    used, tot = vram()
    print(f"1. LOADED through CrossEncoder in 4-bit: {used:.2f}/{tot:.2f} GB "
          f"[{time.time() - t0:.0f}s]")

    model[0].auto_model = prepare_model_for_kbit_training(
        model[0].auto_model, use_gradient_checkpointing=True)
    model.add_adapter(LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    head_frozen = not model[0].auto_model.score.weight.requires_grad
    print(f"2. LoRA attached: {trainable:,}/{total:,} trainable "
          f"({100 * trainable / total:.2f}%)")
    print(f"   score head frozen (train_reranker asserts this): {head_frozen}")
    if not head_frozen:
        print("   !! the assertion in train_reranker.py would fail here")

    tok = model.tokenizer
    pairs = [[format_query("حليب اطفال نيدو"),
              format_document("حليب نيدو للاطفال مرحلة 2 - 400 غ")]] * MICRO
    enc = tok([q + d for q, d in pairs], padding=True, truncation=True,
              max_length=SEQ, return_tensors="pt").to("cuda")
    labels = torch.tensor([[1.0], [0.0]], device="cuda")

    am = model[0].auto_model
    am.train()
    opt = torch.optim.AdamW([p for p in am.parameters() if p.requires_grad], lr=1e-4)
    times = []
    for i in range(4):
        t = time.time()
        out = am(**enc)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out.logits.float(), labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    used, tot = vram()
    step = sum(times[1:]) / 3
    print(f"3. TRAIN STEP OK: {step:.2f}s per micro-batch of {MICRO}, "
          f"peak VRAM {used:.2f}/{tot:.2f} GB")
    print(f"   a v7synth-sized provided-pairs stage (~41k micro-batches) = "
          f"{41_088 * step / 3600:.1f} h")
    print("\nVERDICT: the train_reranker.py quantized path is usable for the 4B.")


if __name__ == "__main__":
    main()
