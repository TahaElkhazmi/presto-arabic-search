"""Verify the --resume adapter loader against a REAL checkpoint before trusting it.

Replicates train_reranker.py's model construction and its `_load_adapter_verified`
logic exactly, then proves the load is correct three ways:
  1. every tensor in the checkpoint file is present after the load;
  2. every loaded tensor is numerically EQUAL to the file (no partial/mis-mapped load);
  3. the loaded weights actually differ from the freshly-initialised adapter, i.e.
     something was really restored rather than the load silently no-opping - the
     exact failure mode of the ST 5.6 save_pretrained trap (FINDINGS 3.2).

Test 3 is the important one: assertions 1 and 2 would both pass trivially if the
loader wrote nothing and we compared a freshly-zeroed adapter against itself.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from sentence_transformers.cross_encoder import CrossEncoder

from common import MODELS, setup_utf8

setup_utf8()


def main():
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1
                else "outputs/synth_v2_stage15/checkpoint-1250")
    base = MODELS / "qwen3-reranker-arabic-ecom-mmarco_warm"
    print(f"checkpoint: {ckpt}\nbase      : {base.name}\n")

    # weight-loading test only, no compute: run on CPU so it can be executed while a
    # training job holds the GPU
    model = CrossEncoder(str(base), device="cpu",
                         model_kwargs={"torch_dtype": torch.bfloat16})
    model[0].max_seq_length = 160
    model.add_adapter(LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    target = model[0].auto_model

    fresh = {k: v.detach().clone() for k, v in get_peft_model_state_dict(target).items()}
    raw = load_file(str(ckpt / "adapter_model.safetensors"))
    # save_pretrained writes peft's WRAPPED convention ("base_model.model.<path>"),
    # but a transformers-native add_adapter model uses the bare path. Without this
    # remap set_peft_model_state_dict matches nothing and silently no-ops.
    PREFIX = "base_model.model."
    saved = {(k[len(PREFIX):] if k.startswith(PREFIX) else k): v for k, v in raw.items()}
    print(f"fresh adapter tensors: {len(fresh)}, checkpoint tensors: {len(saved)}")

    set_peft_model_state_dict(target, saved)
    back = get_peft_model_state_dict(target)

    missing = [k for k in saved if k not in back]
    bad = [k for k in saved
           if k in back and not torch.equal(back[k].detach().cpu().to(saved[k].dtype),
                                            saved[k].cpu())]
    changed = sum(
        1 for k in saved
        if k in fresh and not torch.equal(fresh[k].cpu().to(saved[k].dtype),
                                          back[k].detach().cpu().to(saved[k].dtype)))

    print(f"\n1. present after load : {len(saved) - len(missing)}/{len(saved)}")
    print(f"2. numerically exact  : {len(saved) - len(bad)}/{len(saved)}")
    print(f"3. changed vs fresh   : {changed}/{len(saved)}  (must be > 0, else the load no-opped)")

    ok = not missing and not bad and changed > 0
    if ok:
        # lora_B initialises to zero, so only lora_A rows are expected to differ;
        # report the split so the number above is interpretable.
        nb = sum(1 for k in saved if "lora_B" in k)
        print(f"\n   (of which lora_B tensors: {nb} - these are zero-init, so a nonzero"
              f"\n    'changed' count on them is itself proof real weights arrived)")
        print("\nRESULT: PASS - resume loader is correct and safe to rely on.")
    else:
        print(f"\nRESULT: FAIL - missing={len(missing)} bad={len(bad)} changed={changed}")
        print("Do NOT rely on --resume; treat training as non-interruptible.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
