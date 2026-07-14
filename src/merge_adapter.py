"""Merge a LoRA adapter into the base checkpoint at the state-dict level.

Why not PeftModel + module swapping: sentence-transformers 5.6 save_pretrained
silently ignores a reassigned `st[0].auto_model`, so the "merged" file comes out
byte-identical to the base (verified twice by sha256 before this existed).
Here the merge is pure tensor arithmetic (W += alpha/r * B @ A in float32), the
output directory reuses the base's ST config files, and the result is re-loaded
from disk and numerically verified before the script reports success.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from common import MODELS, setup_utf8


def merge(base: Path, adapter: Path, out: Path) -> None:
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    print(f"scale = alpha/r = {scale}")

    base_sd = load_file(str(base / "model.safetensors"))
    ad = load_file(str(adapter / "adapter_model.safetensors"))

    pairs = {}
    for k in ad:
        if k.endswith(".lora_A.weight"):
            tgt = k[len("base_model.model.") : -len(".lora_A.weight")] + ".weight"
            pairs[tgt] = (ad[k], ad[k.replace("lora_A", "lora_B")])
    assert pairs, "no lora_A tensors found in adapter"

    rel = []
    for tgt, (A, B) in pairs.items():
        assert tgt in base_sd, f"adapter target missing from base: {tgt}"
        W = base_sd[tgt].float()
        delta = scale * (B.float() @ A.float())
        assert delta.shape == W.shape, f"shape mismatch on {tgt}"
        rel.append((delta.abs().mean() / W.abs().mean()).item())
        base_sd[tgt] = (W + delta).contiguous()
    print(f"merged {len(pairs)} matrices; mean relative delta {sum(rel) / len(rel):.2e}")

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name == "model.safetensors" or f.name.startswith("."):
            continue
        dst = out / f.name
        if dst.exists():
            continue
        if f.is_dir():
            shutil.copytree(f, dst)
        else:
            shutil.copy2(f, dst)
    save_file(base_sd, str(out / "model.safetensors"))

    # Trust nothing: reload from disk and verify one merged matrix exactly.
    saved = load_file(str(out / "model.safetensors"))
    orig = load_file(str(base / "model.safetensors"))
    tgt, (A, B) = next(iter(sorted(pairs.items())))
    want = orig[tgt].float() + scale * (B.float() @ A.float())
    exact = torch.equal(saved[tgt].float(), want)
    frac_changed = (saved[tgt].float() != orig[tgt].float()).float().mean().item()
    print(f"reload check on {tgt}: exact={exact}, elements changed vs base={frac_changed:.2%}")
    assert exact, "saved weights do not equal base + scaled delta"
    assert frac_changed > 0.5, "suspiciously few elements changed; merge likely no-op"
    print(f"saved verified merged model to {out}")


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=MODELS / "qwen3-embedding-0.6b-arabic-ecom")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    merge(args.base, args.adapter, args.out)


if __name__ == "__main__":
    main()
