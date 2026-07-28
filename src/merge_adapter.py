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


def shard_map(d: Path) -> dict[str, list[str]]:
    """filename -> tensor names, for single-file OR sharded checkpoints.

    Everything before the 4B was a single-file 0.6B, so this path had never been
    exercised; the 4B ships as model-0000{1,2}-of-00002.safetensors plus an index,
    and the old single-file assumption raised FileNotFoundError after a full night
    of training had already completed. Sharding is preserved on output rather than
    collapsed into one file, which also keeps peak RAM to one shard instead of the
    whole 4B state dict on a 15.7 GB box.
    """
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        out: dict[str, list[str]] = {}
        for tensor, fname in wm.items():
            out.setdefault(fname, []).append(tensor)
        return out
    return {"model.safetensors": []}      # [] = "load the file and take everything"


def load_tensor(d: Path, name: str):
    """Fetch a single tensor by name from a single-file or sharded checkpoint."""
    idx = d / "model.safetensors.index.json"
    fname = "model.safetensors"
    if idx.exists():
        fname = json.loads(idx.read_text(encoding="utf-8"))["weight_map"][name]
    return load_file(str(d / fname))[name]


def merge(base: Path, adapter: Path, out: Path) -> None:
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    print(f"scale = alpha/r = {scale}")

    shards = shard_map(base)
    ad = load_file(str(adapter / "adapter_model.safetensors"))

    pairs = {}
    for k in ad:
        if k.endswith(".lora_A.weight"):
            tgt = k[len("base_model.model.") : -len(".lora_A.weight")] + ".weight"
            pairs[tgt] = (ad[k], ad[k.replace("lora_A", "lora_B")])
    assert pairs, "no lora_A tensors found in adapter"

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name in shards or f.name.startswith("."):
            continue
        dst = out / f.name
        if dst.exists():
            continue
        if f.is_dir():
            shutil.copytree(f, dst)
        else:
            shutil.copy2(f, dst)

    # One shard at a time: load, apply whichever deltas land in it, write it back.
    rel, applied = [], 0
    for fname in shards:
        sd = load_file(str(base / fname))
        touched = False
        for tgt, (A, B) in pairs.items():
            if tgt not in sd:
                continue
            W = sd[tgt].float()
            delta = scale * (B.float() @ A.float())
            assert delta.shape == W.shape, f"shape mismatch on {tgt}"
            rel.append((delta.abs().mean() / W.abs().mean()).item())
            sd[tgt] = (W + delta).contiguous()   # stays float32, per the bf16 rounding trap
            applied += 1
            touched = True
        save_file(sd, str(out / fname), metadata={"format": "pt"})
        del sd
        print(f"  {fname}: {'merged' if touched else 'copied unchanged'}")
    assert applied == len(pairs), \
        f"{len(pairs) - applied} adapter targets never matched a base tensor"
    print(f"merged {applied} matrices; mean relative delta {sum(rel) / len(rel):.2e}")

    # Trust nothing: reload from disk and verify one merged matrix exactly.
    tgt, (A, B) = next(iter(sorted(pairs.items())))
    saved_t = load_tensor(out, tgt)
    orig_t = load_tensor(base, tgt)
    want = orig_t.float() + scale * (B.float() @ A.float())
    exact = torch.equal(saved_t.float(), want)
    frac_changed = (saved_t.float() != orig_t.float()).float().mean().item()
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
