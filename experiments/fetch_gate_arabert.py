"""Download GATE-AraBert-v1, the embedder the benchmark's own baseline used.

Why this model specifically: research into how the benchmark was assembled found the
judged candidate pools were produced by a Meilisearch hybrid of BM25 plus
GATE-AraBert-v1. Two reasons that matters.

1. It is an Arabic-specialist encoder from a completely different lineage than our
   all-Qwen stack. Our five ensemble failures were ALL Qwen-family or bge, and the
   diagnosis of those failures was explicit: decorrelation needs genuinely different
   training, not one different stage of the same recipe.
2. It is the similarity notion that shaped which items got judged at all, so its
   view of "similar" is closer to the annotation process than ours.

Small model (AraBERT-base class, ~540 MB) so the download failures that killed the
8 GB Qwen fetch are far less likely; xet is disabled for resumable partials anyway.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from common import MODELS, setup_utf8

setup_utf8()

REPO = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
DEST = MODELS / "gate-arabert-v1"


def main():
    print(f"downloading {REPO}")
    for attempt in range(1, 21):
        try:
            path = snapshot_download(
                repo_id=REPO,
                local_dir=str(DEST),
                max_workers=1,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt",
                                "tokenizer*", "vocab*", "*.model", "1_Pooling/*",
                                "2_Normalize/*", "modules.json"],
            )
            size = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
            print(f"done: {path}  ({size / 1e6:.0f} MB)")
            return
        except Exception as e:
            print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("download failed")


if __name__ == "__main__":
    main()
