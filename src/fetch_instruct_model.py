"""Download Qwen3-4B-Instruct-2507 for the fair PLAY B listwise test.

The mark-2 probe judged with `qwen3-reranker-4b-seq-cls`, a checkpoint fine-tuned to
emit yes/no. Constrained decoding removed its format failures, but it remains a
task-specialised reranker, not an instruction follower, so its null result closes
listwise WITH THE MODEL WE OWN rather than listwise as a mechanism.

Qwen3-4B-Instruct-2507 is the non-thinking instruct release (no <think> preamble to
burn tokens on), same 4B class so it fits the same 4-bit budget, and it is the fair
test of whether seeing candidates together helps.
"""

import os

# The default hub client has no read timeout, so a silently dropped connection
# hangs forever instead of raising. Observed twice on this link: the fetch froze
# mid-shard (95 MB, then 2.54 GB) and never recovered. A short timeout converts the
# hang into an exception the retry loop below can act on, and every restart resumes
# from the partial file rather than starting over.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from huggingface_hub import snapshot_download

from common import MODELS, setup_utf8

setup_utf8()

REPO = "Qwen/Qwen3-4B-Instruct-2507"
DEST = MODELS / "qwen3-4b-instruct-2507"


def main():
    print(f"downloading {REPO} -> {DEST}")
    # max_workers=1: parallel shard fetches stalled repeatedly on this connection
    # (shard 3 froze at 95 MB while 1 and 2 never started). Serial is slower in
    # theory and far more reliable here. Resumes from whatever is already cached.
    for attempt in range(1, 41):
        try:
            path = snapshot_download(
                repo_id=REPO,
                local_dir=str(DEST),
                max_workers=1,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*",
                                "merges.txt", "vocab.json"],
            )
            print(f"done: {path}")
            return
        except Exception as e:  # transient network failures are expected here
            print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("download failed after 40 attempts")


if __name__ == "__main__":
    main()
