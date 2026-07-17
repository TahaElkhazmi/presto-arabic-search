"""Kaggle kernel: score (text_a, text_b) pairs with a seq-cls reranker, sharded over GPUs.

Runs as a Kaggle script kernel. Reads every *.parquet under /kaggle/input that has
text_a/text_b columns (pre-formatted with the official template at export time, so
this side is template-free). Writes per-block parquet checkpoints (qid, iid, logit)
to /kaggle/working/scores/ and merges them into teacher_scores.parquet at the end.

Parent process: downloads the model once, spawns one worker per visible GPU
(CUDA_VISIBLE_DEVICES pinning, block index modulo world size), waits, merges.
Workers resume for free: existing block files are skipped, so session restarts
lose at most one block of work.

Teacher scores are raw fp32 logits; sigmoid happens downstream when needed so
distillation keeps the uncompressed margin signal.
"""

import glob
import json
import os
import subprocess
import sys
import time

MODEL_ID = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
MAX_LEN = 160
BATCH = 64
BLOCK = 8192
SCORES_DIR = "/kaggle/working/scores"
MERGED_OUT = "/kaggle/working/teacher_scores.parquet"
META_OUT = "/kaggle/working/run_meta.json"


def find_pairs_files() -> list[str]:
    hits = []
    for path in sorted(glob.glob("/kaggle/input/**/*.parquet", recursive=True)):
        import pyarrow.parquet as pq

        cols = set(pq.ParquetFile(path).schema_arrow.names)
        if {"qid", "iid", "text_a", "text_b"} <= cols:
            hits.append(path)
    return hits


def load_pairs():
    import pandas as pd

    files = find_pairs_files()
    assert files, "no pairs parquet found under /kaggle/input"
    print(f"pairs files: {files}", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{len(df)} pairs total", flush=True)
    return df


def worker(rank: int, world: int):
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"[w{rank}] transformers {transformers.__version__}, torch {torch.__version__}, "
          f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()

    df = load_pairs()
    n = len(df)
    batch = BATCH

    def score_block(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        nonlocal batch
        order = np.argsort([len(a) + len(b) for a, b in zip(texts_a, texts_b)])
        logits = np.empty(len(texts_a), dtype=np.float32)
        s = 0
        while s < len(order):
            idx = order[s : s + batch]
            try:
                enc = tok(
                    [texts_a[i] for i in idx],
                    [texts_b[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    out = model(**enc).logits.float().squeeze(-1).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                assert batch > 1, "OOM at batch 1"
                batch //= 2
                print(f"[w{rank}] OOM, batch -> {batch}", flush=True)
                continue
            logits[idx] = out
            s += len(idx)
        return logits

    t0 = time.time()
    done = 0
    my_blocks = [b for b in range(0, n, BLOCK) if (b // BLOCK) % world == rank]
    for b0 in my_blocks:
        b1 = min(b0 + BLOCK, n)
        out_path = f"{SCORES_DIR}/scores_b{b0:09d}.parquet"
        if os.path.exists(out_path):
            print(f"[w{rank}] skip block {b0}", flush=True)
            continue
        blk = df.iloc[b0:b1]
        logits = score_block(list(blk["text_a"]), list(blk["text_b"]))
        pd.DataFrame(
            {"qid": blk["qid"].to_numpy(), "iid": blk["iid"].to_numpy(),
             "logit": logits}
        ).to_parquet(out_path + f".tmp{rank}.parquet", index=False)
        os.replace(out_path + f".tmp{rank}.parquet", out_path)
        done += b1 - b0
        dt = time.time() - t0
        print(f"[w{rank}] block {b0}:{b1} done | {done} pairs | {done / dt:.1f} pairs/s "
              f"| {dt / 60:.1f} min", flush=True)
    print(f"[w{rank}] worker finished, {done} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)


def parent():
    import torch

    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(MODEL_ID)
    except Exception as e:  # preinstalled transformers too old for qwen3
        print(f"config load failed ({type(e).__name__}: {e}); upgrading transformers", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"])
        os.execv(sys.executable, [sys.executable] + sys.argv)

    os.makedirs(SCORES_DIR, exist_ok=True)
    world = max(torch.cuda.device_count(), 1)
    print(f"parent: {world} gpu(s) visible", flush=True)

    from huggingface_hub import snapshot_download

    t0 = time.time()
    snapshot_download(MODEL_ID)
    print(f"model snapshot ready in {time.time() - t0:.0f}s", flush=True)

    procs = []
    for rank in range(world):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                   SHARD_RANK=str(rank), SHARD_WORLD=str(world))
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env))
    codes = [p.wait() for p in procs]
    assert all(c == 0 for c in codes), f"worker exit codes {codes}"

    import pandas as pd

    parts = sorted(glob.glob(f"{SCORES_DIR}/scores_b*.parquet"))
    merged = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    merged.to_parquet(MERGED_OUT, index=False)
    n_expected = len(load_pairs())
    meta = {
        "model": MODEL_ID, "max_len": MAX_LEN, "block": BLOCK,
        "n_scored": int(len(merged)), "n_expected": int(n_expected),
        "n_blocks": len(parts), "world": world,
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"merged {len(merged)} scores (expected {n_expected}) -> {MERGED_OUT}", flush=True)
    assert len(merged) == n_expected, "scored pair count mismatch"
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    if "SHARD_RANK" in os.environ:
        worker(int(os.environ["SHARD_RANK"]), int(os.environ["SHARD_WORLD"]))
    else:
        parent()
