"""Stall-proof downloader for the instruct judge.

This link drops large-file transfers silently: the fetch has frozen three times
(95 MB, 2.54 GB, 4.85 GB) without raising, and HF_HUB_DOWNLOAD_TIMEOUT does not
cover a connection that stays open but stops delivering bytes. Every restart does
resume from the partial file, so the fix is a supervisor: run the download in a
child process, watch the bytes on disk, and if they stop growing, kill the child and
start it again. Repeats until the snapshot completes.

Run it with `run_in_background`; it prints one line per restart and one per progress
tick, so the log stays readable.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS, ROOT, setup_utf8

setup_utf8()

# The shared cache, NOT a local_dir: only here do partials get a stable
# blobs/<etag>.incomplete name that survives a kill and resumes. Restarting a
# local_dir download starts each shard over from zero (see fetch_instruct_model.py).
DEST = (Path.home() / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen3-4B-Instruct-2507")
EXPECT_GB = 7.5          # snapshot is ~8 GB; treat >=7.5 with no .incomplete as done
STALL_SECS = 120         # no growth for this long means the transfer is dead
POLL = 15


def size_gb() -> float:
    if not DEST.exists():
        return 0.0
    return sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e9


def complete() -> bool:
    if not DEST.exists():
        return False
    incomplete = list(DEST.rglob("*.incomplete"))
    shards = list((DEST / "snapshots").rglob("model-*.safetensors"))
    return not incomplete and len(shards) >= 3 and size_gb() >= EXPECT_GB


def main():
    attempt = 0
    while not complete():
        attempt += 1
        print(f"[supervisor] start attempt {attempt}, {size_gb():.2f} GB on disk",
              flush=True)
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "src" / "fetch_instruct_model.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(ROOT),
        )
        last, last_change = size_gb(), time.time()
        while proc.poll() is None:
            time.sleep(POLL)
            cur = size_gb()
            if cur > last + 0.005:
                print(f"[supervisor] {cur:.2f} GB", flush=True)
                last, last_change = cur, time.time()
            elif time.time() - last_change > STALL_SECS:
                print(f"[supervisor] STALLED at {cur:.2f} GB, restarting", flush=True)
                proc.kill()
                break
            if complete():
                break
        try:
            proc.kill()
        except Exception:
            pass
        if complete():
            break
        time.sleep(3)

    print(f"[supervisor] COMPLETE: {size_gb():.2f} GB in {DEST}", flush=True)


if __name__ == "__main__":
    main()
