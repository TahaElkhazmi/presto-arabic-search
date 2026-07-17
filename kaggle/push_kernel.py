"""Push a private script kernel to Kaggle, run it, poll to completion, fetch output.

Usage:
  python kaggle/push_kernel.py --slug user/kernel-slug --title kernel-slug \
      --source kaggle/kernel_score_pairs.py --datasets user/some-dataset \
      --machine NvidiaTeslaT4 --timeout 7200 --out-dir data/teacher/probe_out

Polls the session status every --poll seconds; on COMPLETE downloads the kernel
output into --out-dir via kagglehub; on ERROR prints the tail of the session log.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import setup_utf8  # noqa: E402

from kagglesdk import KaggleClient  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiGetKernelSessionStatusRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType  # noqa: E402


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="username/kernel-slug")
    p.add_argument("--title", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--machine", default="NvidiaTeslaT4")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--poll-only", action="store_true", help="skip push, just poll + download")
    args = p.parse_args()
    user, kslug = args.slug.split("/")

    with KaggleClient() as client:
        api = client.kernels.kernels_api_client
        if not args.poll_only:
            req = ApiSaveKernelRequest()
            req.slug = args.slug
            req.new_title = args.title
            req.text = args.source.read_text(encoding="utf-8")
            req.language = "python"
            req.kernel_type = "script"
            req.is_private = True
            req.enable_gpu = True
            req.enable_internet = True
            req.machine_shape = args.machine
            req.session_timeout_seconds = args.timeout
            if args.datasets:
                req.dataset_data_sources = args.datasets
            req.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
            resp = api.save_kernel(req)
            print(f"pushed: {resp.to_json()}", flush=True)

        t0 = time.time()
        last = None
        while True:
            sreq = ApiGetKernelSessionStatusRequest()
            sreq.user_name = user
            sreq.kernel_slug = kslug
            try:
                status = api.get_kernel_session_status(sreq)
                s = status.to_json()
            except Exception as e:  # transient API hiccups must not kill the poll
                print(f"[{(time.time() - t0) / 60:5.1f} min] status error: {e}", flush=True)
                time.sleep(args.poll)
                continue
            if s != last:
                print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)
                last = s
            up = s.upper()
            if "COMPLETE" in up:
                break
            if "ERROR" in up or "CANCEL" in up:
                print("session failed", flush=True)
                sys.exit(2)
            time.sleep(args.poll)

    if args.out_dir:
        import kagglehub

        path = kagglehub.notebook_output_download(args.slug)
        print(f"output downloaded to: {path}", flush=True)
        import shutil

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for f in Path(path).rglob("*"):
            if f.is_file():
                rel = f.relative_to(path)
                dest = args.out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        print(f"copied into {args.out_dir}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
