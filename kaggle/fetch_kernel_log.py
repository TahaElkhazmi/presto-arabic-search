"""Fetch the tail of a Kaggle kernel session log, for diagnosing a failed run.

push_kernel.py reports ERROR without the reason, which is not enough to fix anything.
Session log streaming is known to cut off mid-session with ChunkedEncodingError, so
read defensively and fall back to the buffered output listing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import setup_utf8  # noqa: E402

from kagglesdk import KaggleClient  # noqa: E402

setup_utf8()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="username/kernel-slug")
    p.add_argument("--tail", type=int, default=60)
    args = p.parse_args()
    user, kslug = args.slug.split("/")

    with KaggleClient() as client:
        api = client.kernels.kernels_api_client
        lines = []
        try:
            from kagglesdk.kernels.types.kernels_api_service import (
                ApiGetKernelSessionLogsRequest)
            req = ApiGetKernelSessionLogsRequest()
            req.user_name = user
            req.kernel_slug = kslug
            resp = api.get_kernel_session_logs(req)
            lines.append(str(resp))
        except Exception as e:
            print(f"log RPC failed ({type(e).__name__}: {e}); trying output listing",
                  flush=True)
            try:
                from kagglesdk.kernels.types.kernels_api_service import (
                    ApiListKernelSessionOutputRequest)
                req2 = ApiListKernelSessionOutputRequest()
                req2.user_name = user
                req2.kernel_slug = kslug
                resp2 = api.list_kernel_session_output(req2)
                lines.append(str(resp2))
            except Exception as e2:
                print(f"output listing failed too: {type(e2).__name__}: {e2}")

    text = "\n".join(lines)
    out = text.splitlines()
    for ln in out[-args.tail:]:
        print(ln)
    print(f"\n({len(out)} log lines total)")


if __name__ == "__main__":
    main()
