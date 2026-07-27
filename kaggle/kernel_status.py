"""One-shot Kaggle kernel session status, independent of any running poller.

push_kernel.py only prints on status CHANGE, so a long RUNNING stretch looks identical
to a dead poller. This asks the API directly and also reports whether output files are
available yet.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import setup_utf8  # noqa: E402

from kagglesdk import KaggleClient  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import (  # noqa: E402
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)

setup_utf8()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default="tahaelkhazmi/presto-listwise-judge")
    args = p.parse_args()
    user, kslug = args.slug.split("/")

    with KaggleClient() as client:
        api = client.kernels.kernels_api_client
        req = ApiGetKernelSessionStatusRequest()
        req.user_name = user
        req.kernel_slug = kslug
        print(f"status: {api.get_kernel_session_status(req).to_json()}")

        try:
            oreq = ApiListKernelSessionOutputRequest()
            oreq.user_name = user
            oreq.kernel_slug = kslug
            out = api.list_kernel_session_output(oreq)
            files = getattr(out, "files", None) or []
            print(f"output files: {len(files)}")
            for f in files:
                print(f"  {getattr(f, 'file_name', f)}")
            log = getattr(out, "log", "") or ""
            if log:
                import json as _json
                try:
                    entries = _json.loads(log)
                    tail = [e.get("data", "").rstrip() for e in entries[-25:]]
                    print("\n--- log tail ---")
                    for t in tail:
                        if t:
                            print(f"  {t}")
                except Exception:
                    print(f"\n--- raw log tail ---\n{log[-2000:]}")
        except Exception as e:
            print(f"output listing unavailable: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
