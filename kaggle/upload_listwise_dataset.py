"""Upload the band probe input as a private Kaggle dataset for the judge kernel.

Only query text and candidate product names go up. No labels, no qrels, nothing
derived from candidates.jsonl, so the kernel cannot fit to the benchmark even in
principle and rule 2.4.b holds by construction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, setup_utf8  # noqa: E402

setup_utf8()

HANDLE = "tahaelkhazmi/presto-listwise-band"
SRC = ROOT / "data" / "kaggle_listwise"


def main():
    import kagglehub

    files = list(SRC.glob("*.parquet"))
    if not files:
        raise SystemExit(f"nothing to upload in {SRC}")
    total = sum(f.stat().st_size for f in files)
    print(f"uploading {len(files)} file(s), {total / 1e6:.1f} MB -> {HANDLE}")
    for f in files:
        print(f"  {f.name}  {f.stat().st_size / 1e6:.2f} MB")

    kagglehub.dataset_upload(HANDLE, str(SRC),
                             version_notes="band conf<0.02, K=20, 600 queries")
    print("uploaded")


if __name__ == "__main__":
    main()
