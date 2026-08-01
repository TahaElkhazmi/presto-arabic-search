"""Dump band queries + candidates as compact JSON for an external LLM judge.

The Kaggle probe answered "can Qwen do this" (no, at both 4B and 14B). This exports
the SAME queries in a model-agnostic form so a different model family can be asked
the identical question, which is the open half of the H1 hypothesis: that the 0.71412
entry used a frontier-class re-ranker rather than a Qwen-class one.

Only query text and candidate product names go out. No labels, so the judge cannot
fit anything; scoring stays local.
"""

import argparse
import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common import ROOT, setup_utf8

setup_utf8()

SRC = ROOT / "data" / "kaggle_listwise" / "listwise_probe_input.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df = pd.read_parquet(SRC)
    qids = sorted(df.qid.unique(), key=int)[: args.n]
    letters = string.ascii_uppercase

    out = []
    for qid in qids:
        g = df[df.qid == qid].sort_values("rank")
        out.append({
            "qid": str(qid),
            "query": g["query"].iloc[0],
            "candidates": {letters[i]: str(n)
                           for i, n in enumerate(g["name"].tolist())},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} : {len(out)} queries, "
          f"{len(out[0]['candidates'])} candidates each")
    (ROOT / "splits" / "probe_judge150.txt").write_text(
        "\n".join(str(q) for q in qids) + "\n", encoding="utf-8")
    print("wrote splits/probe_judge150.txt")


if __name__ == "__main__":
    main()
