"""Gate for v8graded (the graded-curriculum re-ranker), stage 1 of the ladder.

Scores the dev_rand2000 pools with the new model and reports the decisive number:
pure-CE ordering (lam=1.0, temperature-invariant) against its predecessors.

  BCE 0.524 < zero-shot 0.561 < listwise 0.593 < +warm-up 0.603 < v7synth 0.621

v7synth is the bar. A clear jump justifies the ~2 h val pass and the stack
recalibration; anything at or below 0.621 means the graded labels did not improve
ordering and the experiment closes as a documented negative.

Note the blend numbers printed here are NOT the verdict: they use v7synth's tuned
stack constants, and the 2026-07-26 finding is that a new model must have the stack
RE-TUNED before its end-to-end value can be judged. Pure-CE is the clean signal.
"""

import subprocess
import sys

from common import ROOT, setup_utf8

setup_utf8()

MODEL = "models/qwen3-reranker-arabic-ecom-rerank_v8graded"


def main():
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"), "src/rerank.py",
        "--runs", "runs/bm25_all.jsonl", "runs/dense_lora_v1_all.jsonl",
        "--depths", "100", "100", "--weights", "3", "1", "--rrf-k", "60",
        "--model", MODEL,
        "--qids", "splits/dev_rand2000.txt",
        "--lam-grid", "0.0,0.40,0.50,0.60,0.70,1.0",
    ]
    print("scoring dev_rand2000 with v8graded (~42 min GPU) ...\n", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print("FAILED:\n" + r.stderr[-2000:])
        sys.exit(1)

    print("""
INTERPRETATION
  lam=1.00 is pure-CE ordering - the clean, temperature-invariant signal.
    bars: v2 0.593 | v6warm 0.603 | v7synth 0.621
  lam=0.00 must reproduce ~0.6330 (prior-only sanity check).
  Intermediate lambdas use v7synth's stack constants and therefore UNDERSTATE a
  better model; the stack must be re-tuned before any end-to-end verdict.
""")


if __name__ == "__main__":
    main()
