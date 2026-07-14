"""Create the fixed validation split (build order step 2).

Deterministic: sorted query ids shuffled with random.Random(42), first 20% held out
as validation. The result is materialized to splits/val_qids.txt and committed to
git so it can never drift. Tune on the remaining 80% (dev), report on validation.
"""

import random

from common import BENCH, ROOT, load_jsonl, setup_utf8


def main():
    setup_utf8()
    queries = load_jsonl(BENCH / "queries.jsonl")
    qids = sorted(str(q["query_id"]) for q in queries)
    rng = random.Random(42)
    rng.shuffle(qids)
    n_val = int(len(qids) * 0.2)
    val = sorted(qids[:n_val], key=int)

    out = ROOT / "splits"
    out.mkdir(exist_ok=True)
    (out / "val_qids.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"total={len(qids)} val={len(val)} dev={len(qids) - len(val)}")
    print(f"wrote splits/val_qids.txt (first 5: {val[:5]})")


if __name__ == "__main__":
    main()
