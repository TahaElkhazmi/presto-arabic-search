"""Materialise the two slices the model gate uses, so both are files on disk.

Until now the CLEAN arbiter was recomputed inline by every validate_* script as
"all queries minus val minus the six spent dev splits minus the small probes, minus
holdout_remainder8012". Recomputing a gate slice per script is exactly how a slice
definition drifts, so it is written once, here, and read as a file thereafter.

  clean_arbiter2784   the untouched gate. Nothing may be fitted on it, ever.
  gate_retune2000     a deterministic 2,000-query sample of the holdout TUNE half,
                      which recent campaigns already tuned on, so spending it costs
                      nothing that was not already spent. Model re-tuning happens here.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, setup_utf8

setup_utf8()

SPENT = ("dev_rand2000", "dev_unseen2000", "dev_probe2000", "dev_recal2000",
         "dev_ens2000", "dev_final2000", "dev_listwise200", "probe_filler800",
         "probe_band600", "probe_judge150")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    used = set(load_qid_file(ROOT / "splits" / "val_qids.txt"))
    for f in SPENT:
        p = ROOT / "splits" / f"{f}.txt"
        if p.exists():
            used |= load_qid_file(p)
    hold = load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt")

    arb = set(all_qids) - used
    clean = sorted(arb - hold, key=int)
    print(f"arbiter {len(arb):,}  clean arbiter {len(clean):,}")
    assert len(clean) == 2784, f"clean arbiter is {len(clean)}, expected 2,784"

    # same shuffle as every validate_* script, so "tune half" means the same queries
    shuf = sorted(hold, key=int)
    random.Random(77).shuffle(shuf)
    tune = shuf[: len(shuf) // 2]
    rt = sorted(random.Random(4242).sample(tune, 2000), key=int)
    print(f"tune half {len(tune):,}  ->  gate_retune2000 {len(rt):,}")
    assert not (set(rt) & set(clean)), "retune slice leaked into the clean arbiter"

    for name, qs in (("clean_arbiter2784", clean), ("gate_retune2000", rt)):
        path = ROOT / "splits" / f"{name}.txt"
        path.write_text("\n".join(qs) + "\n", encoding="utf-8")
        print(f"wrote {path.name} ({len(qs):,} queries)")


if __name__ == "__main__":
    main()
