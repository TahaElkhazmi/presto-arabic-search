"""Validation of the v7synth recalibration: full val + full benchmark + untouched slice.

Tuning is FINISHED (three passes on dev_recal2000, all five dims bracketed on both
sides). This script only MEASURES, on data the recalibration never saw.

Configs:
  submitted : T=1.6 lam=0.50 eps=0.020 eps_hi=0.080 conf_t=0.10  (the live LB 0.66231 file)
  robust    : T=2.6 lam=0.60 eps=0.010 eps_hi=0.030 conf_t=0.15  (top-15 consensus)
  robust_b  : T=2.6 lam=0.60 eps=0.010 eps_hi=0.040 conf_t=0.15  (eps_hi sensitivity)
  argmax    : T=3.0 lam=0.60 eps=0.010 eps_hi=0.030 conf_t=0.15  (dev peak, expected to be inflated)

The "untouched" slice excludes every split ever used for tuning or gating
(dev_rand2000, dev_unseen2000, dev_probe2000, dev_recal2000, val_qids) and is the
cleanest available proxy for the private leaderboard.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"

CONFIGS = {
    "submitted (live 0.66231)": dict(T=1.6, lam=0.50, eps=0.020, eps_hi=0.080, conf_t=0.10),
    "robust    (consensus)   ": dict(T=2.6, lam=0.60, eps=0.010, eps_hi=0.030, conf_t=0.15),
    "robust_b  (eps_hi .04)  ": dict(T=2.6, lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.15),
    "argmax    (dev peak)    ": dict(T=3.0, lam=0.60, eps=0.010, eps_hi=0.030, conf_t=0.15),
}


def mean_ndcg(run, qrels, qids):
    v = [ndcg_at_k(run[q], qrels[q], 10) for q in qids if q in run and q in qrels]
    return sum(v) / len(v)


def run_slice(label, qids, slices):
    print(f"\n########## {label} ##########", flush=True)
    runs = {}
    qrels = None
    for name, cfg in CONFIGS.items():
        h = Harness(qids, T=cfg["T"], cache_stem=V7)
        qrels = h.qrels
        runs[name] = h.build_run(lam=cfg["lam"], eps=cfg["eps"], eps_hi=cfg["eps_hi"],
                                 conf_t=cfg["conf_t"], key="cont3")
        del h
    base = None
    for sname, qs in slices.items():
        print(f"\n--- {sname} ({len(qs & set(qids))} queries) ---")
        for name, run in runs.items():
            s = mean_ndcg(run, qrels, qs & set(qids))
            if name.startswith("submitted"):
                base = s
                print(f"  {name}: {s:.6f}   [baseline]")
            else:
                print(f"  {name}: {s:.6f}  ({s - base:+.6f})")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    tuned = set()
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt"):
        tuned |= load_qid_file(ROOT / "splits" / f)

    run_slice("FULL VAL (5,802) - first independent read", sorted(val), {"val": val})

    run_slice("FULL BENCHMARK (29,014)", all_qids, {
        "full benchmark": set(all_qids),
        "UNTOUCHED (excl all tuning+val splits)": set(all_qids) - tuned - val,
    })


if __name__ == "__main__":
    main()
