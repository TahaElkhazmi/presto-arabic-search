"""Validation of the prior-composition retune: full val + full benchmark + untouched.

Measurement only; tuning finished (pass 1 + bracketing pass, all dims interior:
k=60 beats 40 and 80/100/150; ratio 3:1-4:1 beats 2.5:1 and 5:1/6:1; lam 0.60;
eps joint-check confirmed 0.010/0.040 carries over unchanged).

Configs (pre-registered before any validation number was seen):
  submitted : k=10, 2:1  (the live LB 0.66550 composition)
  primary   : k=60, 3:1  (top-15 consensus AND argmax)
  alt       : k=60, 4:1  (ratio sensitivity)
Untouched slice now excludes SIX dev splits + val = 11,212 clean queries.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from grid_search_prior import set_prior
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.15, key="cont3")

COMPOSITIONS = {
    "submitted k=10 2:1 (LB 0.66550)": (10, (2.0, 1.0)),
    "primary   k=60 3:1            ": (60, (3.0, 1.0)),
    "alt       k=60 4:1            ": (60, (4.0, 1.0)),
}


def mean_ndcg(run, qrels, qids):
    v = [ndcg_at_k(run[q], qrels[q], 10) for q in qids if q in run and q in qrels]
    return sum(v) / len(v)


def measure(label, qids, slices):
    print(f"\n########## {label} ##########", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7)
    runs = {}
    for name, (k, w) in COMPOSITIONS.items():
        set_prior(h, k, w)
        runs[name] = h.build_run(**STACK)
    qrels = h.qrels
    del h
    for sname, qs in slices.items():
        print(f"\n--- {sname} ({len(qs & set(qids))} queries) ---")
        base = None
        for name, run in runs.items():
            s = mean_ndcg(run, qrels, qs & set(qids))
            if base is None:
                base = s
                print(f"  {name}: {s:.6f}   [baseline]")
            else:
                print(f"  {name}: {s:.6f}  ({s - base:+.6f})")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    tuned = set()
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt"):
        tuned |= load_qid_file(ROOT / "splits" / f)

    measure("FULL VAL (5,802)", sorted(val), {"val": val})
    measure("FULL BENCHMARK (29,014)", all_qids, {
        "full benchmark": set(all_qids),
        "UNTOUCHED (excl 6 dev splits + val)": set(all_qids) - tuned - val,
    })


if __name__ == "__main__":
    main()
