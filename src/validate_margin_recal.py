"""Validation of the confidence-margin redefinition (top1-top5 -> top1-top3, conf_t 0.30).

The margin that routes queries into the wide band was fixed in the v2 era (top1-top5,
conf_t 0.10 then 0.15). Under the strong CE, dev_final2000 says a sharper margin at a
higher threshold clears the gate (+0.0027, interior at conf_t=0.30, broad plateau).
Second tuning wave on that split, so inflation risk is ELEVATED; the untouched
11,212 slice is the decisive arbiter.

Measures: live (LB 0.66738: k=60 3:1, T=2.6, lam=0.60, eps=0.010/0.040, top1-top5
margin, conf_t=0.15) vs candidate (same but top1-top3 margin, conf_t=0.30).
"""

from common import BENCH, ROOT, load_jsonl, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from grid_search_prior import set_prior
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, key="cont3")


def mean_ndcg(run, qrels, qids):
    v = [ndcg_at_k(run[q], qrels[q], 10) for q in qids if q in run and q in qrels]
    return sum(v) / len(v)


def measure(label, qids, slices):
    print(f"\n########## {label} ##########", flush=True)
    h = Harness(qids, T=2.6, cache_stem=V7)
    set_prior(h, 60, (3.0, 1.0))

    live_run = h.build_run(conf_t=0.15, **STACK)  # top1-top5 margin (harness default)

    for qid in h.qids:  # switch to top1-top3 margin
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    cand_run = h.build_run(conf_t=0.30, **STACK)
    qrels = h.qrels
    del h

    for sname, qs in slices.items():
        b = mean_ndcg(live_run, qrels, qs & set(qids))
        c = mean_ndcg(cand_run, qrels, qs & set(qids))
        print(f"--- {sname} ({len(qs & set(qids))}) ---")
        print(f"  live (top1-top5, ct=0.15): {b:.6f}")
        print(f"  cand (top1-top3, ct=0.30): {c:.6f}  ({c - b:+.6f})")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    tuned = set()
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt"):
        tuned |= load_qid_file(ROOT / "splits" / f)

    measure("FULL VAL", sorted(val), {"val": val})
    measure("FULL BENCHMARK", all_qids, {
        "full benchmark": set(all_qids),
        "UNTOUCHED (excl 6 dev + val)": set(all_qids) - tuned - val,
    })


if __name__ == "__main__":
    main()
