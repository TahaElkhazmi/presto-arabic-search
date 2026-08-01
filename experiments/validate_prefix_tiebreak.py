"""Full-benchmark validation of the prefix tie-break, under the standing protocol.

Candidate: inside the eps near-tie band, order equal-containment items by whether the
product title STARTS WITH the (filler-stripped) query, before falling back to the
ascending-id prior. Pure text, zero GPU, nothing tuned.

Discovered because the premise "the tied items are identical strings" turned out to be
false: saturated groups average 90.9 members but 57.3 DISTINCT titles, and 0 of 993
groups were a single repeated string. Containment saturates at 1.000 whenever a title
contains every query gram, no matter what else the title says, so the information was
never absent, only invisible to the key.

Protocol, in order, matching how every shipped gain in this project was validated:
  0. REGRESSION: the harness at live settings must reproduce the live submission file
     exactly, otherwise no number below is trustworthy.
  1. full benchmark (29,014) - the leaderboard proxy, local minus ~0.001.
  2. the UNTOUCHED remainder (excludes val and every named dev split) - the private
     leaderboard proxy and designated final arbiter.
  3. val split, for continuity with earlier reporting.
Nothing here was fitted on any of these slices; the feature is threshold-free.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
LIVE_CSV = ROOT / "runs" / "submission_v7synth_margin3.csv"


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}

    print(f"building harness for {len(all_qids):,} queries (0 GPU) ...", flush=True)
    h = Harness(all_qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    print(f"missing CE scores: {h.missing_ce}\n")

    qstr = {}
    for q in h.qids:
        toks = [t for t in normalize_arabic(h.queries[q]).split()
                if t not in DIALECT_FILLERS]
        qstr[q] = " ".join(toks) if toks else normalize_arabic(h.queries[q])

    qrels = h.qrels
    base_run = h.build_run(**LIVE_STACK)

    # ---- 0. regression against the live file ----
    live = load_run_csv(LIVE_CSV)
    same = sum(1 for q in live if q in base_run and live[q][:10] == base_run[q][:10])
    print(f"REGRESSION vs {LIVE_CSV.name}: {same:,}/{len(live):,} identical top-10s")
    if same != len(live):
        print("  !! harness does not reproduce the live file; numbers below are suspect")
    else:
        print("  harness reproduces the live submission exactly\n")

    cfg = dict(LIVE_STACK)
    cfg["key"] = lambda q, i: h.cont3(q, i) * 1000.0 + (
        1.0 if norm[i].startswith(qstr[q]) else 0.0)
    cand_run = h.build_run(**cfg)

    # ---- slices ----
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt",
              "dev_listwise200.txt", "probe_filler800.txt", "probe_band600.txt",
              "probe_judge150.txt"):
        p = ROOT / "splits" / f
        if p.exists():
            used |= load_qid_file(p)
    untouched = set(all_qids) - used

    def report(label, qs):
        qs = [q for q in qs if q in qrels]
        b = [ndcg_at_k(base_run[q], qrels[q], 10) for q in qs]
        c = [ndcg_at_k(cand_run[q], qrels[q], 10) for q in qs]
        d = [x - y for x, y in zip(c, b)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<34s} n={len(qs):>6,}  live {sum(b) / len(b):.6f}  "
              f"cand {sum(c) / len(c):.6f}  delta {m:+.6f}  t {m / se if se else 0:+.1f}")
        return m

    print("VALIDATION")
    full = report("FULL BENCHMARK (LB proxy)", all_qids)
    report("UNTOUCHED remainder (arbiter)", sorted(untouched, key=int))
    report("val split", sorted(val, key=int))

    changed = sum(1 for q in all_qids
                  if q in base_run and base_run[q][:10] != cand_run[q][:10])
    print(f"\n  top-10 changed on {changed:,}/{len(all_qids):,} queries "
          f"({100 * changed / len(all_qids):.1f}%)")
    live_local = sum(ndcg_at_k(base_run[q], qrels[q], 10)
                     for q in all_qids if q in qrels) / len(qrels)
    print(f"  local full-benchmark: {live_local:.6f} -> {live_local + full:.6f}")
    print(f"  projected LB (local minus ~0.001): ~{live_local + full - 0.001:.4f}  "
          f"(current LB 0.66861)")
    print(f"\n  gate: +0.0015 on the untouched remainder")


if __name__ == "__main__":
    main()
