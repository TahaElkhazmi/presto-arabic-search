"""Validate token match-position against the SHIPPED prefix key, not the old live file.

The prefix tie-break is already on the leaderboard at 0.67956, so the only decision
left is whether token position is worth a second submission on top of it. Everything
here therefore compares cont3pos against cont3prefix, and the gate is the incremental
+0.0015, not the gain over the pre-prefix baseline.

Token position generalises the prefix test: prefix is the k=0 case, and position also
separates "query appears at token 1" from "query appears at token 6", which prefix
lumps together. Character offset was also tried and is slightly worse (0.682680 vs
0.683188) because it is sensitive to how long the preceding words happen to be.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, \
    normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    h = Harness(all_qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qs_, qt_ = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt_[q] = t or normalize_arabic(h.queries[q]).split()
        qs_[q] = " ".join(qt_[q])

    def prefix(q, i):
        return 1.0 if norm[i].startswith(qs_[q]) else 0.0

    def tokpos(q, i):
        toks, qq = ntok[i], qt_[q]
        n = len(qq)
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    def run_with(fn):
        cfg = dict(LIVE)
        cfg["key"] = lambda q, i: h.cont3(q, i) * 1000.0 + fn(q, i)
        return h.build_run(**cfg)

    shipped = run_with(prefix)
    cand = run_with(tokpos)

    # regression: the shipped key must reproduce the file now on the leaderboard
    live_csv = load_run_csv(ROOT / "runs" / "submission_v7synth_prefix.csv")
    same = sum(1 for q in live_csv if q in shipped and live_csv[q][:10] == shipped[q][:10])
    print(f"REGRESSION vs submission_v7synth_prefix.csv: {same:,}/{len(live_csv):,}")
    print("  (this file is the one currently scoring 0.67956 on the public board)\n")

    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt",
              "dev_listwise200.txt", "probe_filler800.txt", "probe_band600.txt",
              "probe_judge150.txt"):
        p = ROOT / "splits" / f
        if p.exists():
            used |= load_qid_file(p)
    untouched = sorted(set(all_qids) - used, key=int)

    def rep(label, qs):
        qs = [q for q in qs if q in qrels]
        a = [ndcg_at_k(shipped[q], qrels[q], 10) for q in qs]
        b = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
        d = [y - x for x, y in zip(a, b)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<32s} n={len(qs):>6,}  shipped {sum(a)/len(a):.6f}  "
              f"matchpos {sum(b)/len(b):.6f}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")
        return m

    print("INCREMENTAL vs the SHIPPED prefix key")
    full = rep("FULL BENCHMARK (LB proxy)", all_qids)
    rep("UNTOUCHED remainder (arbiter)", untouched)
    rep("val split", sorted(val, key=int))

    ch = sum(1 for q in all_qids if shipped[q][:10] != cand[q][:10])
    print(f"\n  top-10 changed vs shipped: {ch:,} ({100*ch/len(all_qids):.1f}%)")
    print(f"  projected LB: 0.67956 {full:+.5f} = ~{0.67956 + full:.5f}")
    print(f"  gate for a second submission: +0.0015 incremental")


if __name__ == "__main__":
    main()
