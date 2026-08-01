"""Arbiter gate for PHASE 2: match-position as an additive blend term (beta = 0.35).

Selected on the tune half of the clean holdout with a bracketed optimum (0.350 beats
both 0.200 and 0.600), confirmed at +0.001690 (t=+2.18) on the confirm half. This runs
the decisive slice: the untouched remainder, which is excluded from every split ever
used for tuning or gating and is the designated private-leaderboard proxy.

Baseline is the SHIPPED cascade (containment, token match-position, id at T=2.6,
lam=0.60, eps=0.010/0.040, conf_t=0.30), so the delta is incremental over what is
currently on the leaderboard at 0.67956.

Pre-registered decision rule: ship only if the arbiter delta is at least +0.0015.
beta was chosen on the tune half and is NOT re-selected here; a lower beta scored
slightly better on the confirm half, and switching to it post-hoc is exactly the
failure this protocol exists to prevent.
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
SHIPPED = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30)
BETA = 0.35


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

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()

    cache = {}

    def matchpos(q, i):
        v = cache.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            cache[(q, i)] = v
        return v

    keyf = lambda q, i: h.cont3(q, i) * 1000.0 + matchpos(q, i)
    shipped = h.build_run(key=keyf, **SHIPPED)
    cand = h.build_run(key=keyf, bonus_fn=lambda q, i: BETA * matchpos(q, i),
                       **SHIPPED)

    live = load_run_csv(ROOT / "runs" / "submission_v7synth_matchpos.csv")
    same = sum(1 for q in live if q in shipped and live[q][:10] == shipped[q][:10])
    print(f"REGRESSION vs submission_v7synth_matchpos.csv: {same:,}/{len(live):,}")
    if same != len(live):
        print("  !! baseline does not reproduce the shipped file; numbers are suspect")
    print()

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
              f"cand {sum(b)/len(b):.6f}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")
        return m

    print(f"PHASE 2 ARBITER GATE (beta={BETA})")
    full = rep("FULL BENCHMARK (LB proxy)", all_qids)
    arb = rep("UNTOUCHED remainder (ARBITER)", untouched)
    rep("val split", sorted(val, key=int))

    ch = sum(1 for q in all_qids if shipped[q][:10] != cand[q][:10])
    print(f"\n  top-10 changed: {ch:,} ({100*ch/len(all_qids):.1f}%)")
    print(f"  local full benchmark: 0.683188 -> {0.683188 + full:.6f}")
    print(f"  projected LB: ~{0.683188 + full - 0.001:.4f}")
    print(f"\n  GATE (+0.0015 on the arbiter): "
          f"{'PASS' if arb >= 0.0015 else 'FAIL'}  (arbiter delta {arb:+.6f})")


if __name__ == "__main__":
    main()
