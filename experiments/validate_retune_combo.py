"""Arbiter gate for PHASE 1 (re-tuned constants) alone and combined with PHASE 2.

Three passes of grid search settled every constant with cross-pass bracketing:
T 2.6->3.2 (2.6 lost in pass 2, 7.0 in pass 3), lam 0.60->0.50 (0.30/0.40 lost),
eps 0.010->0.030 (0.045 lost), eps_hi 0.040->0.050 (0.040 lost in pass 1, 0.070 in
pass 3), conf_t 0.30->0.10 (0.05 lost). The mechanism is coherent: a hotter temperature
and a lower lambda both lean less on the cross-encoder, and a wider band routes more
candidates into a tie-break key that is now strong, which is the opposite of what the
stack wanted when that key fell through to raw id.

The TOP-12 CONSENSUS configuration is taken forward rather than the raw argmax
(0.685078 versus 0.685508 on the tune half). Consensus configurations have transferred
better than argmax in every prior campaign here, and the argmax of a 192-cell grid on
4,006 queries is the single most inflated number available.

Phase 2's blend bonus was independently positive but sub-gate on the arbiter
(+0.001236). Since the two changes touch different parts of the pipeline - one the
blend constants, one an additive term in the blend score - their combination is tested
explicitly rather than assumed additive.
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
SHIPPED = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30)   # T = 2.6
RETUNED = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
BETA = 0.35


def build(T, cfg, beta, all_qids, norm, ntok):
    h = Harness(all_qids, T=T, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
    cache = {}

    def mp(q, i):
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

    bf = None if not beta else (lambda q, i: beta * mp(q, i))
    run = h.build_run(key=lambda q, i: h.cont3(q, i) * 1000.0 + mp(q, i),
                      bonus_fn=bf, **cfg)
    qrels = h.qrels
    del h
    return run, qrels


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}

    print("building shipped baseline ...", flush=True)
    shipped, qrels = build(2.6, SHIPPED, 0.0, all_qids, norm, ntok)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_matchpos.csv")
    same = sum(1 for q in live if q in shipped and live[q][:10] == shipped[q][:10])
    print(f"REGRESSION vs the file on the board: {same:,}/{len(live):,}\n")

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

    def rep(tag, cand):
        out = {}
        for label, qs in (("full", all_qids), ("ARBITER", untouched),
                          ("val", sorted(val, key=int))):
            qs = [q for q in qs if q in qrels]
            a = [ndcg_at_k(shipped[q], qrels[q], 10) for q in qs]
            b = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
            d = [y - x for x, y in zip(a, b)]
            m = sum(d) / len(d)
            se = st.pstdev(d) / len(d) ** 0.5
            out[label] = m
            print(f"  {tag:<22s}{label:<9s} n={len(qs):>6,}  {sum(b)/len(b):.6f}  "
                  f"delta {m:+.6f}  t {m/se if se else 0:+.1f}")
        print()
        return out

    print("PHASE 1 alone: re-tuned constants (T=3.2, lam=0.50, eps=.030/.050, ct=0.10)")
    r1, _ = build(3.2, RETUNED, 0.0, all_qids, norm, ntok)
    o1 = rep("retune", r1)

    print("PHASE 1 + PHASE 2: re-tuned constants plus the beta=0.35 blend bonus")
    r2, _ = build(3.2, RETUNED, BETA, all_qids, norm, ntok)
    o2 = rep("retune+bonus", r2)

    print("GATE (+0.0015 on the ARBITER)")
    for tag, o in (("retune alone", o1), ("retune + bonus", o2)):
        v = o["ARBITER"]
        print(f"  {tag:<18s} arbiter {v:+.6f}  "
              f"{'PASS' if v >= 0.0015 else 'FAIL'}   "
              f"local full {0.683188 + o['full']:.6f}  "
              f"projected LB ~{0.683188 + o['full'] - 0.001:.4f}")


if __name__ == "__main__":
    main()
