"""Full-benchmark and clean-arbiter check for the first-token blend bonus.

The holdout screen gave an odd shape: tune half +0.0004, confirm half +0.0029 (t +4.0),
with an interior optimum at beta_ft=0.05 on both halves. Tune/confirm divergence that
large is usually split variance rather than signal, and it runs the OPPOSITE way to the
usual dev-argmax inflation, so it cannot be waved through on the confirm number alone.
This re-measures on every slice, including the 2,784 queries no campaign has touched.
"""

import random
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
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
T_LIVE, BETA_MP = 3.2, 0.35
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2
BETAS = (0.03, 0.05, 0.08)

SPENT = ("dev_rand2000", "dev_unseen2000", "dev_probe2000", "dev_recal2000",
         "dev_ens2000", "dev_final2000", "dev_listwise200", "probe_filler800",
         "probe_band600", "probe_judge150")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    print("building harness over the full benchmark ...", flush=True)
    h = Harness(all_qids, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt = {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        qt[q] = t or normalize_arabic(h.queries[q]).split()
    mpc, ftc = {}, {}

    def mp(q, i):
        v = mpc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            n = len(qq)
            v = 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    v = 1.0 / (1.0 + k)
                    break
            mpc[(q, i)] = v
        return v

    def ft(q, i):
        v = ftc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
            ftc[(q, i)] = v
        return v

    key = lambda q, i: h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT

    def build(b):
        return h.build_run(key=key,
                           bonus_fn=lambda q, i: BETA_MP * mp(q, i) + b * ft(q, i),
                           **LIVE)

    base = build(0.0)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_firsttok.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs the file on the board (LB 0.68822): {same:,}/{len(live):,}")
    if same != len(live):
        print("  *** baseline does not reproduce the live file - stop ***")
        return

    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune, conf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])
    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in SPENT:
        p = ROOT / "splits" / f"{f}.txt"
        if p.exists():
            used |= load_qid_file(p)
    arb = set(all_qids) - used
    slices = [("tune half (fitted)", tune), ("confirm half", conf),
              ("ARBITER (9,675)", arb), ("CLEAN arbiter", arb - set(hold)),
              ("val", val), ("FULL BENCHMARK", set(all_qids))]

    for b in BETAS:
        cand = build(b)
        print(f"\nbeta_ft = {b:.2f}")
        for label, qs in slices:
            qs = [q for q in sorted(qs, key=int) if q in qrels]
            x = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
            y = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
            d = [v - u for u, v in zip(x, y)]
            m = sum(d) / len(d)
            se = st.pstdev(d) / len(d) ** 0.5
            print(f"  {label:<22s} n={len(qs):>6,}  {sum(y)/len(y):.6f}  "
                  f"delta {m:+.6f}  t {m/se if se else 0:+.1f}")
        full = [q for q in all_qids if q in qrels]
        fm = sum(ndcg_at_k(cand[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
                 for q in full) / len(full)
        print(f"  -> local {0.687911 + fm:.6f}, projected LB ~{0.687911 + fm - 0.001:.4f}")


if __name__ == "__main__":
    main()
