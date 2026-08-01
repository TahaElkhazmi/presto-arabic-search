"""Validate FIRST-TOKEN POSITION as a third level in the in-band key.

WHAT IT IS. The shipped key is (containment desc, match-position desc, id asc), where
match-position only fires when the query tokens appear CONTIGUOUSLY in the title. Every
title that contains the query tokens scattered, or contains only the first of them,
scores match-position 0 and drops straight to raw id. First-token position asks the same
Meilisearch "attribute position" question one level down: where does the FIRST query
token land? It is bucketed, so it PARTITIONS and the low-id prior still orders inside
each class - the property that made the prefix key win and made GATE-AraBert lose.

HOW IT WAS FOUND. `probe_phase3_predicates.py` retired nine features by comparing each
one's STANDALONE within-cell AUC against low-id. That is the wrong gate: the cascade
sorts by (-feature, +id), so the right quantity is the AUC of the COMBINED key. Measured
standalone, first-token position scores 0.5194 against the incumbent's 0.6230 and is
discarded; measured as the cascade actually uses it, the combination scores 0.6368 and
0.6978 versus 0.6230 and 0.6948, and it is worth +0.0023 end to end.

SPLIT HYGIENE, worth stating plainly. `holdout_remainder8012` is 6,891 of the 9,675
queries in the "untouched remainder", so the arbiter that recent campaigns reported was
71% the pool they tuned on. This script therefore also reports ARBITER MINUS HOLDOUT
(2,784 queries), which no campaign has touched, as the strictest available number.

ROBUSTNESS. The bucket edges are a free choice, so four encodings are compared. If the
gain only survives one of them it is a threshold artifact, not a signal.
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
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE, BETA = 3.2, 0.35
S_CONT, S_MP, S_FEAT = 1e6, 1e2, 1e-5


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

    def matchpos(q, i):
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

    def first_k(q, i):
        """Index of the first query token in the title, or -1 if absent."""
        v = ftc.get((q, i))
        if v is None:
            toks, qq = ntok[i], qt[q]
            v = toks.index(qq[0]) if (qq and qq[0] in toks) else -1
            ftc[(q, i)] = v
        return v

    def enc_coarse(q, i):
        k = first_k(q, i)
        return 0.0 if k < 0 else (3.0 if k == 0 else (2.0 if k <= 2 else 1.0))

    def enc_binary(q, i):
        return 0.0 if first_k(q, i) < 0 else 1.0

    def enc_recip(q, i):
        k = first_k(q, i)
        return 0.0 if k < 0 else 3.0 / (1.0 + k)

    def enc_fine(q, i):
        k = first_k(q, i)
        if k < 0:
            return 0.0
        return 4.0 if k == 0 else (3.0 if k == 1 else (2.0 if k <= 3 else 1.0))

    encodings = [("coarse {0,2,3}", enc_coarse), ("binary present", enc_binary),
                 ("reciprocal 3/(1+k)", enc_recip), ("fine {0..4}", enc_fine)]

    def build(fn=None, qids=None):
        if fn is None:
            key = lambda q, i: h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
        else:
            key = lambda q, i: (h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
                                + fn(q, i) * S_FEAT)
        return h.build_run(key=key, bonus_fn=lambda q, i: BETA * matchpos(q, i),
                           qids=qids, **LIVE)

    base = build()
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_retune_bonus.csv")
    same = sum(1 for q in live if q in base and live[q][:10] == base[q][:10])
    print(f"REGRESSION vs the file on the board (LB 0.68527): {same:,}/{len(live):,}")
    if same != len(live):
        print("  *** baseline does NOT reproduce the live file - stop ***")
        return

    # slices
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    val = load_qid_file(ROOT / "splits" / "val_qids.txt")
    used = set(val)
    for f in ("dev_rand2000", "dev_unseen2000", "dev_probe2000", "dev_recal2000",
              "dev_ens2000", "dev_final2000", "dev_listwise200", "probe_filler800",
              "probe_band600", "probe_judge150"):
        p = ROOT / "splits" / f"{f}.txt"
        if p.exists():
            used |= load_qid_file(p)
    arbiter = sorted(set(all_qids) - used, key=int)
    clean = sorted(set(arbiter) - set(hold), key=int)

    slices = [("tune half (fitted)", tune), ("confirm half", conf),
              ("ARBITER (9,675)", arbiter), ("CLEAN arbiter (no holdout)", clean),
              ("val", sorted(val, key=int)), ("FULL BENCHMARK", all_qids)]

    print("\nROBUSTNESS TO THE BUCKET CHOICE (delta vs the shipped key)\n")
    hdr = f"  {'encoding':<22s}" + "".join(f"{n.split(' (')[0]:>15s}" for n, _ in slices)
    print(hdr)
    results = {}
    for name, fn in encodings:
        cand = build(fn)
        row = f"  {name:<22s}"
        for _, qs in slices:
            qs = [q for q in qs if q in qrels]
            d = [ndcg_at_k(cand[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
                 for q in qs]
            row += f"{sum(d)/len(d):>+15.6f}"
        results[name] = cand
        print(row)

    best = "coarse {0,2,3}"
    cand = results[best]
    print(f"\nDETAIL for '{best}' (the encoding carried forward)\n")
    for label, qs in slices:
        qs = [q for q in qs if q in qrels]
        a = [ndcg_at_k(base[q], qrels[q], 10) for q in qs]
        b = [ndcg_at_k(cand[q], qrels[q], 10) for q in qs]
        d = [y - x for x, y in zip(a, b)]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {label:<28s} n={len(qs):>6,}  live {sum(a)/len(a):.6f}  "
              f"cand {sum(b)/len(b):.6f}  delta {m:+.6f}  t {m/se if se else 0:+.1f}")

    full = [q for q in all_qids if q in qrels]
    fm = sum(ndcg_at_k(cand[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in full) / len(full)
    ch = sum(1 for q in all_qids if base[q][:10] != cand[q][:10])
    print(f"\n  top-10 changed: {ch:,} ({100*ch/len(all_qids):.1f}%)")
    print(f"  local full benchmark: 0.684938 {fm:+.6f} = {0.684938 + fm:.6f}")
    print(f"  projected LB (local minus ~0.001): ~{0.684938 + fm - 0.001:.4f}")
    print("  GATE: +0.0015 incremental on a slice that was not fitted.")


if __name__ == "__main__":
    main()
