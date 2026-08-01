"""Re-tune the stack for the new prefix key, per the project's own standing rule.

Every constant in the live stack (T=2.6, lam=0.60, eps=0.010, eps_hi=0.040,
conf_t=0.30) was fitted when the in-band key was (containment desc, id asc). The key
is now (containment desc, starts-with-query desc, id asc), which changes what the eps
band actually does: the band used to hand ambiguous queries to a key that fell through
to raw id, and now hands them to one that discriminates. A band sized for a weak
in-band key is probably mis-sized for a stronger one.

This is the same move that recovered +0.005 after the v7synth swap, and the rule it
follows is the project's most expensive lesson: re-tune every downstream constant
BEFORE judging a change.

Discipline carried over from those campaigns:
  - tune on one half of the clean holdout, confirm on the other half, then the full
    benchmark and the untouched arbiter;
  - require an INTERIOR optimum, beaten on both sides, not a grid-edge value;
  - prefer a top-k consensus configuration over the raw argmax, because dev-argmax
    inflation on this project measures 3-4x.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
# PASS 2. Pass 1 put four of five dimensions on a grid EDGE (T at the max, eps at the
# max, eps_hi at the max, conf_t at the min), which by the standing protocol means the
# optimum is unbracketed and the value cannot be trusted. Each range is extended in the
# direction the pass-1 consensus pushed, so every dimension can be beaten on BOTH sides.
# PASS 3, the last one. Across passes 1 and 2, eps_hi (0.060 beaten on both sides),
# conf_t (0.10-0.20) and eps (0.020-0.030) are now genuinely bracketed. T and lam are
# not: T sat at the max both times and lam collapsed to the grid minimum in 9 of the
# top 12, so both ranges are opened in the direction the consensus pushed. Higher T and
# lower lam are the same mechanism from two sides - lean less on the cross-encoder and
# route more candidates into a tie-break key that is now strong.
# Stopping after this pass by pre-registered discipline: a fourth wave on the same tune
# half starts fitting the split, which is the failure documented five times in FINDINGS.
GRID_T = (3.2, 4.4, 5.5, 7.0)
GRID_LAM = (0.30, 0.40, 0.50, 0.60)
GRID_EPS = (0.020, 0.030)
GRID_HI = (0.050, 0.060, 0.070)
GRID_CT = (0.10, 0.20)
LIVE = (2.6, 0.60, 0.010, 0.040, 0.30)


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    import random
    shuf = list(qids)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    results = []
    for T in GRID_T:
        h = Harness(qids, T=T, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
        for q in h.qids:
            v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
            h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
        qstr = {}
        for q in h.qids:
            t = [x for x in normalize_arabic(h.queries[q]).split()
                 if x not in DIALECT_FILLERS]
            qstr[q] = " ".join(t) if t else normalize_arabic(h.queries[q])
        qtoks = {q: qstr[q].split() for q in h.qids}
        ntok = {i: n.split() for i, n in norm.items()}

        def matchpos(q, i):
            toks, qq = ntok[i], qtoks[q]
            n = len(qq)
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    return 1.0 / (1.0 + k)
            return 0.0

        keyf = lambda q, i: h.cont3(q, i) * 1000.0 + matchpos(q, i)
        qrels = h.qrels

        def sc(qs, **cfg):
            r = h.build_run(qids=qs, key=keyf, **cfg)
            v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r]
            return sum(v) / len(v)

        for lam in GRID_LAM:
            for eps in GRID_EPS:
                for hi in GRID_HI:
                    for ct in GRID_CT:
                        s = sc(tune, lam=lam, eps=eps, eps_hi=hi, conf_t=ct)
                        results.append((s, T, lam, eps, hi, ct))
        print(f"  T={T} done ({len(results)} configs)", flush=True)
        del h

    results.sort(reverse=True)
    print(f"\nTOP 12 of {len(results)} configs on the tune half")
    print(f"  {'nDCG':>9s}{'T':>6s}{'lam':>6s}{'eps':>7s}{'eps_hi':>8s}{'conf_t':>8s}")
    for r in results[:12]:
        print(f"  {r[0]:>9.6f}{r[1]:>6.1f}{r[2]:>6.2f}{r[3]:>7.3f}{r[4]:>8.3f}{r[5]:>8.2f}")

    live_tune = next((r[0] for r in results if tuple(r[1:]) == LIVE), None)
    if live_tune is None:
        # the live point can fall outside an extended grid; score it explicitly so
        # every pass still reports a delta against the shipped configuration
        hL = Harness(qids, T=LIVE[0], cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
        for q in hL.qids:
            v = sorted((hL.ce[(q, i)] for i in hL.pools[q]), reverse=True)
            hL.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
        qsL = {}
        for q in hL.qids:
            t = [x for x in normalize_arabic(hL.queries[q]).split()
                 if x not in DIALECT_FILLERS]
            qsL[q] = (" ".join(t) if t else normalize_arabic(hL.queries[q])).split()
        ntokL = {i: n.split() for i, n in norm.items()}

        def mpL(q, i):
            toks, qq = ntokL[i], qsL[q]
            n = len(qq)
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == qq:
                    return 1.0 / (1.0 + k)
            return 0.0

        r = hL.build_run(qids=tune, lam=LIVE[1], eps=LIVE[2], eps_hi=LIVE[3],
                         conf_t=LIVE[4],
                         key=lambda q, i: hL.cont3(q, i) * 1000.0 + mpL(q, i))
        v = [ndcg_at_k(r[q], hL.qrels[q], 10) for q in tune if q in r]
        live_tune = sum(v) / len(v)
        del hL
    print(f"\n  live config on tune half: {live_tune:.6f}")
    print(f"  best  config on tune half: {results[0][0]:.6f} "
          f"({results[0][0] - live_tune:+.6f})")

    # boundary audit over the top 12
    top = results[:12]
    for name, i, grid in (("T", 1, GRID_T), ("lam", 2, GRID_LAM), ("eps", 3, GRID_EPS),
                          ("eps_hi", 4, GRID_HI), ("conf_t", 5, GRID_CT)):
        vals = [r[i] for r in top]
        mode = max(set(vals), key=vals.count)
        edge = "EDGE" if mode in (min(grid), max(grid)) else "interior"
        print(f"  {name:<8s} consensus {mode} ({vals.count(mode)}/12)  {edge}")

    print(f"\n  NOTE: an EDGE consensus means the optimum is unbracketed and the grid")
    print(f"  must be extended before the value is trusted, per prior campaigns.")


if __name__ == "__main__":
    main()
