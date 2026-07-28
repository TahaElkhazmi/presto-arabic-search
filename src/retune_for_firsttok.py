"""Protocol rule 2: re-tune every downstream constant after the key changes.

This is the project's most expensive lesson. Every constant in the stack was last fitted
when the in-band key fell through after match-position; the key now carries a fourth
level (first-token position), so the constants encode the previous key's strength.
Both prior key changes paid on re-tune: the prefix ship was followed by a re-tune worth
+0.0020, and the mechanism was coherent rather than arbitrary - a hotter temperature and
a lower lambda lean LESS on the cross-encoder, and a wider band routes MORE candidates
into a key that is now stronger.

The grid is centred on the current values and extended in the direction that mechanism
predicts (hotter T, lower lam, wider bands), so an optimum on the far edge is meaningful
rather than an artifact of a lazy range. Anything on an edge is reported as UNBRACKETED
and is not taken.

The top-k CONSENSUS configuration is taken, not the argmax: the argmax of a grid on 4,006
queries is the single most inflated number available, and consensus has transferred
better in every prior campaign here. Dev-argmax inflation is measured at 3-4x.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
BETA = 0.35
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2

CUR = (3.2, 0.50, 0.030, 0.050, 0.10)          # T, lam, eps, eps_hi, conf_t
G_T = (3.2, 4.2, 5.5)
G_LAM = (0.40, 0.45, 0.50)
G_EPS = (0.030, 0.042)
G_HI = (0.050, 0.065)
G_CT = (0.10, 0.15)


def main():
    # Pass 1 uses the defaults above. Pass 2 extends whichever ranges came back on a
    # grid edge, which is the standing rule: an edge optimum is unbracketed and is not
    # taken until a value on the far side of it has been beaten.
    global G_T, G_LAM, G_EPS, G_HI, G_CT
    p = argparse.ArgumentParser()
    for nm in ("t", "lam", "eps", "hi", "ct"):
        p.add_argument(f"--{nm}", default=None, help="comma list overriding the grid")
    a = p.parse_args()
    f = lambda s: tuple(float(x) for x in s.split(","))
    if a.t:
        G_T = f(a.t)
    if a.lam:
        G_LAM = f(a.lam)
    if a.eps:
        G_EPS = f(a.eps)
    if a.hi:
        G_HI = f(a.hi)
    if a.ct:
        G_CT = f(a.ct)

    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    results = []
    cur_tune = cur_conf = None   # only measurable when the current T is in the grid
    for T in G_T:
        print(f"T={T} ...", flush=True)
        h = Harness(hold, T=T, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
        for q in h.qids:
            v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
            h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
        qrels = h.qrels

        qt = {}
        for q in h.qids:
            t = [x for x in normalize_arabic(h.queries[q]).split()
                 if x not in DIALECT_FILLERS]
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

        def firsttok(q, i):
            v = ftc.get((q, i))
            if v is None:
                toks, qq = ntok[i], qt[q]
                v = 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0
                ftc[(q, i)] = v
            return v

        key = lambda q, i: (h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
                            + firsttok(q, i) * S_FT)
        bonus = lambda q, i: BETA * matchpos(q, i)

        def score(qids, **cfg):
            r = h.build_run(key=key, bonus_fn=bonus, qids=qids, **cfg)
            v = [ndcg_at_k(r[q], qrels[q], 10) for q in qids if q in qrels]
            return sum(v) / len(v)

        for lam in G_LAM:
            for eps in G_EPS:
                for hi in G_HI:
                    for ct in G_CT:
                        s = score(tune, lam=lam, eps=eps, eps_hi=hi, conf_t=ct)
                        results.append((s, T, lam, eps, hi, ct))
        if T == CUR[0]:
            cur_tune = score(tune, lam=CUR[1], eps=CUR[2], eps_hi=CUR[3], conf_t=CUR[4])
            cur_conf = score(conf, lam=CUR[1], eps=CUR[2], eps_hi=CUR[3], conf_t=CUR[4])
        del h

    results.sort(reverse=True)
    if cur_tune is not None:
        print(f"\ncurrent constants on tune: {cur_tune:.6f}   on confirm: {cur_conf:.6f}\n")
    else:
        print(f"\n(current T={CUR[0]} not in this grid, so no in-pass baseline)\n")
    print(f"TOP 10 of {len(results)} configs (tune half, n={len(tune):,})")
    print(f"  {'nDCG':>9}  {'T':>4} {'lam':>5} {'eps':>6} {'eps_hi':>6} {'conf_t':>6}")
    for r in results[:10]:
        print(f"  {r[0]:>9.6f}  {r[1]:>4.1f} {r[2]:>5.2f} {r[3]:>6.3f} "
              f"{r[4]:>6.3f} {r[5]:>6.2f}")

    k = 10
    names = ("T", "lam", "eps", "eps_hi", "conf_t")
    grids = (G_T, G_LAM, G_EPS, G_HI, G_CT)
    print(f"\nCONSENSUS over the top {k}, with bracketing")
    cons = []
    for j, (nm, g) in enumerate(zip(names, grids), start=1):
        vals = [r[j] for r in results[:k]]
        pick = max(set(vals), key=vals.count)
        edge = pick in (min(g), max(g)) and len(g) > 1
        cons.append(pick)
        share = vals.count(pick) / k
        print(f"  {nm:<8s} {pick:<8} chosen by {share:.0%} of the top {k}"
              f"   {'UNBRACKETED (grid edge)' if edge else 'bracketed'}")

    print(f"\n  consensus config: T={cons[0]} lam={cons[1]} eps={cons[2]} "
          f"eps_hi={cons[3]} conf_t={cons[4]}")
    print(f"  current  config: T={CUR[0]} lam={CUR[1]} eps={CUR[2]} "
          f"eps_hi={CUR[3]} conf_t={CUR[4]}")
    if tuple(cons) == CUR:
        print("\n  UNCHANGED - the constants are already right for the new key.")
    else:
        print("\n  changed: run the confirm half + clean arbiter before adopting.")


if __name__ == "__main__":
    main()
