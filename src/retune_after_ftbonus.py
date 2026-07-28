"""Protocol rule 2, again: the blend CHANGED after the constants were last fitted.

`retune_for_firsttok.py` re-fitted T/lam/eps/eps_hi/conf_t for the first-token KEY, and
came back null. But that was run BEFORE `--ft-bonus` existed. Adding 0.05*first_token to
the blend shifts the score distribution every downstream constant is defined against - in
particular the eps band, which is a threshold on score GAPS.

Tier 2 also produced a directional hint worth acting on: capping chain length is harmful
monotonically (cap 3 costs -0.021, cap 20 costs -0.004, uncapped is best) and restricting
the re-sort depth is harmful the same way. More re-sorting is better than less, which
argues for testing a WIDER band now rather than assuming the old one still fits.

Two stages, because the seven-dimensional joint grid is not affordable and the two blend
weights are the newly coupled pair:
  stage 1: beta_mp x beta_ft, everything else at live values
  stage 2: eps, eps_hi, conf_t at the stage-1 winner

Consensus over the top-k is taken rather than the argmax, and any value on a grid edge is
reported as unbracketed and not adopted.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
T_LIVE = 3.2
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
BETA_MP, BETA_FT = 0.35, 0.05
S_CONT, S_MP, S_FT = 1e6, 1e2, 1e-2

G_BMP = (0.25, 0.35, 0.45)
G_BFT = (0.03, 0.05, 0.08)
G_EPS = (0.030, 0.042, 0.055)
G_HI = (0.050, 0.065, 0.085)
G_CT = (0.10, 0.15, 0.22)


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    ntok = {str(r["item_id"]): normalize_arabic(r["name"]).split()
            for r in load_jsonl(BENCH / "corpus.jsonl")}

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
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

    def score(qids, bmp, bft, **cfg):
        r = h.build_run(key=key,
                        bonus_fn=lambda q, i: bmp * mp(q, i) + bft * ft(q, i),
                        qids=qids, **cfg)
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qids if q in qrels]
        return sum(v) / len(v)

    live_t = score(tune, BETA_MP, BETA_FT, **LIVE)
    live_c = score(conf, BETA_MP, BETA_FT, **LIVE)
    print(f"live constants: tune {live_t:.6f}  confirm {live_c:.6f}\n")

    print("STAGE 1: beta_mp x beta_ft")
    res1 = []
    for bmp in G_BMP:
        for bft in G_BFT:
            res1.append((score(tune, bmp, bft, **LIVE), bmp, bft))
    res1.sort(reverse=True)
    for s, a, b in res1[:6]:
        print(f"  {s:.6f}  beta_mp {a:.2f}  beta_ft {b:.2f}")
    bmp_b, bft_b = res1[0][1], res1[0][2]
    edge1 = bmp_b in (min(G_BMP), max(G_BMP)) or bft_b in (min(G_BFT), max(G_BFT))
    print(f"  -> best beta_mp {bmp_b:.2f}, beta_ft {bft_b:.2f}"
          f"{'  UNBRACKETED (grid edge)' if edge1 else '  bracketed'}\n")

    print("STAGE 2: eps x eps_hi x conf_t at those betas")
    res2 = []
    for e in G_EPS:
        for hi in G_HI:
            for ct in G_CT:
                cfg = dict(lam=LIVE["lam"], eps=e, eps_hi=hi, conf_t=ct)
                res2.append((score(tune, bmp_b, bft_b, **cfg), e, hi, ct))
    res2.sort(reverse=True)
    print(f"  {'nDCG':>9}  {'eps':>6} {'eps_hi':>7} {'conf_t':>7}")
    for s, e, hi, ct in res2[:8]:
        print(f"  {s:>9.6f}  {e:>6.3f} {hi:>7.3f} {ct:>7.2f}")

    k = 8
    cons = []
    for j, (nm, g) in enumerate(zip(("eps", "eps_hi", "conf_t"), (G_EPS, G_HI, G_CT)),
                                start=1):
        vals = [r[j] for r in res2[:k]]
        pick = max(set(vals), key=vals.count)
        cons.append(pick)
        print(f"  {nm:<8s} {pick:<8} {vals.count(pick)}/{k} of top-k"
              f"{'   UNBRACKETED' if pick in (min(g), max(g)) else '   bracketed'}")

    cfg = dict(lam=LIVE["lam"], eps=cons[0], eps_hi=cons[1], conf_t=cons[2])
    print(f"\nCANDIDATE: beta_mp {bmp_b:.2f} beta_ft {bft_b:.2f} "
          f"eps {cons[0]} eps_hi {cons[1]} conf_t {cons[2]}")
    ct_ = score(conf, bmp_b, bft_b, **cfg)
    print(f"  tune   {score(tune, bmp_b, bft_b, **cfg):.6f}  (live {live_t:.6f})")
    print(f"  CONFIRM {ct_:.6f}  (live {live_c:.6f})  delta {ct_ - live_c:+.6f}")
    print("\n  a candidate must improve the CONFIRM half before it earns a full run.")


if __name__ == "__main__":
    main()
