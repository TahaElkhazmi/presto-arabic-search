"""TIER 1: apply proven mechanisms to signals that never received them.

Both of the last two wins came from the same audit, not from invention: a mechanism was
validated for one signal and simply never carried to another. So this enumerates the
open cells of that grid.

  signal              as a KEY level      as a BLEND bonus
  containment         shipped             NEVER TRIED        <- B
  match-position      shipped             shipped (0.35)
  first-token pos     shipped             shipped (0.05)
  highest-IDF token   NEVER TRIED         NEVER TRIED        <- A

A. HIGHEST-IDF TOKEN POSITION. "First" is a positional accident of how the query was
   typed; the most DISTINCTIVE token is the one that should locate the product. Same
   3/(1+k) shape as the shipped first-token term, so it inherits the threshold-free
   property, but it points at a different token whenever the query leads with a generic
   word. Token-level IDF is computed over unique normalized catalog names, which is
   catalog-derived and therefore legal under the measurement-only rule.

B. CONTAINMENT AS A BLEND BONUS. cont3 is level 1 of the in-group key but has never
   entered the blend, and the other two levels both pay there.

E. KEY LEVEL ORDER. (containment, match-position, first-token) was assembled in the order
   the levels were discovered, never because that order was measured best.

Screening on the holdout halves at the current live operating point; anything that clears
the confirm half earns a full-benchmark and clean-arbiter pass.
"""

import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)
T_LIVE, BETA_MP, BETA_FT = 3.2, 0.35, 0.05
S_CONT, S_MP, S_FT, S_X = 1e6, 1e2, 1e-2, 1e-5


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    ntok = {i: n.split() for i, n in norm.items()}

    # token IDF over UNIQUE normalized names (catalog-derived, rule-2.4.b clean)
    uniq = set(norm.values())
    df = {}
    for n in uniq:
        for t in set(n.split()):
            df[t] = df.get(t, 0) + 1
    N = len(uniq)
    idf = lambda t: math.log(1.0 + N / (df.get(t, 0) + 0.5))

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, qbest = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        qbest[q] = max(t, key=idf) if t else None
    same_as_first = sum(1 for q in h.qids if qt[q] and qbest[q] == qt[q][0])
    print(f"highest-IDF token equals the FIRST token on {same_as_first:,}/{len(h.qids):,} "
          f"queries ({same_as_first/len(h.qids):.1%}) - the rest is where A can differ\n")

    mpc, ftc, xic = {}, {}, {}

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

    def xi(q, i):
        """position of the highest-IDF query token"""
        v = xic.get((q, i))
        if v is None:
            toks, b = ntok[i], qbest[q]
            v = 3.0 / (1.0 + toks.index(b)) if (b is not None and b in toks) else 0.0
            xic[(q, i)] = v
        return v

    base_key = lambda q, i: h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT
    base_bonus = lambda q, i: BETA_MP * mp(q, i) + BETA_FT * ft(q, i)

    def run(key, bonus, qids=None):
        return h.build_run(key=key, bonus_fn=bonus, qids=qids, **LIVE)

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    base = run(base_key, base_bonus)
    bt, bc = sc(base, tune), sc(base, conf)
    print(f"baseline (current best, local 0.689919): tune {bt:.6f}  confirm {bc:.6f}\n")

    variants = {}
    # A: highest-IDF token position, as a 4th key level and as a blend bonus
    variants["A1 idf-tok as key level"] = (
        lambda q, i: base_key(q, i) + xi(q, i) * S_X, base_bonus)
    for b in (0.02, 0.05, 0.10):
        variants[f"A2 idf-tok bonus {b:.2f}"] = (
            base_key, lambda q, i, b=b: base_bonus(q, i) + b * xi(q, i))
    # B: containment as a blend bonus
    for g in (0.02, 0.05, 0.10, 0.20):
        variants[f"B  cont3 bonus {g:.2f}"] = (
            base_key, lambda q, i, g=g: base_bonus(q, i) + g * h.cont3(q, i))
    # E: permute the key level order
    variants["E1 key cont,ft,mp"] = (
        lambda q, i: h.cont3(q, i) * S_CONT + ft(q, i) * S_MP / 3.0 + mp(q, i) * S_FT,
        base_bonus)

    print(f"  {'variant':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>9s}")
    survivors = []
    for nm, (k, b) in variants.items():
        rt, rc = run(k, b, tune), run(k, b, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0.0
        ok = m >= 0.0015 and t > 2
        if ok:
            survivors.append(nm)
        print(f"  {nm:<26s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}{'CARRY' if ok else 'drop':>9s}")

    print(f"\n  survivors: {survivors if survivors else 'none'}")


if __name__ == "__main__":
    main()
