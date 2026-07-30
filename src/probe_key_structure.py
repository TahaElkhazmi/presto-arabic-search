"""The tie-break key as a structure: ablate its levels, and permute their order.

Justified by measurement rather than plausibility: `probe_triage_specificity.py` shows
**75.8% of the oracle-inversion swap gain inside the shipped top-10 is INTRA-chain**, so the
key is the component that had its chance and got it wrong. Blend-side signals are spent.

Two things the project has never done to the key:

ABLATION. Every level was added on top of the stack as it existed at the time, and the band
constants, the top-5 widening and the three blend bonuses all POSTDATE some of them. A level
that earned its place against a weaker downstream stack may now be inert or harmful, in which
case deleting it is free. Note last-token-position in particular shipped JOINTLY with two
other changes as a +0.0039 stack and has never been isolated since.

PERMUTATION. The level order is globally fixed and has been permuted exactly once. A
lexicographic key's order is a real degree of freedom: whichever level comes first decides,
and the rest only break its ties.

Both are pure CPU re-sorts over cached scores. Fitted on the tune half, read on the confirm
half, which is the same protocol every other post-processing change in this project used.
"""

import itertools
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ROOT, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_ensemble_current import (B_IDF, BETA_FT, BETA_MP, CONF_T, EPS, EPS_HI, LAM,
                                    RRF_MAX, TOP_EPS, TOP_K, Stack3)
from probe_harness import _id_key

setup_utf8()

# scale factors for the shipped lexicographic key, most significant first
LEVELS = ("cont3", "mp", "ft", "lt")


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    print(f"building harness over {len(hold):,} holdout queries ...", flush=True)
    S = Stack3(hold)
    h = S.h
    qrels = h.qrels

    getters = {
        "cont3": lambda q, i: h.cont3(q, i),
        "mp": lambda q, i: S.mp(q, i),
        "ft": lambda q, i: S.ft(q, i),
        "lt": lambda q, i: S.lt(q, i),
    }

    # blended score is fixed across every configuration; only the key changes
    order_cache = {}
    for q in h.qids:
        scored = []
        for i in h.pools[q]:
            v = (LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / RRF_MAX
                 + BETA_MP * S.mp(q, i) + BETA_FT * S.ft(q, i) + B_IDF * S.xi(q, i))
            scored.append((v, i))
        scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
        ces = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        conf = ces[0] - ces[2] if len(ces) >= 3 else 1.0
        order_cache[q] = (scored, EPS_HI if conf < CONF_T else EPS)

    def build(levels):
        """levels: tuple of level names, most significant first. () = id only."""
        out = {}
        for q in h.qids:
            scored, e0 = order_cache[q]
            gk = lambda x, qq=q: (tuple(-getters[n](qq, x[1]) for n in levels),
                                  _id_key(x[1]))
            res, grp = [], []
            for posn, (sc, iid) in enumerate(scored):
                e = TOP_EPS if posn < TOP_K else e0
                if grp and (grp[-1][0] - sc) > e:
                    grp.sort(key=gk)
                    res.extend(i for _, i in grp)
                    grp = []
                grp.append((sc, iid))
            grp.sort(key=gk)
            res.extend(i for _, i in grp)
            out[q] = res[:10]
        return out

    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune, cf = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])

    def halves(run):
        o = []
        for qs in (tune, cf):
            sel = [q for q in h.qids if q in qs and q in qrels]
            o.append([ndcg_at_k(run[q], qrels[q], 10) for q in sel])
        return o

    base = build(LEVELS)
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    same = sum(1 for q in h.qids if live.get(q, [])[:10] == base[q][:10])
    print(f"REGRESSION vs submission_v7synth_stack3.csv: {same:,}/{len(h.qids):,}")
    if same != len(h.qids):
        print("  *** baseline does not reproduce - stop ***")
        return
    b_t, b_c = halves(base)
    print(f"baseline  tune {sum(b_t)/len(b_t):.6f}  confirm {sum(b_c)/len(b_c):.6f}\n")

    def report(label, levels):
        c_t, c_c = halves(build(levels))
        dt = sum(y - x for x, y in zip(b_t, c_t)) / len(b_t)
        dd = [y - x for x, y in zip(b_c, c_c)]
        dc = sum(dd) / len(dd)
        se = st.pstdev(dd) / len(dd) ** 0.5
        t = dc / se if se else 0.0
        flag = "  <-- both halves positive" if (dt > 0 and dc > 0) else ""
        print(f"{label:<34s}{dt:>+12.6f}{dc:>+12.6f}{t:>+7.2f}{flag}", flush=True)
        return dt, dc, t

    print("=== ABLATION: drop levels from the shipped key")
    print(f"{'config':<34s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")
    results = []
    for drop in LEVELS:
        keep = tuple(n for n in LEVELS if n != drop)
        results.append((*report(f"drop {drop}", keep), f"drop {drop}"))
    results.append((*report("id only (drop all four)", ()), "id only"))

    print("\n=== PERMUTATION: reorder all four levels (23 alternatives)")
    print(f"{'config':<34s}{'tune':>12s}{'confirm':>12s}{'t':>7s}")
    for perm in itertools.permutations(LEVELS):
        if perm == LEVELS:
            continue
        results.append((*report(" > ".join(perm), perm), " > ".join(perm)))

    print("\nbest by confirm half (shipped key is the 0.000000 reference):")
    for dt, dc, t, label in sorted(results, key=lambda r: -r[1])[:6]:
        mark = "" if (dt > 0 and dc > 0) else "   [sign flip across halves - noise]"
        print(f"  {label:<32s} tune {dt:+.6f}  confirm {dc:+.6f}  t {t:+.2f}{mark}")


if __name__ == "__main__":
    main()
