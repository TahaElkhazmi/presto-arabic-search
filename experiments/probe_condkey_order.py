"""Is the optimal tie-break key ORDER query-dependent rather than global?

`probe_triage_specificity.py` established that 75.8% of the oracle-inversion swap gain inside
the shipped top-10 is INTRA-chain, so the tie-break key is the component at fault.
`probe_key_structure.py` then established that the GLOBAL order is optimal: all 23 permutations
of (cont3, mp, ft, lt) are null or negative, and all 18 that move cont3 out of first place are
clearly negative. What was never tested is whether one order fits every query.

The concrete reason to think it might not: for a 1-token query, match-position and first-token
position are computed from the SAME token index, so the key is effectively shorter and more
pairs fall through to the id fallback. There is no a priori reason a 1-token query and a
4-token query with digits want the same level order.

METHOD. nDCG@10 is a mean over queries and each query's re-sort depends only on that query, so
a conditional configuration's delta decomposes EXACTLY as the sum over classes of the chosen
order's within-class contribution. One global build per order therefore prices every
conditional configuration over every partition, for free. That buys two things:

  1. the honest protocol - pick the per-class argmax on the TUNE half, read the assembled
     configuration once on CONFIRM;
  2. the CONFIRM-HALF ORACLE - pick the per-class argmax with hindsight on the confirm half
     itself. That is an upper bound no conditional scheme in this family can beat, so if it is
     small the whole family is dead regardless of how the search is run.

PRE-REGISTERED, before looking at any number:
  orders     6 (cont3 held first, all permutations of the trailing three)
  partitions 3 (stripped token count 1/2/3+; digit token present; max-token-IDF tercile)
  => 5 alternatives x 8 classes = 40 cells searched, 3 assembled configurations read on confirm.

Everything is a CPU re-sort over cached scores. The blend is held fixed; only the key changes.
"""

import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_ensemble_current import (B_IDF, BETA_FT, BETA_MP, CONF_T, EPS, EPS_HI, LAM,
                                    RRF_MAX, TOP_EPS, TOP_K, Stack3)
from probe_harness import _id_key

setup_utf8()

SHIPPED = ("cont3", "mp", "ft", "lt")
ORDERS = [
    ("cont3", "mp", "ft", "lt"),   # shipped
    ("cont3", "mp", "lt", "ft"),
    ("cont3", "ft", "mp", "lt"),
    ("cont3", "ft", "lt", "mp"),
    ("cont3", "lt", "mp", "ft"),
    ("cont3", "lt", "ft", "mp"),
]


def tag(o):
    return ">".join(n for n in o[1:])


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

    # blended score and band geometry are identical in every configuration
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

    # ---------- baseline must reproduce the shipped submission ----------
    runs = {tag(SHIPPED): build(SHIPPED)}
    live = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    base_run = runs[tag(SHIPPED)]
    same = sum(1 for q in h.qids if live.get(q, [])[:10] == base_run[q][:10])
    print(f"REGRESSION vs submission_v7synth_stack3.csv: {same:,}/{len(h.qids):,}")
    if same != len(h.qids):
        print("  *** baseline does not reproduce - stop ***")
        return

    for o in ORDERS[1:]:
        print(f"  building {tag(o)} ...", flush=True)
        runs[tag(o)] = build(o)

    # ---------- query classes ----------
    ntok = {q: len(S.qt[q]) for q in h.qids}
    hasdig = {q: any(c.isdigit() for t in S.qt[q] for c in t) for q in h.qids}
    bidf = {q: (S.idf(S.qbest[q]) if S.qbest[q] is not None else 0.0) for q in h.qids}
    vals = sorted(bidf[q] for q in h.qids)
    lo_c, hi_c = vals[len(vals) // 3], vals[2 * len(vals) // 3]

    def cls_ntok(q):
        n = ntok[q]
        return "ntok=1" if n <= 1 else ("ntok=2" if n == 2 else "ntok>=3")

    def cls_dig(q):
        return "digit" if hasdig[q] else "no-digit"

    def cls_idf(q):
        v = bidf[q]
        return "idf-lo" if v < lo_c else ("idf-mid" if v < hi_c else "idf-hi")

    PARTITIONS = [("token count", cls_ntok, ["ntok=1", "ntok=2", "ntok>=3"]),
                  ("digit token", cls_dig, ["digit", "no-digit"]),
                  ("max-token-IDF tercile", cls_idf, ["idf-lo", "idf-mid", "idf-hi"])]

    # ---------- structural check: which classes can a permutation even move? ----------
    print("\n=== STRUCTURAL: queries whose top-10 CHANGES, by class")
    print("(a class no permutation can move is void by construction, not a null result)")
    hdr = f"{'class':<12s}{'queries':>9s}" + "".join(f"{tag(o):>14s}" for o in ORDERS[1:])
    for pname, cf, names in PARTITIONS:
        print(f"\n-- {pname}")
        print(hdr)
        for cn in names:
            qs = [q for q in h.qids if cf(q) == cn]
            row = f"{cn:<12s}{len(qs):>9,}"
            for o in ORDERS[1:]:
                r = runs[tag(o)]
                nch = sum(1 for q in qs if r[q] != base_run[q])
                row += f"{100.0 * nch / max(len(qs), 1):>13.1f}%"
            print(row)

    # ---------- id-fallback diagnostic, per class ----------
    print("\n=== DIAGNOSTIC: share of shipped top-10 slots decided by the id fallback")
    print("(slot sits in a chain where >=2 members share its full 4-level key)")
    for pname, cf, names in PARTITIONS:
        print(f"-- {pname}")
        for cn in names:
            tot = fell = 0
            for q in h.qids:
                if cf(q) != cn:
                    continue
                scored, e0 = order_cache[q]
                chain, cid, prev = {}, 0, None
                for posn, (sc, iid) in enumerate(scored):
                    e = TOP_EPS if posn < TOP_K else e0
                    if prev is not None and (prev - sc) > e:
                        cid += 1
                    chain[iid] = cid
                    prev = sc
                fullkey = {}
                for iid in chain:
                    k = (chain[iid],) + tuple(getters[n](q, iid) for n in SHIPPED)
                    fullkey[k] = fullkey.get(k, 0) + 1
                for iid in base_run[q]:
                    k = (chain.get(iid, -1),) + tuple(getters[n](q, iid) for n in SHIPPED)
                    tot += 1
                    if fullkey.get(k, 0) >= 2:
                        fell += 1
            print(f"  {cn:<12s}{100.0 * fell / max(tot, 1):>7.1f}%  of {tot:,} slots")

    # ---------- tune / confirm halves ----------
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune, cfm = set(shuf[: len(shuf) // 2]), set(shuf[len(shuf) // 2:])
    scored_q = [q for q in h.qids if q in qrels]
    nd = {t: {q: ndcg_at_k(r[q], qrels[q], 10) for q in scored_q} for t, r in runs.items()}
    b = nd[tag(SHIPPED)]
    n_t = sum(1 for q in scored_q if q in tune)
    n_c = sum(1 for q in scored_q if q in cfm)
    print(f"\nbaseline  tune {sum(b[q] for q in scored_q if q in tune)/n_t:.6f}"
          f"  confirm {sum(b[q] for q in scored_q if q in cfm)/n_c:.6f}"
          f"   ({n_t:,} / {n_c:,} judged queries)")

    # contribution of applying order o to class members only, as a delta on the FULL mean
    def contrib(o, qs_class, half, n_half):
        d = [nd[tag(o)][q] - b[q] for q in qs_class if q in half]
        return sum(d) / n_half, len(d)

    print("\n=== PER-CLASS: contribution to the FULL-slice mean of using this order")
    print("    for that class only (shipped elsewhere). tune | confirm.")
    for pname, cf, names in PARTITIONS:
        print(f"\n-- {pname}")
        print(f"{'order for class':<22s}" + "".join(f"{cn:>22s}" for cn in names))
        for o in ORDERS[1:]:
            row = f"{tag(o):<22s}"
            for cn in names:
                qs = [q for q in scored_q if cf(q) == cn]
                dt, _ = contrib(o, qs, tune, n_t)
                dc, _ = contrib(o, qs, cfm, n_c)
                row += f"{dt:>+11.6f}{dc:>+11.6f}"
            print(row, flush=True)

    # ---------- honest protocol + hindsight ceiling ----------
    print("\n=== VERDICT per partition")
    for pname, cf, names in PARTITIONS:
        chosen, oracle_sum, oracle_pick = {}, 0.0, {}
        for cn in names:
            qs = [q for q in scored_q if cf(q) == cn]
            best_t, best_o = 0.0, SHIPPED           # shipped = 0.0 reference, must be beaten
            best_c, best_co = 0.0, SHIPPED
            for o in ORDERS[1:]:
                dt, _ = contrib(o, qs, tune, n_t)
                dc, _ = contrib(o, qs, cfm, n_c)
                if dt > best_t:
                    best_t, best_o = dt, o
                if dc > best_c:
                    best_c, best_co = dc, o
            chosen[cn] = best_o
            oracle_sum += best_c
            oracle_pick[cn] = (tag(best_co), best_c)

        dd = []
        for q in scored_q:
            if q in cfm:
                dd.append(nd[tag(chosen[cf(q)])][q] - b[q])
        dc = sum(dd) / len(dd)
        se = st.pstdev(dd) / len(dd) ** 0.5
        dt_tot = sum(contrib(chosen[cn], [q for q in scored_q if cf(q) == cn], tune, n_t)[0]
                     for cn in names)
        print(f"\n-- {pname}")
        for cn in names:
            print(f"   tune-argmax for {cn:<10s} -> {tag(chosen[cn])}")
        print(f"   assembled: tune {dt_tot:+.6f}   CONFIRM {dc:+.6f}   t {dc/se if se else 0:+.2f}"
              f"   {'both halves positive' if dt_tot > 0 and dc > 0 else 'FAILS both-halves'}")
        print(f"   confirm-half ORACLE (hindsight per class, ceiling for this family): "
              f"{oracle_sum:+.6f}")
        for cn in names:
            t_, v_ = oracle_pick[cn]
            print(f"      hindsight {cn:<10s} -> {t_:<14s} {v_:+.6f}")

    # ---------- maximally generous ceiling: fully crossed partition, hindsight ----------
    # A richer partition can only raise a hindsight ceiling, so if the CROSSED oracle is also
    # small, no conditioner built from these three variables can reach the target. The random
    # -partition null below prices how much of any such oracle is pure selection noise.
    def crossed(q):
        return f"{cls_ntok(q)}|{cls_dig(q)}|{cls_idf(q)}"

    def oracle(labeller, half, n_half):
        cells = {}
        for q in scored_q:
            if q in half:
                cells.setdefault(labeller(q), []).append(q)
        tot = 0.0
        for qs in cells.values():
            best = 0.0
            for o in ORDERS[1:]:
                v = sum(nd[tag(o)][q] - b[q] for q in qs) / n_half
                best = max(best, v)
            tot += best
        return tot, len(cells)

    print("\n=== CEILINGS (hindsight on the confirm half itself) vs a RANDOM-PARTITION NULL")
    print("null = same cell sizes, membership shuffled, 200 draws. A conditioner that carries")
    print("real information must beat its own null; anything at the null is selection noise.")
    print(f"{'partition':<26s}{'cells':>7s}{'oracle':>12s}{'null mean':>12s}"
          f"{'null p95':>12s}{'null max':>12s}")

    cf_qs = [q for q in scored_q if q in cfm]
    rng = random.Random(4242)
    for pname, cf, _n in list(PARTITIONS) + [("crossed (all three)", crossed, None)]:
        orc, ncell = oracle(cf, cfm, n_c)
        sizes = {}
        for q in cf_qs:
            sizes[cf(q)] = sizes.get(cf(q), 0) + 1
        szl = sorted(sizes.values(), reverse=True)
        nulls = []
        for _ in range(200):
            sh = list(cf_qs)
            rng.shuffle(sh)
            lab, pos = {}, 0
            for ci, sz in enumerate(szl):
                for q in sh[pos:pos + sz]:
                    lab[q] = ci
                pos += sz
            nulls.append(oracle(lambda q: lab[q], cfm, n_c)[0])
        nulls.sort()
        print(f"{pname:<26s}{ncell:>7d}{orc:>+12.6f}{sum(nulls)/len(nulls):>+12.6f}"
              f"{nulls[int(0.95 * len(nulls))]:>+12.6f}{nulls[-1]:>+12.6f}", flush=True)

    print("\ncells searched: 5 alternative orders x 8 classes = 40; "
          "3 assembled configurations read on the confirm half.")
    print("Dev-argmax inflation on this project is 3-4x; treat every tune number as an "
          "upper bound.")


if __name__ == "__main__":
    main()
