"""Screen the two external suggestions that survive the dead list, at the decision cell.

IDEA 1, cell-size routing: branch the band geometry on how MANY candidates share the
decision cell, rather than on a score margin. Every band variant tested so far conditioned
on score geometry (top1-top3, top1-top5, pool mean, relative margin, three tiers), so
ambiguity SIZE is a genuinely different conditioning variable. What we measure here is its
coverage: the share of shipped top-10 slots sitting in a cell big enough to route on.

IDEA 2, id NEIGHBOURHOOD geometry: the proposal was to cluster ids and sort clusters, but
that is a no-op - clustering by id, ordering clusters by their minimum id and sorting by id
within each cluster reproduces plain ascending id exactly. The non-void version scores a
candidate by a property of its LOCAL id spacing, which is information ascending id does not
use. Large gaps plausibly mark catalog insertion epochs, and items imported together are
plausibly duplicate uploads of one product.

Both are screened where it actually matters: the cell where containment, match-position,
first-token and last-token all tie and ascending id currently decides. Ten features have
lost there, and a global AUC would be the fourth estimator trap all over again.

Target is positive-versus-negative (the shipping target). AUC is computed within each query
and averaged, never pooled.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, load_run_csv, setup_utf8
from eval_ndcg import load_candidates
from probe_ensemble_current import Stack3

setup_utf8()


def idnum(i):
    d = "".join(c for c in i if c.isdigit())
    return int(d) if d else 0


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_stack3.csv")
    print(f"building harness over {len(hold):,} holdout queries ...", flush=True)
    S = Stack3(hold)
    h = S.h

    # ---------- IDEA 1: how big are the decision cells, over shipped top-10 slots ----------
    sizes = Counter()
    slots = 0
    for q in hold:
        top = [i for i in run.get(q, [])[:10] if (q, i) in h.ce]
        key = lambda i: (h.cont3(q, i), S.mp(q, i), S.ft(q, i), S.lt(q, i))
        # cell = candidates ANYWHERE in the pool sharing this slot's key
        pool_keys = {}
        for i in h.pools[q]:
            pool_keys.setdefault(key(i), []).append(i)
        for i in top:
            slots += 1
            sizes[len(pool_keys[key(i)])] += 1

    print(f"\n=== IDEA 1: decision-cell size over {slots:,} shipped top-10 slots")
    cum = 0
    for s in sorted(sizes):
        cum += sizes[s]
        if s <= 6 or s in (10, 20, 50):
            print(f"  cell size {s:>4d}: {sizes[s]:>7,} ({100*sizes[s]/slots:5.2f}%)"
                  f"  cumulative {100*cum/slots:6.2f}%")
    big = sum(v for k, v in sizes.items() if k >= 4)
    solo = sizes.get(1, 0)
    print(f"  slots in a cell of size 1 (nothing to route): {100*solo/slots:.2f}%")
    print(f"  slots in a cell of size >= 4: {100*big/slots:.2f}%  "
          f"(the proposer's own bar was 10%)")

    # ---------- IDEA 2: id-neighbourhood features AT the decision cell ----------
    feats = {
        "low-id (incumbent)": lambda c, n, i: -idnum(i),
        "gap to previous id in cell": lambda c, n, i: -_gap_prev(c, i),
        "gap to next id in cell": lambda c, n, i: -_gap_next(c, i),
        "local block size": lambda c, n, i: _block_size(c, i),
        "is block start": lambda c, n, i: 1.0 if _is_block_start(c, i) else 0.0,
    }
    pairs = {k: [] for k in feats}
    ncell = nq = 0
    for q in hold:
        rel = qrels.get(q)
        if not rel:
            continue
        key = lambda i: (h.cont3(q, i), S.mp(q, i), S.ft(q, i), S.lt(q, i))
        cells = {}
        for i in h.pools[q]:
            cells.setdefault(key(i), []).append(i)
        used = False
        for _, members in cells.items():
            if len(members) < 2:
                continue
            ids = sorted(members, key=idnum)
            pos = [i for i in ids if rel.get(i, 0) > 0]
            neg = [i for i in ids if rel.get(i, 0) <= 0 and i in rel]
            if not pos or not neg:
                continue
            used = True
            for a in pos:
                for b in neg:
                    ncell += 1
                    for k, f in feats.items():
                        pairs[k].append((f(ids, None, a), f(ids, None, b)))
        nq += 1 if used else 0

    print(f"\n=== IDEA 2: id-neighbourhood features AT THE DECISION CELL")
    print(f"(positive, negative) pairs inside a shared cell: {ncell:,} over {nq:,} queries")
    if ncell:
        print(f"{'feature':<30s}{'AUC':>8s}{'ties %':>9s}")
        rows = []
        for k, p in pairs.items():
            w = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a, b in p) / len(p)
            ties = 100.0 * sum(1 for a, b in p if a == b) / len(p)
            rows.append((w, ties, k))
        for w, ties, k in sorted(rows, reverse=True):
            print(f"{k:<30s}{w:>8.4f}{ties:>9.1f}")
        base = [w for w, _, k in rows if k.startswith("low-id")][0]
        print(f"\nIncumbent low-id AUC at the cell: {base:.4f}")
        print("A challenger must beat that AND have low ties (coverage) to be worth a run.")


def _idxs(cell):
    # NOT cached on id(cell): CPython recycles object ids after collection, so an
    # id-keyed cache silently returns another cell's list.
    return [idnum(x) for x in cell]


def _gap_prev(cell, i):
    ids = _idxs(cell)
    p = ids.index(idnum(i))
    return ids[p] - ids[p - 1] if p > 0 else 10 ** 6


def _gap_next(cell, i):
    ids = _idxs(cell)
    p = ids.index(idnum(i))
    return ids[p + 1] - ids[p] if p < len(ids) - 1 else 10 ** 6


def _blocks(cell, thresh=10):
    ids = _idxs(cell)
    out, cur = [], [ids[0]]
    for a, b in zip(ids, ids[1:]):
        if b - a <= thresh:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def _block_size(cell, i):
    v = idnum(i)
    for b in _blocks(cell):
        if v in b:
            return float(len(b))
    return 1.0


def _is_block_start(cell, i):
    v = idnum(i)
    return any(b[0] == v for b in _blocks(cell))


if __name__ == "__main__":
    main()
