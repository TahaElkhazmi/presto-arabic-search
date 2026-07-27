"""Is there a SECOND binary predicate below the prefix test?

The shipped key is now (containment desc, starts-with-query desc, id asc) and it is
worth +0.0110 on the public leaderboard. But inside each prefix partition we still
fall through to raw ascending id, which is the same failure shape the prefix test just
fixed one level up: a decision made with a signal that has stopped discriminating.

So this measures a FAMILY of binary structural predicates at the level below.
Binary matters: it partitions and lets the low-id prior keep ordering each part, which
is exactly why the prefix test succeeded where the continuous GATE-AraBert score
(0.5637, worse than the incumbent) cost -0.0306 at the same injection point.

The decisive column is CONDITIONAL AUC: among items that already agree on containment
AND on the prefix test, does the predicate separate relevant from non-relevant? A
predicate that only looks good unconditionally is probably just restating the prefix.

Per-query AUC throughout, never pooled. Zero GPU.
"""

import bisect
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_run_csv, normalize_arabic, setup_utf8
from eval_ndcg import load_candidates, ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def per_query_auc(pairs):
    conc = disc = ties = 0
    for pos, neg in pairs:
        if not pos or not neg:
            continue
        ns = sorted(neg)
        for p in pos:
            lo, hi = bisect.bisect_left(ns, p), bisect.bisect_right(ns, p)
            conc += lo
            ties += hi - lo
            disc += len(ns) - hi
    tot = conc + disc + ties
    return (conc + 0.5 * ties) / tot if tot else float("nan")


def main():
    qrels = load_candidates(BENCH / "candidates.jsonl")
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_prefix.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    import random
    rng = random.Random(53)
    zeros = sorted(rng.sample([q for q, v in nd.items() if v == 0.0], 1500), key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names = {str(r["item_id"]): r["name"] for r in corpus}
    norm = {i: normalize_arabic(n) for i, n in names.items()}
    toks = {i: n.split() for i, n in norm.items()}

    uniq = set(norm.values())
    df = Counter()
    for t in uniq:
        for tok in set(t.split()):
            df[tok] += 1
    N = len(uniq)

    def idf(t):
        return math.log(1.0 + N / (df.get(t, 0) + 0.5))

    print("building pools (0 GPU) ...", flush=True)
    h = Harness(zeros, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    qs_, qt_ = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt_[q] = t or normalize_arabic(h.queries[q]).split()
        qs_[q] = " ".join(qt_[q])

    def prefix(q, i):
        return 1.0 if norm[i].startswith(qs_[q]) else 0.0

    preds = {
        "ends with query": lambda q, i: float(norm[i].endswith(qs_[q])),
        "query tokens contiguous": lambda q, i: float(
            " " + qs_[q] + " " in " " + norm[i] + " "),
        "first token matches": lambda q, i: float(
            bool(toks[i]) and bool(qt_[q]) and toks[i][0] == qt_[q][0]),
        "all query tokens present": lambda q, i: float(
            set(qt_[q]) <= set(toks[i])),
        "<=2 extra tokens": lambda q, i: float(
            len(toks[i]) - len(qt_[q]) <= 2),
        "<=4 extra tokens": lambda q, i: float(
            len(toks[i]) - len(qt_[q]) <= 4),
        "no rare residual token": lambda q, i: float(
            all(idf(t) < 8.0 for t in toks[i] if t not in set(qt_[q]))),
        "title has a digit": lambda q, i: float(any(c.isdigit() for c in norm[i])),
        "query-has-digit match": lambda q, i: float(
            any(c.isdigit() for c in qs_[q]) == any(c.isdigit() for c in norm[i])),
    }

    groups = {}
    for q in zeros:
        pool = h.pools[q]
        cs = [h.cont3(q, i) for i in pool]
        mx = max(cs)
        g = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
        if len(g) >= 2 and any(qrels[q].get(i, 0) > 0 for i in g) \
           and any(qrels[q].get(i, 0) <= 0 for i in g):
            groups[q] = g
    print(f"{len(groups)} saturated tie groups\n")

    def score(fn, conditional):
        pairs = []
        for q, g in groups.items():
            subs = ([[i for i in g if prefix(q, i) == v] for v in (1.0, 0.0)]
                    if conditional else [g])
            for sub in subs:
                if len(sub) < 2:
                    continue
                p = [fn(q, i) for i in sub if qrels[q].get(i, 0) > 0]
                n = [fn(q, i) for i in sub if qrels[q].get(i, 0) <= 0]
                if p and n:
                    pairs.append((p, n))
        return per_query_auc(pairs)

    print(f"  {'predicate':<28s}{'uncond':>9s}{'COND on prefix':>16s}")
    print(f"  {'-' * 53}")
    print(f"  {'starts with query (shipped)':<28s}{score(prefix, False):>9.4f}"
          f"{'--':>16s}")
    print(f"  {'low-id (incumbent below)':<28s}"
          f"{score(lambda q, i: -num(i), False):>9.4f}"
          f"{score(lambda q, i: -num(i), True):>16.4f}")
    print(f"  {'-' * 53}")
    rows = []
    for name, fn in preds.items():
        u, c = score(fn, False), score(fn, True)
        rows.append((c, name, u))
        print(f"  {name:<28s}{u:>9.4f}{c:>16.4f}")

    rows.sort(reverse=True)
    print(f"\n  best conditional predicate: {rows[0][1]} at {rows[0][0]:.4f}")
    print("  it must beat the low-id conditional figure above to be worth testing")
    print("  end to end; a predicate merely restating the prefix will look strong")
    print("  unconditionally and collapse in the conditional column.")


if __name__ == "__main__":
    main()
