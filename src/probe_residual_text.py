"""Is the saturated tie group an information VOID, or only a metric blind spot?

The framing under test: "70 identical titles share the exact same string and the exact
same 3-gram containment (1.000), so no text signal can separate them."

The first half needs checking. Our containment key is |query_grams AND name_grams| /
|query_grams|, which saturates at 1.0 the moment a title contains every query gram,
REGARDLESS of what else the title says. So "مولي" and "طعام القطط مولي بالدجاج
والسمك - 2 كغ" both score exactly 1.000 while being completely different strings.
If the tie groups hold many DISTINCT titles, the saturation is a property of our
metric, not an absence of information, and residual-text features are still available.

Measures, inside the maximum-containment tie group of zero-cohort queries:
  1. how many distinct normalized titles the group actually contains;
  2. whether text features our key throws away carry per-query signal, in particular
     Jaccard (which does NOT saturate because it penalises extra tokens), residual
     token count, residual-token IDF, title length, and query-match position.

All AUCs are computed PER QUERY and aggregated, never pooled across queries; the two
provably identical orderings included as controls must coincide exactly.
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
from stacker_features import char_ngrams

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


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
    run = load_run_csv(ROOT / "runs" / "submission_v7synth_margin3.csv")
    nd = {q: ndcg_at_k(ids, qrels[q], 10) for q, ids in run.items() if q in qrels}
    import random
    rng = random.Random(41)
    zeros = sorted(rng.sample([q for q, v in nd.items() if v == 0.0], 1500), key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names = {str(r["item_id"]): r["name"] for r in corpus}

    # token IDF over unique normalized titles
    uniq = {normalize_arabic(n) for n in names.values()}
    df = Counter()
    for t in uniq:
        for tok in set(t.split()):
            df[tok] += 1
    N = len(uniq)

    def idf(tok):
        return math.log(1.0 + N / (df.get(tok, 0) + 0.5))

    print("building pools (0 GPU) ...", flush=True)
    h = Harness(zeros, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0

    groups = {}
    for q in zeros:
        pool = h.pools[q]
        cs = [h.cont3(q, i) for i in pool]
        mx = max(cs)
        g = [i for i, c in zip(pool, cs) if c >= mx - 1e-12]
        if len(g) >= 2 and any(qrels[q].get(i, 0) > 0 for i in g) \
           and any(qrels[q].get(i, 0) <= 0 for i in g):
            groups[q] = g

    # ---- 1. how identical are they really? ----
    sizes, distinct, frac = [], [], []
    for q, g in groups.items():
        nm = {normalize_arabic(names[i]) for i in g}
        sizes.append(len(g))
        distinct.append(len(nm))
        frac.append(len(nm) / len(g))
    print(f"\n{len(groups)} saturated tie groups on the zero cohort")
    print(f"  mean group size          : {st.mean(sizes):.1f}")
    print(f"  mean DISTINCT titles     : {st.mean(distinct):.1f}")
    print(f"  distinct share of group  : {100 * st.mean(frac):.1f}%")
    allsame = sum(1 for d in distinct if d == 1)
    print(f"  groups that are genuinely ONE repeated string: {allsame}/{len(groups)} "
          f"({100 * allsame / len(groups):.1f}%)")

    qtok = {q: [t for t in normalize_arabic(h.queries[q]).split()
                if t not in DIALECT_FILLERS] for q in groups}

    def resid(q, iid):
        return [t for t in normalize_arabic(names[iid]).split() if t not in set(qtok[q])]

    feats = {
        "low-id (control)": lambda q, i: -num(i),
        "rank-by-id (control)": None,  # filled below, must equal low-id
        "our CE": lambda q, i: h.ce[(q, i)],
        "3gram JACCARD (non-sat)": lambda q, i: (
            len(char_ngrams(" ".join(qtok[q]), 3) & char_ngrams(normalize_arabic(names[i]), 3))
            / max(len(char_ngrams(" ".join(qtok[q]), 3) | char_ngrams(normalize_arabic(names[i]), 3)), 1)),
        "residual token count": lambda q, i: -len(resid(q, i)),
        "residual IDF sum": lambda q, i: -sum(idf(t) for t in resid(q, i)),
        "residual IDF mean": lambda q, i: -(
            st.mean([idf(t) for t in resid(q, i)]) if resid(q, i) else 0.0),
        "title token count": lambda q, i: -len(normalize_arabic(names[i]).split()),
        "title char length": lambda q, i: -len(normalize_arabic(names[i])),
        "exact string match": lambda q, i: float(
            normalize_arabic(names[i]) == " ".join(qtok[q])),
        "query at title start": lambda q, i: float(
            normalize_arabic(names[i]).startswith(" ".join(qtok[q]))),
    }

    print("\nPER-QUERY AUC INSIDE THE SATURATED TIE GROUP (zero cohort)")
    print("0.5 = worthless; the incumbent low-id is 0.5897 and the CE 0.5976\n")
    for name, fn in feats.items():
        if fn is None:
            fn = lambda q, i: -sorted(groups[q], key=num).index(i)
        pairs = []
        for q, g in groups.items():
            p = [fn(q, i) for i in g if qrels[q].get(i, 0) > 0]
            n = [fn(q, i) for i in g if qrels[q].get(i, 0) <= 0]
            pairs.append((p, n))
        a = per_query_auc(pairs)
        mark = ""
        if a == a:
            if a >= 0.62:
                mark = "  <-- BEATS the incumbent"
            elif a <= 0.38:
                mark = "  <-- strongly inverted (usable flipped)"
        print(f"  {name:<26s} AUC {a:.4f}{mark}")

    print("\nREADING: if 'distinct titles' is well above 1, the group is NOT a set of")
    print("identical strings and containment saturation is a limitation of our KEY,")
    print("not an information void. A residual-text feature beating 0.5897 would then")
    print("be a genuinely new signal available at zero GPU cost.")


if __name__ == "__main__":
    main()
