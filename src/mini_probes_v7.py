"""Final micro-probe batch under the v7synth-recal stack (all cache-only).

Re-measures the Play-5 sub-gate positives through the CORRECT lens (they were all
judged under v6warm's wide bands), plus four never-tested micro-mechanisms:
  a. cont3idf in-band key
  b. filler-routed lambda (filler-bearing queries get their own lam)
  c. global low-id bonus
  d. confidence-margin DEFINITION (top1-top2 / top1-top3 / top1-top10 vs the fixed
     top1-top5), with conf_t re-centered per definition
  e. multi-scale containment key (mean of cont2/cont3/cont4)
  f. definite-article stripping in the cont3 query grams
  g. v6warm score as tertiary in-band tie-key (cont3 desc, v6 desc, id asc)
Split: dev_final2000 (shared with the prior grid; any winner must clear val + the
untouched remainder before it counts). Bar = submitted config on this split.
"""

import math

import pandas as pd

from common import DATA, ROOT, load_qid_file, normalize_arabic, setup_utf8
from grid_search_prior import set_prior
from probe_harness import DIALECT_FILLERS, Harness, rescale
from stacker_features import char_ngrams, containment

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
V6 = "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16"
STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.15, key="cont3")


def main():
    qids = sorted(load_qid_file(ROOT / "splits" / "dev_final2000.txt"))
    h = Harness(qids, T=2.6, cache_stem=V7)
    set_prior(h, 60, (3.0, 1.0))  # the LIVE composition (LB 0.66738), not the old k=10 2:1
    base = h.evaluate(**STACK)
    print(f"BASELINE (live config k=60 3:1): {base:.4f}\n", flush=True)

    def log(name, s):
        print(f"  {name:46s}: {s:.4f}  ({s - base:+.4f})", flush=True)

    # a. cont3idf key
    log("a: key=cont3idf", h.evaluate(**{**STACK, "key": "cont3idf"}))

    # b. filler-routed lambda
    feat = {q: any(t in DIALECT_FILLERS for t in normalize_arabic(h.queries[q]).split())
            for q in qids}
    for lf in (0.40, 0.50, 0.70):
        cfg = {k: v for k, v in STACK.items() if k != "lam"}
        s = h.evaluate(lam_fn=lambda q, lf=lf: lf if feat[q] else 0.60, **cfg)
        log(f"b: filler lam={lf} (rest 0.60)", s)

    # c. id bonus
    for bta in (0.003, 0.01, 0.03):
        log(f"c: id_bonus={bta}", h.evaluate(**STACK, id_bonus=bta))

    # d. confidence-margin definition (recompute h.conf, restore after)
    conf5 = dict(h.conf)
    for topk in (2, 3, 10):
        for qid in qids:
            v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
            h.conf[qid] = v[0] - v[topk - 1] if len(v) >= topk else 1.0
        for ct in (0.05, 0.10, 0.15, 0.20):
            s = h.evaluate(**{**STACK, "conf_t": ct})
            log(f"d: margin=top1-top{topk}, conf_t={ct}", s)
    h.conf = conf5

    # e. multi-scale containment key
    g_cache: dict[tuple[str, int], frozenset] = {}

    def grams(text, n):
        k = (text, n)
        if k not in g_cache:
            g_cache[k] = char_ngrams(text, n)
        return g_cache[k]

    q_core = {}
    for qid in qids:
        qn = normalize_arabic(h.queries[qid])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        q_core[qid] = " ".join(kept) if kept else qn

    def cont_multi(qid, iid):
        nn = normalize_arabic(h.names[iid])
        return sum(containment(grams(q_core[qid], n), grams(nn, n)) for n in (2, 3, 4)) / 3

    log("e: key=mean(cont2,cont3,cont4)", h.evaluate(**{**STACK, "key": cont_multi}))

    # f. definite-article stripping in the cont3 key
    def strip_al(tok):
        return tok[2:] if tok.startswith("ال") and len(tok) > 4 else tok

    q_al = {qid: " ".join(strip_al(t) for t in q_core[qid].split()) for qid in qids}

    def cont3_al(qid, iid):
        nn = " ".join(strip_al(t) for t in normalize_arabic(h.names[iid]).split())
        return containment(grams(q_al[qid], 3), grams(nn, 3))

    log("f: key=cont3 with al-stripping (both sides)", h.evaluate(**{**STACK, "key": cont3_al}))

    # g. v6warm as tertiary in-band key
    df6 = pd.read_parquet(DATA / "cache" / f"{V6}.parquet")
    df6 = df6[df6["qid"].isin(set(qids))]
    v6 = {(q, i): rescale(s, 2.3) for q, i, s in zip(df6["qid"], df6["iid"], df6["score"])}
    del df6

    def cont3_v6(qid, iid):
        # combined float: cont3 dominates, v6 breaks cont3 ties (cont3 quantum ~0.02)
        return h.cont3(qid, iid) + 0.001 * v6.get((qid, iid), 0.0)

    log("g: key=cont3 then v6warm then id", h.evaluate(**{**STACK, "key": cont3_v6}))

    print("\ndone; any winner >= +0.0015 needs val + untouched confirmation.")


if __name__ == "__main__":
    main()
