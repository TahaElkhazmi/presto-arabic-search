"""Reconcile the match-position variants: which definition actually wins?

A subagent reported that ordering saturated ties by query match POSITION beats the
shipped binary prefix test, quoting 0.683838 full-benchmark. Reimplementing it through
the production path gave 0.674186, which is WORSE than the shipped prefix (0.680892),
and prefix is only the p=0 special case of position, so one of the two numbers rests on
a different definition. This settles it by evaluating every plausible definition on the
full benchmark against the same baseline.

Definitions differ in what string is searched and how the offset is scored:
  raw        - find the stripped query in normalize_arabic(name)
  joined     - collapse whitespace on both sides first ("token-joined")
  tokenpos   - offset measured in TOKENS rather than characters
  binary     - the shipped prefix test, p == 0
and in whether an absent match sorts below everything (posf=0) or is treated as a
plain tie.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")


def main():
    all_qids = [str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")]
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    joined = {i: " ".join(n.split()) for i, n in norm.items()}
    ntok = {i: n.split() for i, n in norm.items()}

    print(f"harness over {len(all_qids):,} queries (0 GPU) ...", flush=True)
    h = Harness(all_qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qs_, qt_ = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split()
             if x not in DIALECT_FILLERS]
        qt_[q] = t or normalize_arabic(h.queries[q]).split()
        qs_[q] = " ".join(qt_[q])

    def sc(run):
        v = [ndcg_at_k(run[q], qrels[q], 10) for q in all_qids if q in qrels]
        return sum(v) / len(v)

    base = sc(h.build_run(**LIVE))
    print(f"baseline (cont3 + id, the pre-prefix live file): {base:.6f}\n")

    def raw_pos(q, i):
        p = norm[i].find(qs_[q])
        return 0.0 if p < 0 else 1.0 / (1.0 + p)

    def joined_pos(q, i):
        p = joined[i].find(qs_[q])
        return 0.0 if p < 0 else 1.0 / (1.0 + p)

    def token_pos(q, i):
        toks = ntok[i]
        n = len(qt_[q])
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qt_[q]:
                return 1.0 / (1.0 + k)
        return 0.0

    def binary_prefix(q, i):
        return 1.0 if norm[i].startswith(qs_[q]) else 0.0

    def joined_prefix(q, i):
        return 1.0 if joined[i].startswith(qs_[q]) else 0.0

    variants = {
        "binary prefix (SHIPPED)": binary_prefix,
        "binary prefix, joined": joined_prefix,
        "char position, raw": raw_pos,
        "char position, joined": joined_pos,
        "token position": token_pos,
    }

    print(f"  {'variant':<28s}{'nDCG@10':>11s}{'delta':>11s}")
    best = (None, -1)
    for name, fn in variants.items():
        cfg = dict(LIVE)
        cfg["key"] = (lambda f: lambda q, i: h.cont3(q, i) * 1000.0 + f(q, i))(fn)
        s = sc(h.build_run(**cfg))
        print(f"  {name:<28s}{s:>11.6f}{s - base:>+11.6f}")
        if s > best[1]:
            best = (name, s)
    print(f"\n  best: {best[0]} at {best[1]:.6f}")


if __name__ == "__main__":
    main()
