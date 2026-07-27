"""Score the Gemini judge orderings, directly comparable to the Qwen Kaggle results.

The Kaggle probe answered "can Qwen do this": no, at both 4B (tie-group -0.0049) and
14B (tie-group -0.0396). This asks the same question of a different model family, on
the same band queries, through the same policies, so the numbers sit on one scale.

Letters A..T were assigned in LIVE rank order, so letter i corresponds to our own
rank i+1. That means the identity permutation A,B,C,... is exactly our current
ordering, and any deviation is the judge expressing an opinion.
"""

import argparse
import json
import re
import statistics as st
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ROOT, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE_STACK = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
BAND_FRAC = 3308 / 8012
LETTERS = string.ascii_uppercase


def load_orders(paths):
    out = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{") or '"qid"' not in line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            letters = [x.strip() for x in str(o["order"]).split(",") if x.strip()]
            idx, seen = [], set()
            for ch in letters:
                if ch in LETTERS and ch not in seen:
                    seen.add(ch)
                    idx.append(LETTERS.index(ch))
            if idx:
                out[str(o["qid"])] = idx
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    args = p.parse_args()

    judge_idx = load_orders(args.files)
    print(f"parsed {len(judge_idx)} judge orderings\n")
    qids = sorted(judge_idx, key=int)

    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    shown = h.build_run(topn=20, **LIVE_STACK)
    base_run = h.build_run(**LIVE_STACK)
    qrels = h.qrels

    common = [q for q in qids if q in qrels and len(shown.get(q, [])) >= 20]
    base = {q: ndcg_at_k(base_run[q], qrels[q], 10) for q in common}

    def ceil20(q):
        rel = sorted(((qrels[q].get(i, 0), i) for i in shown[q]), key=lambda x: -x[0])
        return ndcg_at_k([i for g, i in rel[:10] if g > 0], qrels[q], 10)

    head = sum(ceil20(q) - base[q] for q in common) / len(common)
    print(f"n={len(common)}   live baseline {sum(base.values()) / len(common):.6f}   "
          f"reachable headroom {head:+.4f}\n")

    judge = {q: [shown[q][i] for i in judge_idx[q] if i < len(shown[q])] for q in common}

    def tie_groups(cont, tol=1e-9):
        groups, cur = [], [0]
        for i in range(1, len(cont)):
            (cur if abs(cont[i] - cont[i - 1]) <= tol else None)
            if abs(cont[i] - cont[i - 1]) <= tol:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        return groups

    def tiegroup(q):
        sh = shown[q]
        cont = [h.cont3(q, i) for i in sh]
        rank = {iid: r for r, iid in enumerate(judge[q])}
        out = list(sh)
        for g in tie_groups(cont):
            if len(g) < 2:
                continue
            items = sorted([sh[i] for i in g], key=lambda x: rank.get(x, 10**6))
            for pos, i in zip(g, items):
                out[pos] = i
        return out[:10]

    def prefix(m):
        def f(q):
            pre = judge[q][:m]
            rest = [i for i in shown[q] if i not in set(pre)]
            return (pre + rest)[:10]
        return f

    print(f"  {'policy':<14s}{'delta':>11s}{'se':>9s}{'t':>7s}{'capture':>9s}"
          f"{'proj':>10s}{'w/l':>9s}")
    for name, fn in (("tie-group", tiegroup), ("prefix-1", prefix(1)),
                     ("prefix-3", prefix(3)), ("prefix-5", prefix(5)),
                     ("full-10", lambda q: judge[q][:10])):
        d = [ndcg_at_k(fn(q), qrels[q], 10) - base[q] for q in common]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        w = sum(1 for x in d if x > 1e-9)
        l = sum(1 for x in d if x < -1e-9)
        print(f"  {name:<14s}{m:>+11.6f}{se:>9.6f}{m / se if se else 0:>+7.2f}"
              f"{100 * m / head if head else 0:>8.1f}%{m * BAND_FRAC:>+10.6f}"
              f"{f'{w}/{l}':>9s}")

    ident = sum(1 for q in common if judge_idx[q][:10] == list(range(10)))
    echo = sum(1 for q in common if judge_idx[q][0] == 0)
    print(f"\n  returned our exact top-10: {ident}/{len(common)} "
          f"({100 * ident / len(common):.0f}%)")
    print(f"  kept our top-1 first     : {echo}/{len(common)} "
          f"({100 * echo / len(common):.0f}%)")


if __name__ == "__main__":
    main()
