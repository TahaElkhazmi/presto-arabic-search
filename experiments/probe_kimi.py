"""External-model proposals, triaged and measured.

Seven mechanisms were proposed by an external model. Three are excluded before testing,
with reasons, and four are measured here.

EXCLUDED:
  #1 position-dependent eps - already measured. The proposal is TIGHTER eps at the top to
     protect the cross-encoder; probe_tier2 measured that at -0.0054 tune / -0.0064 confirm.
     The opposite (eps 0.080 over ranks 1-5) measured +0.0013/+0.0012 and is already SHIPPED
     inside stack3. The premise that the CE deserves protection at the top is false here.
  #4 dropping a redundant level on 1-token queries - the observation is right (match_position
     and first_token_position are perfectly correlated when the query is one token) but the
     conclusion does not follow: a perfectly correlated level partitions identically, so
     removing it cannot change any ordering. Only its eps half is testable, and that is
     included below as D.
  #6 duplicate-sibling CE transfer - void by construction. The CE scores (query, name), so
     exact duplicate names receive IDENTICAL scores and a sibling's maximum is the
     candidate's own score. Verified numerically below rather than asserted.

MEASURED HERE:
  A  hierarchical multi-scale chaining - nested 0.010 / 0.030 / 0.050 groups so that a
     0.005 CE gap and a 0.028 CE gap are not treated as equally uninformative.
  B  retriever provenance - discrete 3-way partition, both retrievers > bm25-only >
     dense-only. Caveat to keep in mind: the RRF prior already rewards appearing in both
     lists, so this may be consumed upstream, which is exactly the trap the Meilisearch
     membership idea fell into.
  C  CE local-maximum protection - pin candidates whose CE exceeds both neighbours by delta.
  D  query-length-adaptive eps.
  E  RRF-rank coarse buckets as an in-group key level.
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
from probe_harness import DIALECT_FILLERS, Harness, _id_key, stream_topk

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LAM, EPS, EPS_HI, CONF_T = 0.50, 0.030, 0.050, 0.10
T_LIVE, BETA_MP, BETA_FT, B_IDF = 3.2, 0.35, 0.05, 0.05
TOP_EPS, TOP_K = 0.080, 5
S_CONT, S_MP, S_FT, S_LT, S_6 = 1e6, 1e2, 1e-2, 1e-5, 1e-8


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf_h = sorted(shuf[len(shuf) // 2:], key=int)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    norm = {str(r["item_id"]): normalize_arabic(r["name"]) for r in corpus}
    ntok = {i: n.split() for i, n in norm.items()}
    uniq = set(norm.values())
    dfm = {}
    for n in uniq:
        for t in set(n.split()):
            dfm[t] = dfm.get(t, 0) + 1
    NU = len(uniq)
    idfv = lambda t: math.log(1.0 + NU / (dfm.get(t, 0) + 0.5))

    print("building harness ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    # ---- #6 void-by-construction check, measured rather than asserted ----
    bm = stream_topk(ROOT / "runs" / "bm25_all.jsonl", 100, set(hold))
    dn = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", 100, set(hold))
    diff = same = 0
    for q in hold[:2000]:
        by = {}
        for i in h.pools[q]:
            by.setdefault(norm[i], []).append(h.ce[(q, i)])
        for v in by.values():
            if len(v) > 1:
                if max(v) - min(v) > 1e-9:
                    diff += 1
                else:
                    same += 1
    tot = diff + same
    print(f"#6 check: duplicate-name groups with IDENTICAL cross-encoder scores: "
          f"{same:,}/{tot:,} ({same/max(tot,1):.1%})")
    print("   -> sibling score transfer has nothing to transfer where scores are identical\n")

    qt, qbest = {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        qbest[q] = max(t, key=idfv) if t else None
    C = {}

    def memo(tag, fn):
        def g(q, i):
            k = (tag, q, i)
            v = C.get(k)
            if v is None:
                v = C[k] = fn(q, i)
            return v
        return g

    def _mp(q, i):
        toks, qq = ntok[i], qt[q]
        n = len(qq)
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    def _ft(q, i):
        toks, qq = ntok[i], qt[q]
        return 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0

    def _xi(q, i):
        toks, b = ntok[i], qbest[q]
        return 3.0 / (1.0 + toks.index(b)) if (b is not None and b in toks) else 0.0

    def _lt(q, i):
        toks, last = ntok[i], (qt[q][-1] if qt[q] else None)
        if last is None or last not in toks:
            return 0.0
        k = toks.index(last)
        return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

    mp, ft, xi, lt = memo("mp", _mp), memo("ft", _ft), memo("xi", _xi), memo("lt", _lt)
    rrf_max = 4.0 / 61.0

    # B: provenance, 2 = both retrievers, 1 = bm25 only, 0 = dense only
    PROV = {}
    for q in h.qids:
        sb, sd = set(bm.get(q, [])), set(dn.get(q, []))
        PROV[q] = {i: (2.0 if (i in sb and i in sd) else (1.0 if i in sb else 0.0))
                   for i in h.pools[q]}
    # E: RRF rank buckets
    RRFB = {}
    for q in h.qids:
        order = sorted(h.pools[q], key=lambda i: -h.priors[q][i])
        RRFB[q] = {i: (3.0 if k < 10 else (2.0 if k < 30 else (1.0 if k < 100 else 0.0)))
                   for k, i in enumerate(order)}

    def base_key(q, i):
        return (h.cont3(q, i) * S_CONT + mp(q, i) * S_MP + ft(q, i) * S_FT
                + lt(q, i) * S_LT)

    def scored_for(q):
        s = [((LAM * h.ce[(q, i)] + (1 - LAM) * h.priors[q][i] / rrf_max
               + BETA_MP * mp(q, i) + BETA_FT * ft(q, i) + B_IDF * xi(q, i)), i)
             for i in h.pools[q]]
        s.sort(key=lambda x: (-x[0], _id_key(x[1])))
        return s

    def chain(seq, eps_fn):
        out, grp = [], []
        for posn, (sc_, iid) in enumerate(seq):
            if grp and (grp[-1][0] - sc_) > eps_fn(posn):
                out.append(grp)
                grp = []
            grp.append((sc_, iid))
        if grp:
            out.append(grp)
        return out

    def build(qids, mode="base", qlen_eps=False):
        out = {}
        for q in qids:
            s = scored_for(q)
            e0 = EPS_HI if h.conf[q] < CONF_T else EPS
            if qlen_eps:
                n = len(qt[q])
                e0 = 0.050 if n == 1 else (0.030 if n == 2 else 0.020)
            eps_fn = lambda p: TOP_EPS if p < TOP_K else e0

            if mode == "prov":
                gk = lambda x, qq=q: (-PROV[qq][x[1]], -base_key(qq, x[1]), _id_key(x[1]))
            elif mode == "rrfb":
                gk = lambda x, qq=q: (-RRFB[qq][x[1]], -base_key(qq, x[1]), _id_key(x[1]))
            elif mode == "provlast":
                gk = lambda x, qq=q: (-base_key(qq, x[1]), -PROV[qq][x[1]], _id_key(x[1]))
            else:
                gk = lambda x, qq=q: (-base_key(qq, x[1]), _id_key(x[1]))

            if mode == "peak":
                ce = [h.ce[(q, i)] for _, i in s]
                pk = []
                for k in range(len(s)):
                    l = ce[k - 1] - ce[k] if k > 0 else 1e9
                    r = ce[k] - ce[k + 1] if k < len(s) - 1 else 1e9
                    pk.append(1.0 if (l > 0.015 and r > 0.015) else 0.0)
                peak = {s[k][1]: pk[k] for k in range(len(s))}
                gk = lambda x, qq=q, pm=peak: (-pm[x[1]], -base_key(qq, x[1]), _id_key(x[1]))

            res = []
            if mode == "hier":
                # nested: loose 0.050 -> mid 0.030 -> micro 0.010 (id only inside micro)
                for L in chain(s, lambda p: 0.050):
                    mid_out = []
                    for M in chain(L, lambda p: 0.030):
                        micro = []
                        for U in chain(M, lambda p: 0.010):
                            micro.extend(sorted(U, key=lambda x: _id_key(x[1])))
                        mid_out.extend(sorted(micro, key=gk))
                    res.extend(mid_out)
            else:
                for g in chain(s, eps_fn):
                    res.extend(sorted(g, key=gk))
            out[q] = [i for _, i in res][:10]
        return out

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt_r, bc_r = build(tune), build(conf_h)
    bt, bc = sc(bt_r, tune), sc(bc_r, conf_h)
    print(f"stack3 baseline: tune {bt:.6f}  confirm {bc:.6f}\n")

    cands = {
        "A hierarchical chaining": dict(mode="hier"),
        "B provenance FIRST level": dict(mode="prov"),
        "B provenance LAST level": dict(mode="provlast"),
        "C CE peak protection": dict(mode="peak"),
        "D query-length eps": dict(qlen_eps=True),
        "E RRF-rank buckets": dict(mode="rrfb"),
    }
    print(f"  {'candidate':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'keep':>7s}")
    for nm, kw in cands.items():
        rt, rc = build(tune, **kw), build(conf_h, **kw)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(bc_r[q], qrels[q], 10)
             for q in conf_h if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        print(f"  {nm:<26s}{dt:>+12.6f}{m:>+12.6f}{m/se if se else 0:>+8.2f}"
              f"{'YES' if (dt > 0 and m > 0) else '-':>7s}")


if __name__ == "__main__":
    main()
