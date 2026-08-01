"""EXPERIMENT 1: Arabic proclitic / definite-article stripping in the in-band key.

TARGET: recover containment and position matches lost when the catalog writes a bare
form and the query writes a cliticised one, or vice versa - "حذاء" versus "للحذاء",
"بالحذاء", "وحذاء".

WHAT IS ALREADY DONE, so this does not re-do it: `common.normalize_arabic` ALREADY
folds the Alif/Hamza variants (أ إ آ -> ا), the maksura (ى -> ي) and the marbuta
(ة -> ه), and strips diacritics and tatweel. Step 1's normalisation half is therefore
in force in every number this project has ever reported. The genuinely new part is
CLITIC STRIPPING: the proclitics ب ل ف و ك and the definite article ال.

TWO READINGS OF THE BRIEF, both measured, because they conflict.

  A. DESTRUCTIVE (what the brief literally asks): stem BEFORE computing containment,
     match-position and first-token position, i.e. replace the shipped key's own
     components with stemmed ones. This can only be judged end to end, because it moves
     the levels themselves rather than adding one.

  B. NON-DESTRUCTIVE (what the brief's own execution rule demands): leave the shipped
     key exactly as it is and APPEND a stemmed level above the low-id fallback, so it
     only ever breaks ties the existing levels could not. This is the shape that has
     actually won here - the prefix key and first-token position both partition without
     displacing anything.

Reading A is what the standing evidence argues against: every continuous or replacing
in-group signal tried so far has lost (GATE-AraBert -0.031, cross-encoder as key -0.031,
TQD buckets -0.008), while every additive partition has won. Measuring both settles it
rather than assuming.

PRIOR WORTH STATING: this project's context file records that Arabic STEMMING was
already found unhelpful, but that was for BM25 RETRIEVAL. The in-band key is a different
mechanism and is where every recent gain has come from, so it is worth one clean test.

Screening runs on holdout_remainder8012 (tune/confirm halves). Full-benchmark and
arbiter confirmation is a separate pass, reserved for whatever survives.
"""

import bisect
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import DIALECT_FILLERS, Harness
from probe_phase3_predicates import per_query_auc
from stacker_features import char_ngrams, containment

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.50, eps=0.030, eps_hi=0.050, conf_t=0.10)   # T = 3.2
T_LIVE, BETA = 3.2, 0.35
S_CONT, S_MP, S_FT, S_FEAT = 1e6, 1e2, 1e-2, 1e-6

PROCLITICS = ("ب", "ل", "ف", "و", "ك")


def stem(w: str) -> str:
    """Strip one proclitic and/or the definite article. Input is already normalized.

    Order matters: the proclitic comes first, so "بالحذاء" -> "الحذاء" -> "حذاء".
    "لل" is special-cased because the article's alef elides after the lam ("للحذاء"),
    which the generic rules would otherwise leave as "لحذاء".
    Length guards keep short function words intact, per the brief's "length > 2".
    """
    if len(w) > 3 and w.startswith("لل"):
        return w[2:]
    if len(w) > 3 and w.startswith("ال"):
        return w[2:]
    if len(w) > 2 and w[0] in PROCLITICS:
        w = w[1:]
        if len(w) > 3 and w.startswith("ال"):
            w = w[2:]
    return w


def stem_text(t: str) -> str:
    return " ".join(stem(w) for w in t.split())


def num(iid):
    d = "".join(ch for ch in iid if ch.isdigit())
    return int(d) if d else 0


def main():
    hold = sorted(load_qid_file(ROOT / "splits" / "holdout_remainder8012.txt"), key=int)
    shuf = list(hold)
    random.Random(77).shuffle(shuf)
    tune = sorted(shuf[: len(shuf) // 2], key=int)
    conf = sorted(shuf[len(shuf) // 2:], key=int)

    norm = {str(r["item_id"]): normalize_arabic(r["name"])
            for r in load_jsonl(BENCH / "corpus.jsonl")}
    ntok = {i: n.split() for i, n in norm.items()}
    snorm = {i: stem_text(n) for i, n in norm.items()}
    stok = {i: n.split() for i, n in snorm.items()}
    sgram = {}

    # how much does stemming actually change the text? cheap sanity number
    changed = sum(1 for i in norm if norm[i] != snorm[i])
    print(f"catalog titles altered by stemming: {changed:,}/{len(norm):,} "
          f"({changed/len(norm):.1%})")

    print("building harness over the holdout ...", flush=True)
    h = Harness(hold, T=T_LIVE, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for q in h.qids:
        v = sorted((h.ce[(q, i)] for i in h.pools[q]), reverse=True)
        h.conf[q] = v[0] - v[2] if len(v) >= 3 else 1.0
    qrels = h.qrels

    qt, sqt, sqg = {}, {}, {}
    for q in h.qids:
        t = [x for x in normalize_arabic(h.queries[q]).split() if x not in DIALECT_FILLERS]
        t = t or normalize_arabic(h.queries[q]).split()
        qt[q] = t
        sqt[q] = [stem(x) for x in t]
        sqg[q] = char_ngrams(" ".join(sqt[q]), 3)
    qchanged = sum(1 for q in h.qids if qt[q] != sqt[q])
    print(f"queries altered by stemming: {qchanged:,}/{len(h.qids):,} "
          f"({qchanged/len(h.qids):.1%})\n")

    def sgrams(i):
        g = sgram.get(i)
        if g is None:
            g = sgram[i] = char_ngrams(snorm[i], 3)
        return g

    mpc, ftc, smpc, sftc, scc = {}, {}, {}, {}, {}

    def _mp(toks, qq):
        n = len(qq)
        if not n:
            return 0.0
        for k in range(len(toks) - n + 1):
            if toks[k:k + n] == qq:
                return 1.0 / (1.0 + k)
        return 0.0

    def _ft(toks, qq):
        return 3.0 / (1.0 + toks.index(qq[0])) if (qq and qq[0] in toks) else 0.0

    def matchpos(q, i):
        v = mpc.get((q, i))
        if v is None:
            v = mpc[(q, i)] = _mp(ntok[i], qt[q])
        return v

    def firsttok(q, i):
        v = ftc.get((q, i))
        if v is None:
            v = ftc[(q, i)] = _ft(ntok[i], qt[q])
        return v

    def s_matchpos(q, i):
        v = smpc.get((q, i))
        if v is None:
            v = smpc[(q, i)] = _mp(stok[i], sqt[q])
        return v

    def s_firsttok(q, i):
        v = sftc.get((q, i))
        if v is None:
            v = sftc[(q, i)] = _ft(stok[i], sqt[q])
        return v

    def s_cont(q, i):
        v = scc.get((q, i))
        if v is None:
            v = scc[(q, i)] = containment(sqg[q], sgrams(i))
        return v

    bonus = lambda q, i: BETA * matchpos(q, i)

    def build(keyf, qids=None):
        return h.build_run(key=keyf, bonus_fn=bonus, qids=qids, **LIVE)

    shipped = lambda q, i: (h.cont3(q, i) * S_CONT + matchpos(q, i) * S_MP
                            + firsttok(q, i) * S_FT)
    base = build(shipped)

    def sc(r, qs):
        v = [ndcg_at_k(r[q], qrels[q], 10) for q in qs if q in r and q in qrels]
        return sum(v) / len(v)

    bt, bc = sc(base, tune), sc(base, conf)
    print(f"SHIPPED 4-level key: tune {bt:.6f}  confirm {bc:.6f}\n")

    ndm = {q: ndcg_at_k(base[q], qrels[q], 10) for q in hold if q in qrels}
    zc = [q for q, v in ndm.items() if v == 0.0]
    hc = [q for q, v in ndm.items() if v >= 0.9]

    # ---------- reading A: destructive, stem the key's own components ----------
    variants = {
        "A1 stem cont only": lambda q, i: (s_cont(q, i) * S_CONT + matchpos(q, i) * S_MP
                                           + firsttok(q, i) * S_FT),
        "A2 stem cont+mp+ft": lambda q, i: (s_cont(q, i) * S_CONT + s_matchpos(q, i) * S_MP
                                            + s_firsttok(q, i) * S_FT),
        "A3 stem mp+ft only": lambda q, i: (h.cont3(q, i) * S_CONT + s_matchpos(q, i) * S_MP
                                            + s_firsttok(q, i) * S_FT),
    }
    # ---------- reading B: non-destructive, append a stemmed 5th level ----------
    addl = {
        "B1 +stemmed containment": s_cont,
        "B2 +stemmed match-pos": s_matchpos,
        "B3 +stemmed first-token": s_firsttok,
    }
    for nm, fn in addl.items():
        variants[nm] = (lambda q, i, f=fn: shipped(q, i) + f(q, i) * S_FEAT)

    print("END TO END against the shipped key (holdout halves)\n")
    print(f"  {'variant':<26s}{'tune d':>12s}{'confirm d':>12s}{'t':>8s}{'verdict':>10s}")
    survivors = []
    for nm, kf in variants.items():
        rt, rc = build(kf, tune), build(kf, conf)
        dt = sc(rt, tune) - bt
        d = [ndcg_at_k(rc[q], qrels[q], 10) - ndcg_at_k(base[q], qrels[q], 10)
             for q in conf if q in qrels]
        m = sum(d) / len(d)
        se = st.pstdev(d) / len(d) ** 0.5
        t = m / se if se else 0
        ok = m >= 0.0015 and t > 2
        print(f"  {nm:<26s}{dt:>+12.6f}{m:>+12.6f}{t:>+8.2f}{'CARRY' if ok else 'drop':>10s}")
        if ok:
            survivors.append(nm)

    # ---------- combined-key AUC for the additive forms, at the real decision cell ----
    def cells(q):
        out = {}
        for i in h.pools[q]:
            k = (round(h.cont3(q, i), 12), round(matchpos(q, i), 12),
                 round(firsttok(q, i), 12))
            out.setdefault(k, []).append(i)
        return out

    def auc(fn, qs):
        pairs = []
        for q in qs:
            for sub in cells(q).values():
                if len(sub) < 2:
                    continue
                order = sorted(sub, key=lambda i: (-fn(q, i), num(i)))
                val = {i: -r for r, i in enumerate(order)}
                p = [val[i] for i in sub if qrels[q].get(i, 0) > 0]
                n = [val[i] for i in sub if qrels[q].get(i, 0) <= 0]
                if p and n:
                    pairs.append((p, n))
        return per_query_auc(pairs)

    lowid = lambda q, i: -num(i)
    print(f"\nCOMBINED-key AUC in the cell where all three shipped levels tie "
          f"(zero n={len(zc):,}, healthy n={len(hc):,})")
    print(f"  {'low-id (incumbent)':<26s}{auc(lowid, zc):>11.4f}{auc(lowid, hc):>11.4f}")
    for nm, fn in addl.items():
        print(f"  {nm:<26s}{auc(fn, zc):>11.4f}{auc(fn, hc):>11.4f}")

    print(f"\n  survivors needing a full-benchmark + arbiter pass: "
          f"{survivors if survivors else 'none'}")


if __name__ == "__main__":
    main()
