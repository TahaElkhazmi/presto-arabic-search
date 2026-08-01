"""Play 2 Stage-1.5 data: corpus-grounded synthetic LambdaLoss groups.

Mechanism (FINDINGS 2026-07-22 autopsy): the dominant live failure is same-brand /
same-category variant confusion inside near-tie bands, and the provided training
labels cannot teach it (8-17% corpus grounding).
This builder generates groups that are grounded in the scoring corpus BY
CONSTRUCTION and replicate the curated-confusable signal profile that made the
provided hard negatives work (and mined rank-26+ negatives fail):

  query    = size/unit-stripped token prefix of a corpus name (k sampled to match
             the benchmark token-count distribution), optional Set-D filler
             injected at the measured benchmark rate;
  positive = the source corpus name (raw surface form, lowest-id representative);
  hard negatives = char-3gram nearest-neighbor corpus names that are DIFFERENT
             products AND that the query lexically discriminates against
             (containment gap >= GAP vs the positive) - the same-brand variant
             confusables the live system loses on;
  random negatives = 2 random corpus names.

False-negative hygiene (the v3 lesson, measured BEFORE training): the containment
gap rule structurally excludes neighbors the query cannot distinguish (same base
product, different size; sibling duplicates), and the validation report measures
residual contamination on synthetic queries whose text exactly matches a
benchmark query (qrels for MEASUREMENT ONLY).

Output: data/teacher/synthetic_curriculum_groups.parquet (cols query, docs, labels)
consumed by train_reranker.py --loss lambda --groups-parquet.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import BENCH, DATA, load_jsonl, normalize_arabic, setup_utf8
from stacker_features import _UNIT_CLASS, char_ngrams, containment

setup_utf8()

FILLER_APPEND = "للدار"  # measured 12.37% of benchmark queries, suffix usage
FILLER_VERBS = ["عطيني", "هاتلي", "ندور", "محتاج", "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي"]

_NUMERIC = re.compile(r"^\d+([.,]\d+)?$")
_NUM_UNIT = re.compile(r"^\d+([.,]\d+)?[a-zء-ي]+$")
_SEP = re.compile(r"^[-x×*/+.,()]+$")
UNITS = set(_UNIT_CLASS)


def content_tokens(norm_name: str) -> list[str]:
    """Tokens with sizes/units/numbers/separators dropped (query-side statistics:
    benchmark queries carry digits in ~12% vs 49% of names)."""
    out = []
    for t in norm_name.split():
        if _SEP.match(t) or _NUMERIC.match(t) or _NUM_UNIT.match(t) or t in UNITS:
            continue
        out.append(t)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-groups", type=int, default=60_000)
    p.add_argument("--neighbors", type=int, default=30, help="mined neighbor candidates per source")
    p.add_argument("--hard-cap", type=int, default=5, help="confusable negatives kept per group")
    p.add_argument("--min-hard", type=int, default=3, help="skip sources with fewer valid confusables")
    p.add_argument("--gap", type=float, default=0.15, help="required cont(q,pos)-cont(q,neg) gap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DATA / "teacher" / "synthetic_curriculum_groups.parquet")
    args = p.parse_args()
    rnd = random.Random(args.seed)
    t0 = time.time()

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    surface: dict[str, str] = {}  # normalized name -> raw name of lowest id
    for row in sorted(corpus, key=lambda r: str(r["item_id"])):
        surface.setdefault(normalize_arabic(row["name"]), row["name"])
    uniq = sorted(surface)
    N = len(uniq)
    print(f"{len(corpus)} ids, {N} unique names  [{time.time() - t0:.0f}s]", flush=True)

    # char-3gram incidence matrix + idf (same construction as build_cont3_voter.py)
    vocab: dict[str, int] = {}
    rows, cols = [], []
    gram_sets = []
    for ni, n in enumerate(uniq):
        gs = char_ngrams(n, 3)
        gram_sets.append(gs)
        for g in gs:
            gi = vocab.setdefault(g, len(vocab))
            rows.append(ni)
            cols.append(gi)
    V = len(vocab)
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, V))
    df_ = np.asarray(A.sum(axis=0)).ravel()
    idf = np.log(1.0 + N / (df_ + 0.5)).astype(np.float32)
    A_idf = (A @ sparse.diags(idf)).tocsr()
    print(f"{V} grams, matrix built  [{time.time() - t0:.0f}s]", flush=True)

    # eligible sources: >= 2 content tokens
    tok_cache = {n: content_tokens(n) for n in uniq}
    eligible = [ni for ni, n in enumerate(uniq) if len(tok_cache[n]) >= 2]
    rnd.shuffle(eligible)
    print(f"{len(eligible)} eligible sources  [{time.time() - t0:.0f}s]", flush=True)

    groups = []
    seen_qp = set()
    n_skipped_hard = 0
    BATCH = 512
    for b0 in range(0, len(eligible), BATCH):
        batch = eligible[b0 : b0 + BATCH]
        S = (A_idf[batch] @ A.T).toarray()  # neighbor similarity by shared-gram idf mass
        for r, ni in enumerate(batch):
            if len(groups) >= args.max_groups:
                break
            name = uniq[ni]
            toks = tok_cache[name]

            # query core: token-count sampled to benchmark shape (1: ~10%, 2: ~77%, 3+: ~13%)
            u = rnd.random()
            k = 1 if u < 0.07 else (2 if u < 0.80 else 3)
            k = min(k, len(toks))
            if rnd.random() < 0.15 and len(toks) > k:  # skip-token variant: 1st + later tokens
                core_toks = [toks[0]] + rnd.sample(toks[1:], k - 1) if k > 1 else [toks[0]]
            else:
                core_toks = toks[:k]
            core = " ".join(core_toks)
            qgrams = char_ngrams(core, 3)
            cpos = containment(qgrams, gram_sets[ni])
            if cpos < 0.60:  # query core must actually live in the positive
                continue

            scores = S[r]
            scores[ni] = -1.0
            cand = np.argpartition(-scores, args.neighbors)[: args.neighbors]
            cand = cand[np.argsort(-scores[cand])]
            negs = []
            for cj in cand:
                if scores[cj] <= 0:
                    break
                if cpos - containment(qgrams, gram_sets[cj]) < args.gap:
                    continue  # query cannot discriminate: potential false negative
                negs.append(uniq[cj])
                if len(negs) >= args.hard_cap:
                    break
            if len(negs) < args.min_hard:
                n_skipped_hard += 1
                continue

            # filler injection at measured benchmark rates
            u = rnd.random()
            if u < 0.12:
                query = f"{core} {FILLER_APPEND}"
            elif u < 0.155:
                query = f"{rnd.choice(FILLER_VERBS)} {core}"
            else:
                query = core

            if (query, name) in seen_qp:
                continue
            seen_qp.add((query, name))

            for _ in range(2):  # random catalog negatives
                while True:
                    rj = rnd.randrange(N)
                    if rj != ni and uniq[rj] not in negs:
                        break
                negs.append(uniq[rj])

            docs = [surface[name]] + [surface[n] for n in negs]
            groups.append((query, docs, [1.0] + [0.0] * len(negs)))
        if len(groups) >= args.max_groups:
            break
        if (b0 // BATCH) % 20 == 0:
            print(f"  {b0}/{len(eligible)} sources -> {len(groups)} groups  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    sizes = [len(d) for _, d, _ in groups]
    print(f"\n{len(groups)} groups (skipped {n_skipped_hard} thin-confusable sources), "
          f"docs/group mean {sum(sizes) / len(sizes):.2f} max {max(sizes)}, "
          f"total pair-forwards {sum(sizes)}  [{time.time() - t0:.0f}s]")
    out = pd.DataFrame(groups, columns=["query", "docs", "labels"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
