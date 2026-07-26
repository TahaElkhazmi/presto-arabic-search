"""Synthetic curriculum v2: GRADED relevance labels + the hard variants v1 threw away.

v1 (which produced the project's largest model gain, +0.0157 pure-CE) had a
structural limitation: binary labels. To avoid false negatives it EXCLUDED every
neighbour the query could not discriminate from the positive (containment gap
< 0.15) - and those excluded near-duplicates are precisely the same-brand /
size-variant confusables that the loss autopsy identified as the dominant remaining
error mode. Binary labels forced us to discard the most informative examples.

v2 fixes that with graded labels, which is what LambdaLoss was built to consume
(verified: it computes gains as 2^label - 1, and its docstring states continuous
labels are supported - we had been feeding it 1/0 all along):

  grade 3  the source product
  grade 2  a pure size/pack variant (identical content tokens, different number/unit)
  grade 1  a close variant (content-token Jaccard >= 0.5: flavour/sub-variant)
  grade 0  a low-overlap neighbour, or a random catalogue item

So the near-duplicates are now KEPT and taught as "nearly as good" rather than
either discarded (v1) or mislabelled as wrong (which is the v3 contamination failure
mode from 2026-07-18). Grade 2 also mirrors a real property of the benchmark: within
duplicate-name groups, relevance grades are non-increasing rather than all-or-nothing.

Scale: v1 emitted one query per source (capped by ~66k eligible sources). v2 emits
several distinct query variants per source (prefix / skip-token / head-only / filler),
so group count is decoupled from catalogue size.

Group size is held at 8 documents to keep the VRAM profile identical to the validated
v1 recipe (micro-batch 2 x 8 docs = 16 pair-sequences, safely under the ~20 spill
threshold documented 2026-07-16).

CHECKPOINTED: sources are processed in chunks; each chunk appends a parquet part and
updates a state file, so a power cut costs at most one chunk. Re-run the identical
command to resume.
"""

import argparse
import json
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

FILLER_APPEND = "للدار"
FILLER_VERBS = ["عطيني", "هاتلي", "ندور", "محتاج", "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي"]

_NUMERIC = re.compile(r"^\d+([.,]\d+)?$")
_NUM_UNIT = re.compile(r"^\d+([.,]\d+)?[a-zء-ي]+$")
_SEP = re.compile(r"^[-x×*/+.,()]+$")
UNITS = set(_UNIT_CLASS)

GROUP_DOCS = 8          # 1 source + up to MAX_NB neighbours + random padding
MAX_NB = 5              # neighbour slots per group
MAX_G2, MAX_G1 = 2, 2   # caps on the graded buckets; g0 neighbours fill the remainder
MIN_NB = 3              # reject groups with fewer real neighbours (matches v1's min-hard)


def content_tokens(norm_name: str) -> list[str]:
    out = []
    for t in norm_name.split():
        if _SEP.match(t) or _NUMERIC.match(t) or _NUM_UNIT.match(t) or t in UNITS:
            continue
        out.append(t)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def grade_for(content_src: set, content_nb: set) -> int:
    """Graded relevance of a neighbour relative to the source product."""
    if content_nb == content_src:
        return 2                      # identical content, differs only in size/pack
    if jaccard(content_src, content_nb) >= 0.5:
        return 1                      # flavour / sub-variant
    return 0


def make_queries(toks: list[str], rnd: random.Random, n: int) -> list[str]:
    """Distinct query variants for one source: prefix / skip-token / head-only."""
    out = []
    for _ in range(n * 3):            # oversample, dedupe below
        if len(out) >= n:
            break
        u = rnd.random()
        k = 1 if u < 0.07 else (2 if u < 0.80 else 3)
        k = min(k, len(toks))
        style = rnd.random()
        if style < 0.15 and len(toks) > k and k > 1:
            core = " ".join([toks[0]] + rnd.sample(toks[1:], k - 1))
        elif style < 0.25:
            core = toks[0]            # head/brand only (the weakest query segment)
        else:
            core = " ".join(toks[:k])
        u2 = rnd.random()
        if u2 < 0.12:
            q = f"{core} {FILLER_APPEND}"
        elif u2 < 0.155:
            q = f"{rnd.choice(FILLER_VERBS)} {core}"
        else:
            q = core
        if q not in out:
            out.append(q)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-groups", type=int, default=80_000)
    p.add_argument("--queries-per-source", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=40)
    p.add_argument("--min-docs", type=int, default=5, help="skip sources with fewer graded docs")
    p.add_argument("--chunk", type=int, default=8000, help="sources per checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=DATA / "teacher" / "synthetic_curriculum_v2_groups.parquet")
    args = p.parse_args()

    work = args.out.parent / (args.out.stem + "_parts")
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    t0 = time.time()

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    surface: dict[str, str] = {}
    for row in sorted(corpus, key=lambda r: str(r["item_id"])):
        surface.setdefault(normalize_arabic(row["name"]), row["name"])
    uniq = sorted(surface)
    N = len(uniq)
    print(f"{len(corpus)} ids, {N} unique names  [{time.time() - t0:.0f}s]", flush=True)

    vocab: dict[str, int] = {}
    rows, cols = [], []
    gram_sets, tok_sets = [], []
    for ni, n in enumerate(uniq):
        gs = char_ngrams(n, 3)
        gram_sets.append(gs)
        tok_sets.append(set(content_tokens(n)))
        for g in gs:
            gi = vocab.setdefault(g, len(vocab))
            rows.append(ni)
            cols.append(gi)
    A = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(N, len(vocab)))
    df_ = np.asarray(A.sum(axis=0)).ravel()
    idf = np.log(1.0 + N / (df_ + 0.5)).astype(np.float32)
    A_idf = (A @ sparse.diags(idf)).tocsr()
    print(f"{len(vocab)} grams, matrix built  [{time.time() - t0:.0f}s]", flush=True)

    eligible = [ni for ni in range(N) if len(tok_sets[ni]) >= 2]
    random.Random(args.seed).shuffle(eligible)
    print(f"{len(eligible)} eligible sources  [{time.time() - t0:.0f}s]", flush=True)

    done_chunks, n_groups = [], 0
    if state_path.exists():
        st = json.loads(state_path.read_text())
        done_chunks, n_groups = st["done_chunks"], st["n_groups"]
        print(f"RESUMING: {len(done_chunks)} chunks done, {n_groups} groups banked", flush=True)

    chunks = [(i, min(i + args.chunk, len(eligible)))
              for i in range(0, len(eligible), args.chunk)]

    for ci, (lo, hi) in enumerate(chunks):
        if ci in done_chunks:
            continue
        if n_groups >= args.max_groups:
            break
        rnd = random.Random(args.seed + ci)
        batch = eligible[lo:hi]
        out_rows = []
        S = None
        for bi in range(0, len(batch), 512):
            sub = batch[bi : bi + 512]
            S = (A_idf[sub] @ A.T).toarray()
            for r, ni in enumerate(sub):
                if n_groups + len(out_rows) >= args.max_groups:
                    break
                toks = content_tokens(uniq[ni])
                queries = make_queries(toks, rnd, args.queries_per_source)
                if not queries:
                    continue

                scores = S[r]
                scores[ni] = -1.0
                cand = np.argpartition(-scores, args.neighbors)[: args.neighbors]
                cand = cand[np.argsort(-scores[cand])]
                # graded buckets are capped; grade-0 neighbours (high char-3gram
                # similarity, low content overlap - v1's staple hard negative) fill
                # whatever slots remain, so yield matches v1 while grades are added
                # wherever the catalogue actually provides variants.
                g2, g1, g0 = [], [], []
                for cj in cand:
                    if scores[cj] <= 0:
                        break
                    if len(g2) + len(g1) + len(g0) >= MAX_NB:
                        break
                    g = grade_for(tok_sets[ni], tok_sets[cj])
                    if g == 2 and len(g2) < MAX_G2:
                        g2.append((uniq[cj], 2))
                    elif g == 1 and len(g1) < MAX_G1:
                        g1.append((uniq[cj], 1))
                    elif g == 0:
                        g0.append((uniq[cj], 0))
                graded = g2 + g1 + g0
                if len(graded) < MIN_NB:
                    continue

                n_rand = GROUP_DOCS - 1 - len(graded)
                rands = []
                while len(rands) < n_rand:
                    rj = rnd.randrange(N)
                    if rj != ni and uniq[rj] not in {n for n, _ in graded}:
                        rands.append((uniq[rj], 0))

                docs = [surface[uniq[ni]]] + [surface[n] for n, _ in graded + rands]
                labels = [3.0] + [float(g) for _, g in graded + rands]
                for q in queries:
                    if n_groups + len(out_rows) >= args.max_groups:
                        break
                    out_rows.append((q, docs, labels))
            if n_groups + len(out_rows) >= args.max_groups:
                break

        if out_rows:
            pd.DataFrame(out_rows, columns=["query", "docs", "labels"]).to_parquet(
                work / f"part_{ci:04d}.parquet", index=False)
            n_groups += len(out_rows)
        done_chunks.append(ci)
        state_path.write_text(json.dumps({"done_chunks": done_chunks, "n_groups": n_groups}))
        print(f"chunk {ci + 1}/{len(chunks)}: +{len(out_rows)} groups, total {n_groups} "
              f"[{time.time() - t0:.0f}s] [saved]", flush=True)

    parts = sorted(work.glob("part_*.parquet"))
    data = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    data = data.iloc[: args.max_groups]
    data.to_parquet(args.out, index=False)

    lab = np.concatenate([np.asarray(x) for x in data["labels"]])
    sizes = data["docs"].map(len)
    print(f"\n{len(data)} groups, docs/group mean {sizes.mean():.2f} max {sizes.max()}, "
          f"total pair-forwards {sizes.sum()}")
    print("grade distribution: " + ", ".join(
        f"g{g}={int((lab == g).sum())} ({(lab == g).mean():.1%})" for g in (3, 2, 1, 0)))
    print(f"wrote {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
