"""TASK 1: character 3/4-gram BM25 as a candidate third retrieval voter.

Rationale (Taha's): Libyan-dialect queries miss MORPHOLOGICALLY under word-level
Okapi BM25 - "احذية" versus "حذاء", clitic prefixes للـ and بـ. A char-ngram index
matches sub-word, so a query whose surface form never equals the catalog form can
still retrieve.

Why not rank_bm25: its get_scores does a Python list comprehension over all 107,041
docs PER QUERY TERM. Word queries carry ~3 terms; char-gram queries carry ~25, so the
same code path would take hours. This builds the BM25 weight matrix once as a sparse
(term x doc) CSR and scores a query by summing the rows of its terms - seconds total.

Boundary handling is char_wb style: every word is padded to " word " before n-grams
are taken, so grams encode word starts and ends. That is what makes the clitic case
work - " للحذاء " and " حذاء " share حذا/ذاء/اء_ while differing only at the prefix.

Writes runs/<name>.jsonl (depth --top-k) in the official run format, so it drops
straight into fuse_rrf / the probe harness as a third voter.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from common import BENCH, RUNS, load_jsonl, normalize_arabic, select_queries, setup_utf8, \
    write_run_csv

K1 = 1.5
B = 0.75


def gram_counts(text: str, ns: tuple[int, ...]) -> dict[str, int]:
    """char_wb-style n-grams: pad each word with spaces, then slide within the word."""
    out: dict[str, int] = {}
    for w in normalize_arabic(text).split():
        p = f" {w} "
        for n in ns:
            if len(p) < n:
                out[p] = out.get(p, 0) + 1
                continue
            for i in range(len(p) - n + 1):
                g = p[i : i + n]
                out[g] = out.get(g, 0) + 1
    return out


class CharBM25:
    def __init__(self, docs: list[str], ns: tuple[int, ...]):
        self.ns = ns
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        dl = np.zeros(len(docs), dtype=np.float64)
        for d, text in enumerate(tqdm(docs, desc="index")):
            gc = gram_counts(text, ns)
            dl[d] = sum(gc.values())
            for g, c in gc.items():
                t = vocab.get(g)
                if t is None:
                    t = vocab[g] = len(vocab)
                rows.append(t)
                cols.append(d)
                tfs.append(c)
        self.vocab = vocab
        n_docs, n_terms = len(docs), len(vocab)
        tf = np.asarray(tfs, dtype=np.float64)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        df = np.bincount(rows, minlength=n_terms).astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = dl.mean()
        denom = tf + K1 * (1.0 - B + B * dl[cols] / avgdl)
        w = idf[rows] * tf * (K1 + 1.0) / denom
        self.W = csr_matrix((w, (rows, cols)), shape=(n_terms, n_docs))
        self.n_docs = n_docs
        print(f"  terms={n_terms:,} nnz={self.W.nnz:,} avgdl={avgdl:.1f}")

    def scores(self, query: str) -> np.ndarray:
        idx = [self.vocab[g] for g in gram_counts(query, self.ns) if g in self.vocab]
        if not idx:
            return np.zeros(self.n_docs)
        return np.asarray(self.W[idx].sum(axis=0)).ravel()


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="3,4", help="comma-separated n-gram sizes")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--n-queries", type=int, default=0, help="0 = all")
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    ns = tuple(int(x) for x in args.ns.split(","))
    name = args.name or f"charbm25_{''.join(map(str, ns))}gram_all"

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), args.qids, args.n_queries)
    item_ids = [str(r["item_id"]) for r in corpus]
    print(f"corpus={len(corpus):,} queries={len(queries):,} ns={ns} run={name}")

    t0 = time.time()
    bm = CharBM25([r["name"] for r in corpus], ns)
    print(f"index built in {time.time() - t0:.1f}s")

    run: dict[str, list[str]] = {}
    k = min(args.top_k, len(item_ids))
    for row in tqdm(queries, desc="retrieve"):
        s = bm.scores(row["query"])
        # overfetch then sort (score desc, doc index asc): argpartition alone slices
        # tie groups arbitrarily, and doc index ascending IS the low-product-id prior.
        pool = min(k + 512, len(item_ids))
        top = np.argpartition(-s, pool - 1)[:pool]
        top = top[np.lexsort((top, -s[top]))][:k]
        run[str(row["query_id"])] = [item_ids[i] for i in top]

    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in run.items():
            f.write(json.dumps({"query_id": qid,
                                "retrieved": [{"item_id": i} for i in ids]},
                               ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{name}.csv", {q: ids[:10] for q, ids in run.items()})
    print(f"wrote runs/{name}.jsonl (depth {k}) and runs/{name}.csv")


if __name__ == "__main__":
    main()
