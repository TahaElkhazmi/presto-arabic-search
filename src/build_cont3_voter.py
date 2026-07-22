"""Play 1: char-3gram containment retriever over the full 107k catalog (CPU only).

Scores every corpus name against every benchmark query with IDF-weighted char-3gram
containment (the proven cont3 signal, promoted from in-band tie-key to a full
retrieval run), Set-D fillers stripped from queries, low-id tie-break with
boundary-tie completion (the argpartition trap is handled).

Sparse matrix over the 68k UNIQUE normalized names, then id expansion.
Output: runs/cont3idf_all.jsonl (depth 1000, official run format).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\LOQ\presto_v2\src")

import numpy as np
from scipy import sparse

from common import BENCH, ROOT, load_jsonl, normalize_arabic
from stacker_features import char_ngrams

DIALECT_FILLERS = {normalize_arabic(w) for w in (
    "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
    "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

DEPTH = 1000
OVERFETCH_NAMES = 1400


def main():
    t0 = time.time()
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    queries = load_jsonl(BENCH / "queries.jsonl")

    # unique normalized names with ascending-id lists
    name_ids: dict[str, list[str]] = {}
    for r in corpus:
        name_ids.setdefault(normalize_arabic(r["name"]), []).append(str(r["item_id"]))
    uniq = sorted(name_ids)
    for n in uniq:
        name_ids[n].sort()  # zero-padded ids: lexicographic == numeric
    N = len(uniq)
    print(f"{len(corpus)} ids, {N} unique names  [{time.time() - t0:.0f}s]", flush=True)

    # vocab + binary incidence matrix (names x grams)
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for ni, n in enumerate(uniq):
        for g in char_ngrams(n, 3):
            gi = vocab.setdefault(g, len(vocab))
            rows.append(ni)
            cols.append(gi)
    V = len(vocab)
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, V))
    df = np.asarray(A.sum(axis=0)).ravel()
    idf = np.log(1.0 + N / (df + 0.5)).astype(np.float32)
    print(f"{V} grams  [{time.time() - t0:.0f}s]", flush=True)

    # weight matrix: A * idf per column (for numerator sum of idf over q&n)
    A_idf = (A @ sparse.diags(idf)).tocsr()

    # query gram vectors (Set-D stripped)
    idf_unk = float(np.log(1.0 + N / 0.5))
    qids, qrows, qcols, qden = [], [], [], []
    for qi, q in enumerate(queries):
        qn = normalize_arabic(q["query"])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        grams = char_ngrams(" ".join(kept) if kept else qn, 3)
        den = 0.0
        for g in grams:
            gi = vocab.get(g)
            if gi is None:
                den += idf_unk
            else:
                qrows.append(qi)
                qcols.append(gi)
                den += float(idf[gi])
        qids.append(str(q["query_id"]))
        qden.append(max(den, 1e-9))
    Q = sparse.csr_matrix(
        (np.ones(len(qrows), dtype=np.float32), (qrows, qcols)),
        shape=(len(qids), V))
    qden = np.array(qden, dtype=np.float32)
    print(f"queries vectorized  [{time.time() - t0:.0f}s]", flush=True)

    out_path = ROOT / "runs" / "cont3idf_all.jsonl"
    BATCH = 256
    with open(out_path, "w", encoding="utf-8") as f:
        for b0 in range(0, len(qids), BATCH):
            b1 = min(b0 + BATCH, len(qids))
            S = (Q[b0:b1] @ A_idf.T).toarray()  # (batch, N) idf-overlap sums
            S /= qden[b0:b1, None]
            for r in range(b1 - b0):
                scores = S[r]
                k = min(OVERFETCH_NAMES, N)
                top = np.argpartition(-scores, k - 1)[:k]
                thr = scores[top].min()
                if thr > 0:
                    top = np.flatnonzero(scores >= thr)  # boundary-tie completion
                else:
                    top = top[scores[top] > 0]
                pairs = []
                for ni in top:
                    s = float(scores[ni])
                    for iid in name_ids[uniq[ni]]:
                        pairs.append((-s, iid))
                pairs.sort()
                retrieved = [{"item_id": iid} for _, iid in pairs[:DEPTH]]
                f.write(json.dumps(
                    {"query_id": qids[b0 + r], "retrieved": retrieved},
                    ensure_ascii=False, separators=(",", ":")) + "\n")
            if (b0 // BATCH) % 20 == 0:
                print(f"  {b1}/{len(qids)}  [{time.time() - t0:.0f}s]", flush=True)

    print(f"wrote {out_path}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
