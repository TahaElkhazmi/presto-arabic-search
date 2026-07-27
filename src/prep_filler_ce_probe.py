"""Prep for LEAD 1: does stripping dialect fillers from the CROSS-ENCODER input help?

Today we strip the Set-D dialect fillers ONLY when building the cont3 tie-break key.
BM25, the dense retriever and the cross-encoder all still see the RAW query including
the filler. `autopsy_zero_ordering.py` found the zero cohort is 35.3% filler-bearing
against 11.6% for high-scoring queries, a 3x over-representation, with failures like
'حزام للدار', 'عدس للدار', 'foramen للدار', 'ندور على بوكس هديه'.

Writes a query file whose TEXT is filler-stripped but whose ids are unchanged, so
`rerank.py --queries` re-scores the identical candidate pools with a cleaned query.
Pools come from the run files and are untouched, which isolates the CE effect from
any retrieval change.

EXPERIMENTAL DESIGN NOTE: the sample is drawn from ALL filler-bearing queries in the
untouched remainder, NOT from the zero cohort. Zero-cohort queries score 0 and can
only improve, so measuring there would guarantee a positive result regardless of
merit. The unbiased sample is what makes the number mean something.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import BENCH, ROOT, load_jsonl, load_qid_file, normalize_arabic, setup_utf8
from probe_harness import DIALECT_FILLERS

setup_utf8()

OUT_Q = ROOT / "data" / "queries_filler_stripped.jsonl"
OUT_IDS = ROOT / "splits" / "probe_filler800.txt"
N = 800


def main():
    queries = load_jsonl(BENCH / "queries.jsonl")

    used = load_qid_file(ROOT / "splits" / "val_qids.txt")
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt",
              "dev_listwise200.txt"):
        p = ROOT / "splits" / f
        if p.exists():
            used |= load_qid_file(p)

    filler_qs, stripped = [], {}
    for q in queries:
        qid = str(q["query_id"])
        toks = normalize_arabic(q["query"]).split()
        kept = [t for t in toks if t not in DIALECT_FILLERS]
        if len(kept) == len(toks) or not kept:
            continue  # no filler, or stripping would empty it
        stripped[qid] = " ".join(kept)
        if qid not in used:
            filler_qs.append(qid)

    print(f"filler-bearing queries in benchmark: {len(stripped):,} "
          f"({100 * len(stripped) / len(queries):.1f}%)")
    print(f"available in the untouched remainder: {len(filler_qs):,}")

    picked = sorted(random.Random(48).sample(filler_qs, min(N, len(filler_qs))), key=int)
    OUT_IDS.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"sampled {len(picked)} -> {OUT_IDS.relative_to(ROOT)}")

    # full query file, stripped text where applicable, ids preserved
    with open(OUT_Q, "w", encoding="utf-8") as f:
        for q in queries:
            qid = str(q["query_id"])
            text = stripped.get(qid, q["query"])
            f.write(json.dumps({"query_id": q["query_id"], "query": text},
                               ensure_ascii=False) + "\n")
    print(f"wrote {OUT_Q.relative_to(ROOT)}")

    print("\nexamples (raw -> stripped):")
    for qid in picked[:8]:
        raw = next(q["query"] for q in queries if str(q["query_id"]) == qid)
        print(f"  {raw!r:<44s} -> {stripped[qid]!r}")

    print("\nNEXT: score the stripped variant into its OWN cache (live cache untouched)")
    print("  ./.venv/Scripts/python.exe src/rerank.py \\")
    print("    --runs runs/bm25_all.jsonl runs/dense_lora_v1_all.jsonl \\")
    print("    --depths 100 100 --weights 3 1 --rrf-k 60 \\")
    print("    --model models/qwen3-reranker-arabic-ecom-rerank_v7synth \\")
    print(f"    --queries {OUT_Q.relative_to(ROOT)} --qids {OUT_IDS.relative_to(ROOT)} \\")
    print("    --cache-name rerank_v7synth_fillerstripped_len160_bfloat16 --score-only")


if __name__ == "__main__":
    main()
