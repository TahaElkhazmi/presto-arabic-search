"""Play 0: loss autopsy of the live v6warm submission.

Per-query nDCG@10 of runs/submission_v6warm_calibrated_T23.csv plus the per-query
pool-oracle ceiling (best possible ordering of the frozen 100+100 pools), sliced by
query text features, CE confidence, and retrieval agreement, to locate where the
remaining loss concentrates. qrels are used for MEASUREMENT ONLY.

Outputs: printed tables + per-query parquet at scratchpad/autopsy_per_query.parquet.
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\LOQ\presto_v2\src")

import pandas as pd

from common import BENCH, DATA, ROOT, load_jsonl, load_run_csv, normalize_arabic, tokenize
from eval_ndcg import load_candidates, ndcg_at_k

SCRATCH = ROOT / "outputs"  # per-query parquet lands here (gitignored)

DIALECT_FILLERS = {normalize_arabic(w) for w in (
    "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
    "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

T = 2.3


def rescale(raw: float) -> float:
    s = min(max(raw, 1e-12), 1.0 - 1e-12)
    logit = math.log(s / (1.0 - s))
    return 1.0 / (1.0 + math.exp(-logit / T))


def stream_top100(path: Path) -> dict[str, list[str]]:
    """Parse a depth-1000 run jsonl keeping only top-100 ids per query (RAM-safe)."""
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["query_id"])] = [str(e["item_id"]) for e in obj["retrieved"][:100]]
    return out


def pool_oracle_ndcg(pool: list[str], qrel: dict[str, int], k: int = 10) -> float:
    grades = sorted((qrel.get(iid, 0) for iid in pool), reverse=True)
    dcg = 0.0
    for i, g in enumerate(grades[:k], start=1):
        if g > 0:
            dcg += (2.0 ** g - 1.0) / math.log2(i + 1)
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)
    idcg = 0.0
    for i, g in enumerate(ideal[:k], start=1):
        idcg += (2.0 ** g - 1.0) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("loading queries, corpus, qrels, live run ...", flush=True)
    queries = {str(q["query_id"]): q["query"] for q in load_jsonl(BENCH / "queries.jsonl")}
    names = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}
    name_count: dict[str, int] = {}
    for n in names.values():
        nn = normalize_arabic(n)
        name_count[nn] = name_count.get(nn, 0) + 1
    qrels = load_candidates(BENCH / "candidates.jsonl")
    live = load_run_csv(ROOT / "runs" / "submission_v6warm_calibrated_T23.csv")

    print("streaming bm25/dense top-100 ...", flush=True)
    bm25 = stream_top100(ROOT / "runs" / "bm25_all.jsonl")
    dense = stream_top100(ROOT / "runs" / "dense_lora_v1_all.jsonl")

    print("loading v6warm CE cache ...", flush=True)
    cdf = pd.read_parquet(DATA / "cache" / "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16.parquet")
    cache = {(q, i): s for q, i, s in zip(cdf["qid"], cdf["iid"], cdf["score"])}
    del cdf

    rows = []
    qids = sorted(set(live) & set(qrels))
    print(f"scoring {len(qids)} queries ...", flush=True)
    for idx, qid in enumerate(qids):
        qrel = qrels[qid]
        qtext = queries[qid]
        qn = normalize_arabic(qtext)
        toks = tokenize(qtext)
        pool = list(dict.fromkeys(bm25.get(qid, []) + dense.get(qid, [])))

        ces = sorted((rescale(cache.get((qid, iid), 0.0)) for iid in pool), reverse=True)
        conf = ces[0] - ces[4] if len(ces) >= 5 else 1.0

        b10, d10 = set(bm25.get(qid, [])[:10]), set(dense.get(qid, [])[:10])
        top10 = live[qid]
        nd = ndcg_at_k(top10, qrel, 10)
        oracle = pool_oracle_ndcg(pool, qrel)
        rows.append({
            "qid": qid,
            "ndcg": nd,
            "oracle": oracle,
            "loss": 1.0 - nd,
            "order_loss": oracle - nd,
            "pool_loss": 1.0 - oracle,
            "n_tok": len(toks),
            "n_chars": len(qn),
            "has_filler": any(t in DIALECT_FILLERS for t in qn.split()),
            "has_digit": bool(re.search(r"[0-9]", qn)),
            "has_latin": bool(re.search(r"[a-z]", qn)),
            "n_pos": sum(1 for g in qrel.values() if g > 0),
            "max_grade": max((g for g in qrel.values()), default=0),
            "conf": conf,
            "wide_band": conf < 0.10,
            "overlap10": len(b10 & d10) / 10.0,
            "hits10": sum(1 for iid in top10 if qrel.get(iid, 0) > 0),
            "dup_top10": (sum(1 for iid in top10 if name_count.get(normalize_arabic(names.get(iid, "")), 0) > 1)
                          / max(len(top10), 1)),
        })
        if (idx + 1) % 5000 == 0:
            print(f"  {idx + 1}/{len(qids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(SCRATCH / "autopsy_per_query.parquet", index=False)

    total_loss = df["loss"].sum()
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("\n=== GLOBAL DECOMPOSITION ===")
    print(f"queries: {len(df)}")
    print(f"mean nDCG@10          : {df['ndcg'].mean():.4f}")
    print(f"mean pool-oracle      : {df['oracle'].mean():.4f}")
    print(f"mean ordering loss    : {df['order_loss'].mean():.4f}  (oracle - actual; addressable without pool changes)")
    print(f"mean pool loss        : {df['pool_loss'].mean():.4f}  (1 - oracle; needs pool/recall changes)")

    print("\n=== LOSS CONCENTRATION (queries sorted by loss) ===")
    d = df.sort_values("loss", ascending=False).reset_index(drop=True)
    cum = d["loss"].cumsum() / total_loss
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(len(d) * frac)
        print(f"top {frac:4.0%} of queries hold {cum.iloc[k - 1]:6.1%} of total loss")
    for thr in (0.0, 0.3, 0.5):
        sub = df[df["ndcg"] <= thr]
        print(f"queries with nDCG <= {thr:.1f}: {len(sub):6d} ({len(sub) / len(df):5.1%}), "
              f"loss share {sub['loss'].sum() / total_loss:5.1%}, mean oracle there {sub['oracle'].mean():.4f}")

    def bucket_table(col, buckets=None, labels=None):
        x = df.copy()
        if buckets is not None:
            x["bucket"] = pd.cut(x[col], bins=buckets, labels=labels, include_lowest=True)
        else:
            x["bucket"] = x[col]
        g = x.groupby("bucket", observed=True).agg(
            n=("qid", "size"), mean_ndcg=("ndcg", "mean"), mean_oracle=("oracle", "mean"),
            mean_order_loss=("order_loss", "mean"), loss_sum=("loss", "sum"))
        g["loss_share"] = g["loss_sum"] / total_loss
        g["order_share_of_loss"] = (x.groupby("bucket", observed=True)["order_loss"].sum() / total_loss)
        return g.drop(columns="loss_sum")

    print("\n=== BY TOKEN COUNT ===")
    print(bucket_table("n_tok", buckets=[0, 1, 2, 3, 4, 100], labels=["1", "2", "3", "4", "5+"]))
    print("\n=== BY N JUDGED POSITIVES ===")
    print(bucket_table("n_pos", buckets=[0, 2, 5, 10, 20, 10000], labels=["1-2", "3-5", "6-10", "11-20", "21+"]))
    print("\n=== BY CE CONFIDENCE MARGIN (T=2.3, top1-top5) ===")
    print(bucket_table("conf", buckets=[-1, 0.02, 0.05, 0.10, 0.25, 0.5, 2], labels=["<.02", ".02-.05", ".05-.10", ".10-.25", ".25-.50", ">.50"]))
    print("\n=== BY BAND TIER (wide = conf < 0.10) ===")
    print(bucket_table("wide_band"))
    print("\n=== BY FILLER PRESENCE ===")
    print(bucket_table("has_filler"))
    print("\n=== BY DIGIT PRESENCE ===")
    print(bucket_table("has_digit"))
    print("\n=== BY LATIN PRESENCE ===")
    print(bucket_table("has_latin"))
    print("\n=== BY BM25/DENSE TOP-10 OVERLAP ===")
    print(bucket_table("overlap10", buckets=[-1, 0.0, 0.2, 0.4, 0.6, 1.0], labels=["0", ".1-.2", ".3-.4", ".5-.6", ".7-1"]))
    print("\n=== BY DUP-NAME SHARE IN LIVE TOP-10 ===")
    print(bucket_table("dup_top10", buckets=[-1, 0.0, 0.3, 0.7, 1.0], labels=["0", ".1-.3", ".4-.7", ".8-1"]))

    print("\ndone; per-query parquet saved.", flush=True)


if __name__ == "__main__":
    main()
