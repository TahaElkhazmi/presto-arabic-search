"""Tabular LightGBM LambdaMART stacker over frozen lora_v1 candidate pools.

Trains on TRAINING queries only (rule 2.4.b): pools are the fused BM25 +
dense_lora_v1 union over the training-query runs, labels are the provided
positives, and non-positive candidates at fused survivor rank <= 3 get row
weight 0 (the 2026-07-18 contamination finding: 38% of top-3 pool slots are
unlabeled positives, so training them as negatives repeats the v3 failure).

Feature schema: 30 features (see stacker_features.py). CE features come from a
rerank score cache parquet (--ce-cache); with --dry-run and no cache, a dummy CE
column is used so the whole pipeline (features, weights, groups, booster) can be
verified before the ~4h GPU CE-scoring pass exists.

At inference the same feature builder runs over the benchmark 100+100 pools and
the booster score replaces the linear blend inside the low-id tie-break sort.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import BENCH, DATA, load_jsonl, normalize_arabic, setup_utf8
from eval_ndcg import load_run_jsonl
from rerank import _id_key, build_pools
from stacker_features import (
    NGRAM_NS,
    build_ngram_cache,
    build_numeric_cache,
    containment,
    jaccard,
    numeric_features,
)
from common import tokenize

FEATURES = [
    "bm25_rank", "bm25_rr", "dense_rank", "dense_rr", "in_bm25", "in_dense",
    "rrf_prior_norm", "ce_sigmoid", "ce_logit", "ce_rank_in_pool", "ce_margin_to_top",
    "tok_containment", "tok_jaccard", "exact_norm_match", "n_query_tokens",
    "n_name_tokens", "len_ratio", "num_match_count", "num_conflict", "unit_conflict",
    "id_int", "name_dup_count", "is_lowest_dup_id", "pool_size",
    "jaccard_char2", "jaccard_char3", "jaccard_char4",
    "containment_char2", "containment_char3", "containment_char4",
]
MONOTONE = [(-1 if f == "id_int" else 0) for f in FEATURES]


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return float(np.clip(np.log(p / (1 - p)), -12, 12))


def build_rows(qids, qtext, pools, priors, ce_lookup, names_norm, name_ids, corpus_id_int):
    q_norm = {q: normalize_arabic(qtext[q]) for q in qids}
    ng_cache = build_ngram_cache(list(q_norm.values()) + list(names_norm.values()))
    nu_cache = build_numeric_cache(list(q_norm.values()) + list(names_norm.values()))
    qtok = {q: set(tokenize(qtext[q])) for q in qids}

    rows = []
    for qid in qids:
        pool = pools[qid]
        prior = priors[qid]
        # fused survivor rank by RRF prior (1-based) for the contamination mask
        order = sorted(pool, key=lambda i: (-prior[i], _id_key(i)))
        surv_rank = {iid: r for r, iid in enumerate(order, start=1)}
        ces = {iid: ce_lookup(qid, iid) for iid in pool}
        ce_top = max(ces.values()) if ces else 0.0
        ce_sorted = sorted(pool, key=lambda i: -ces[i])
        ce_rank = {iid: r for r, iid in enumerate(ce_sorted, start=1)}
        qn = q_norm[qid]
        qg = ng_cache[qn]
        qnum = nu_cache[qn]
        qt = qtok[qid]
        for iid in pool:
            nm_norm = names_norm[iid]
            ng = ng_cache[nm_norm]
            nt = set(tokenize(nm_norm))
            ce = ces[iid]
            inter_tok = len(qt & nt)
            br = bm25_rank_of(qid, iid)
            dr = dense_rank_of(qid, iid)
            feat = {
                "bm25_rank": br,
                "bm25_rr": 1.0 / (10 + br) if br != 999 else 0.0,
                "dense_rank": dr,
                "dense_rr": 1.0 / (10 + dr) if dr != 999 else 0.0,
                "in_bm25": int(br != 999),
                "in_dense": int(dr != 999),
                "rrf_prior_norm": prior[iid],
                "ce_sigmoid": ce,
                "ce_logit": logit(ce),
                "ce_rank_in_pool": ce_rank[iid],
                "ce_margin_to_top": logit(ce_top) - logit(ce),
                "tok_containment": inter_tok / max(len(qt), 1),
                "tok_jaccard": inter_tok / max(len(qt | nt), 1),
                "exact_norm_match": int(qn == nm_norm),
                "n_query_tokens": len(qt),
                "n_name_tokens": len(nt),
                "len_ratio": len(nm_norm) / max(len(qn), 1),
                "id_int": corpus_id_int[iid],
                "name_dup_count": len(name_ids[nm_norm]),
                "is_lowest_dup_id": int(iid == min(name_ids[nm_norm], key=_id_key)),
                "pool_size": len(pool),
                "_qid": qid, "_iid": iid, "_surv_rank": surv_rank[iid],
            }
            mc, nc, uc = numeric_features(qnum, nu_cache[nm_norm])
            feat["num_match_count"], feat["num_conflict"], feat["unit_conflict"] = mc, nc, uc
            for n in NGRAM_NS:
                feat[f"jaccard_char{n}"] = jaccard(qg[n], ng[n])
                feat[f"containment_char{n}"] = containment(qg[n], ng[n])
            rows.append(feat)
    return pd.DataFrame(rows)


# rank lookups filled by main() closures (kept module-level for build_rows clarity)
bm25_rank_of = dense_rank_of = None


def main():
    global bm25_rank_of, dense_rank_of
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+",
                   default=[Path("runs/bm25_train82k.jsonl"), Path("runs/dense_lora_v1_train82k.jsonl")])
    p.add_argument("--depths", type=int, nargs="+", default=[100, 100])
    p.add_argument("--weights", type=float, nargs="+", default=[2.0, 1.0])
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--queries", type=Path, default=DATA / "teacher" / "train_queries.jsonl")
    p.add_argument("--ce-cache", type=Path, default=None, help="rerank score parquet (qid,iid,score)")
    p.add_argument("--n-queries", type=int, default=0, help="0=all; subsample for dry run")
    p.add_argument("--dry-run", action="store_true", help="small slice + dummy CE, verify pipeline")
    p.add_argument("--out", type=Path, default=DATA.parent / "models" / "stacker_v1.txt")
    p.add_argument("--label-union", action="store_true",
                   help="merge positives/train.parquet into the label set "
                        "(coverage 10%% -> ~17%%; that file's noise sank retriever v2, gate hard)")
    args = p.parse_args()

    n_q = 200 if args.dry_run else args.n_queries
    queries = load_jsonl(args.queries)
    if n_q:
        queries = queries[:n_q]
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}

    run_files = [load_run_jsonl(path) for path in args.runs]
    bm25_pos = {q: {iid: r for r, iid in enumerate(run_files[0].get(q, []), 1)} for q in qids}
    dense_pos = {q: {iid: r for r, iid in enumerate(run_files[1].get(q, []), 1)} for q in qids}
    bm25_rank_of = lambda q, i: bm25_pos[q].get(i, 999)
    dense_rank_of = lambda q, i: dense_pos[q].get(i, 999)

    pools, priors = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)

    corpus = load_jsonl(BENCH / "corpus.jsonl")
    names_raw = {str(r["item_id"]): r["name"] for r in corpus}
    names_norm = {iid: normalize_arabic(nm) for iid, nm in names_raw.items()}
    corpus_id_int = {iid: _id_key(iid) if isinstance(_id_key(iid), int) else int(iid.split("_")[1])
                     for iid in names_raw}
    name_ids: dict[str, list[str]] = {}
    for iid, nmn in names_norm.items():
        name_ids.setdefault(nmn, []).append(iid)

    # labels from provided positives (training queries only)
    tri = pd.read_parquet(DATA.parent / "data" / "training" / "pairs_with_negatives" / "train.parquet")
    pos_by_text: dict[str, set] = {}
    for q, pos in zip(tri["user_query"], tri["positive_product_name"]):
        pos_by_text.setdefault(q, set()).add(normalize_arabic(pos))
    if args.label_union:
        ptr = pd.read_parquet(DATA.parent / "data" / "training" / "positives" / "train.parquet")
        for q, pos in zip(ptr["user_query"], ptr["positive_product_name"]):
            pos_by_text.setdefault(q, set()).add(normalize_arabic(pos))
        print("label union: positives/train merged into the label set")

    if args.ce_cache and args.ce_cache.exists():
        cdf = pd.read_parquet(args.ce_cache)
        ce_map = {(str(q), str(i)): float(s) for q, i, s in zip(cdf["qid"], cdf["iid"], cdf["score"])}
        ce_lookup = lambda q, i: ce_map.get((q, i), 0.5)
        print(f"CE cache {args.ce_cache.name}: {len(ce_map)} scores")
    else:
        assert args.dry_run, "real run needs --ce-cache; only --dry-run may use dummy CE"
        rng = np.random.default_rng(0)
        dummy = {}
        ce_lookup = lambda q, i: dummy.setdefault((q, i), float(rng.uniform(0, 1)))
        print("DRY RUN: dummy random CE column")

    print(f"building features for {len(qids)} queries...")
    df = build_rows(qids, qtext, pools, priors, ce_lookup, names_norm, name_ids, corpus_id_int)
    df["label"] = [int(names_norm[i] in pos_by_text.get(qtext[q], set()))
                   for q, i in zip(df["_qid"], df["_iid"])]
    df["weight"] = np.where((df["label"] == 0) & (df["_surv_rank"] <= 3), 0.0, 1.0)
    print(f"rows {len(df)}, positives {df['label'].sum()}, "
          f"weight-0 masked {(df['weight'] == 0).sum()}")

    # query-level split for early stopping (train fits, valid stops)
    rng = np.random.default_rng(42)
    val_qids = set(rng.choice(qids, size=max(1, len(qids) // 10), replace=False))
    tr = df[~df["_qid"].isin(val_qids)].reset_index(drop=True)
    va = df[df["_qid"].isin(val_qids)].reset_index(drop=True)

    import lightgbm as lgb

    def dataset(frame, ref=None):
        frame = frame.sort_values("_qid", kind="stable")
        grp = frame.groupby("_qid", sort=False).size().to_numpy()
        return lgb.Dataset(
            frame[FEATURES].to_numpy(dtype=np.float64),
            label=frame["label"].to_numpy(),
            weight=frame["weight"].to_numpy(),
            group=grp,
            reference=ref,
            free_raw_data=False,
        ), frame

    dtr, _ = dataset(tr)
    dva, _ = dataset(va, ref=dtr)
    params = dict(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
        num_leaves=31, min_data_in_leaf=50, learning_rate=0.05,
        feature_fraction=0.9, monotone_constraints=MONOTONE,
        monotone_constraints_method="intermediate", verbosity=-1, seed=42,
    )
    print(f"LightGBM lambdarank: {len(FEATURES)} features, monotone id_int=-1 (intermediate), "
          f"train groups {len(tr['_qid'].unique())}, valid groups {len(va['_qid'].unique())}")
    booster = lgb.train(
        params, dtr, num_boost_round=(20 if args.dry_run else 500),
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best = booster.best_score.get("valid", {}).get("ndcg@10")
    print(f"booster trained: best_iter={booster.best_iteration}, valid ndcg@10={best}")
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])[:8]
    print("top gain features:", ", ".join(f"{f}={g:.0f}" for f, g in imp))
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(args.out))
        print(f"saved {args.out}")
    else:
        print("DRY RUN OK: features + weights + groups + booster all verified")


if __name__ == "__main__":
    main()
