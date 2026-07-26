import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import pandas as pd
from common import BENCH, DATA, MODELS, RUNS, load_jsonl, normalize_arabic, select_queries, setup_utf8, write_run_csv
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from rerank import build_pools, load_cache, _id_key
from stacker_features import char_ngrams, containment, parse_numeric, numeric_features

def eps_resort(scored: list[tuple[float, str]], eps: float, key) -> list[str]:
    """Chain-group descending scores whose consecutive gap is <= eps; each group
    sorts by `key` (the in-band prior beats sub-eps score noise)."""
    out: list[str] = []
    group: list[tuple[float, str]] = []
    for s, iid in scored:
        if group and (group[-1][0] - s) > eps:
            group.sort(key=key)
            out.extend(i for _, i in group)
            group = []
        group.append((s, iid))
    group.sort(key=key)
    out.extend(i for _, i in group)
    return out

def main():
    setup_utf8()
    runs = [RUNS / "bm25_all.jsonl", RUNS / "dense_lora_v1_all.jsonl"]
    depths = [100, 100]
    weights = [2.0, 1.0]
    rrf_k = 10
    
    # ------------------ 1. Full Validation Verification ------------------
    print("=== Step 1: Full Validation Verification ===", flush=True)
    val_qids_path = Path("splits/val_qids.txt")
    val_queries = select_queries(load_jsonl(BENCH / "queries.jsonl"), val_qids_path, 0)
    val_qids = [str(q["query_id"]) for q in val_queries]
    val_qtext = {str(q["query_id"]): q["query"] for q in val_queries}
    val_pools, val_priors = build_pools([load_run_jsonl(p) for p in runs], depths, weights, rrf_k, val_qids)
    
    cache_path = DATA / "cache" / "rerank_qwen3-reranker-arabic-ecom-rerank_v2_len160_bfloat16.parquet"
    cache = load_cache(cache_path)
    
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    
    DIALECT_FILLERS = {normalize_arabic(w) for w in (
        "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
        "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

    def query_text_baseline(qid: str, qtext_dict) -> str:
        qn = normalize_arabic(qtext_dict[qid])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        return " ".join(kept) if kept else qn

    # Pre-parse name numbers for all pool items
    print("Pre-parsing numeric specifications for validation...", flush=True)
    val_all_pids = set()
    for p in val_pools.values():
        val_all_pids.update(p)
    
    n_nums = {pid: parse_numeric(normalize_arabic(names[pid])) for pid in val_all_pids}
    val_q_nums = {qid: parse_numeric(normalize_arabic(val_qtext[qid])) for qid in val_qids}
    
    val_qgrams = {qid: char_ngrams(query_text_baseline(qid, val_qtext), 3) for qid in val_qids}
    val_ngrams = {}
    
    def val_cont3(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in val_ngrams:
            val_ngrams[name] = char_ngrams(normalize_arabic(name), 3)
        return containment(val_qgrams[qid], val_ngrams[name])

    def get_num_match_flag(qid: str, iid: str, q_nums_dict) -> int:
        qnum = q_nums_dict[qid]
        nnum = n_nums[iid]
        q_bare, q_units = qnum
        if not q_bare and not q_units:
            return 0
        
        num_match_count, num_conflict, unit_conflict = numeric_features(qnum, nnum)
        if unit_conflict or num_conflict:
            return -1
        if num_match_count > 0:
            return 1
        return 0

    rrf_max = sum(weights) / (rrf_k + 1)
    
    val_conf = {}
    for qid in val_qids:
        ces = sorted((cache[(qid, iid)] for iid in val_pools[qid]), reverse=True)
        val_conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0

    def blended_eval(lam: float, eps_hi: float, use_num_guard: bool, qids_list, pools_dict, priors_dict, conf_dict, cont3_func, q_nums_dict) -> dict[str, list[str]]:
        def group_key(qid: str):
            if use_num_guard:
                return lambda x: (-get_num_match_flag(qid, x[1], q_nums_dict), -cont3_func(qid, x[1]), _id_key(x[1]))
            return lambda x: (-cont3_func(qid, x[1]), _id_key(x[1]))
            
        out = {}
        for qid in qids_list:
            scored = [
                (lam * cache[(qid, iid)] + (1 - lam) * priors_dict[qid][iid] / rrf_max, iid)
                for iid in pools_dict[qid]
            ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e = 0.015
            if conf_dict.get(qid, 1.0) < 0.10:
                e = eps_hi
            out[qid] = (eps_resort(scored, e, group_key(qid)) if e > 0
                        else [iid for _, iid in scored])
        return out

    qrels = load_candidates(BENCH / "candidates.jsonl")

    # Eval baseline
    val_run_base = {qid: ids[:10] for qid, ids in blended_eval(0.40, 0.080, False, val_qids, val_pools, val_priors, val_conf, val_cont3, val_q_nums).items()}
    val_res_base = evaluate_run(val_run_base, qrels)
    print(f"Val Baseline nDCG@10: {val_res_base['nDCG@10']:.6f}")

    # Eval experiment (lam=0.415, eps_hi=0.110, num guard)
    val_run_exp = {qid: ids[:10] for qid, ids in blended_eval(0.415, 0.110, True, val_qids, val_pools, val_priors, val_conf, val_cont3, val_q_nums).items()}
    val_res_exp = evaluate_run(val_run_exp, qrels)
    print(f"Val Numeric Guard nDCG@10: {val_res_exp['nDCG@10']:.6f}")
    print(f"Val Delta: {val_res_exp['nDCG@10'] - val_res_base['nDCG@10']:.6f}")

    # ------------------ 2. Full Benchmark Submission ------------------
    print("\n=== Step 2: Full Benchmark Submission Generation ===", flush=True)
    all_queries = load_jsonl(BENCH / "queries.jsonl")
    all_qids = [str(q["query_id"]) for q in all_queries]
    all_qtext = {str(q["query_id"]): q["query"] for q in all_queries}
    
    print("Building full candidate pools and priors...", flush=True)
    all_pools, all_priors = build_pools([load_run_jsonl(p) for p in runs], depths, weights, rrf_k, all_qids)
    
    print("Pre-parsing numeric specs for full benchmark...", flush=True)
    all_pids = set()
    for p in all_pools.values():
        all_pids.update(p)
        
    # Merge existing n_nums and parse any remaining
    for pid in all_pids:
        if pid not in n_nums:
            n_nums[pid] = parse_numeric(normalize_arabic(names[pid]))
            
    all_q_nums = {qid: parse_numeric(normalize_arabic(all_qtext[qid])) for qid in all_qids}
    
    all_qgrams = {qid: char_ngrams(query_text_baseline(qid, all_qtext), 3) for qid in all_qids}
    all_ngrams = {}
    
    def all_cont3(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in all_ngrams:
            all_ngrams[name] = char_ngrams(normalize_arabic(name), 3)
        return containment(all_qgrams[qid], all_ngrams[name])

    all_conf = {}
    for qid in all_qids:
        ces = sorted((cache[(qid, iid)] for iid in all_pools[qid]), reverse=True)
        all_conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0

    print("Running blended reranking pipeline...", flush=True)
    full_run = blended_eval(0.415, 0.110, True, all_qids, all_pools, all_priors, all_conf, all_cont3, all_q_nums)
    
    RUNS.mkdir(exist_ok=True)
    out_csv = RUNS / "rerank_v2_lam415_adapt_hi110_dialect_numguard.csv"
    
    # Save submission file (top 10)
    write_run_csv(out_csv, {q: ids[:10] for q, ids in full_run.items()})
    
    print("\nSubmission File Verification:")
    print(f"File exists: {out_csv.exists()}")
    print(f"File size: {out_csv.stat().st_size} bytes")
    
    # Check row count
    sub_df = pd.read_csv(out_csv)
    print(f"Total rows in submission: {len(sub_df)}")
    print(f"Unique queries: {sub_df['query_id'].nunique()}")
    
    print(f"\n[SUCCESS] Final Submission File Path: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
