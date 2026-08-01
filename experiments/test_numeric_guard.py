import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import pandas as pd
from common import BENCH, DATA, MODELS, RUNS, load_jsonl, normalize_arabic, select_queries, setup_utf8
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from rerank import build_pools, load_cache, _id_key

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
from stacker_features import char_ngrams, containment, parse_numeric, numeric_features

def main():
    setup_utf8()
    runs = [RUNS / "bm25_all.jsonl", RUNS / "dense_lora_v1_all.jsonl"]
    depths = [100, 100]
    weights = [2.0, 1.0]
    rrf_k = 10
    qids_path = Path("splits/dev_rand2000.txt")
    queries_path = BENCH / "queries.jsonl"
    
    run_files = [load_run_jsonl(path) for path in runs]
    queries = select_queries(load_jsonl(queries_path), qids_path, 0)
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    pools, priors = build_pools(run_files, depths, weights, rrf_k, qids)
    
    cache_path = DATA / "cache" / "rerank_qwen3-reranker-arabic-ecom-rerank_v2_len160_bfloat16.parquet"
    cache = load_cache(cache_path)
    
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    
    # Baseline cont3 parameters
    DIALECT_FILLERS = {normalize_arabic(w) for w in (
        "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
        "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

    def query_text_baseline(qid: str) -> str:
        qn = normalize_arabic(qtext[qid])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        return " ".join(kept) if kept else qn

    qgrams = {qid: char_ngrams(query_text_baseline(qid), 3) for qid in qids}
    ngrams = {}
    
    def cont3(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in ngrams:
            ngrams[name] = char_ngrams(normalize_arabic(name), 3)
        return containment(qgrams[qid], ngrams[name])

    # Pre-parse query and product numbers
    print("Pre-parsing numeric specifications...", flush=True)
    q_nums = {qid: parse_numeric(normalize_arabic(qtext[qid])) for qid in qids}
    
    # Pre-parse name numbers for all candidates in the pools
    all_pids = set()
    for p in pools.values():
        all_pids.update(p)
    n_nums = {pid: parse_numeric(normalize_arabic(names[pid])) for pid in all_pids}
    print(f"Pre-parsed {len(q_nums)} queries and {len(n_nums)} unique products.", flush=True)

    def get_num_match_flag(qid: str, iid: str) -> int:
        qnum = q_nums[qid]
        nnum = n_nums[iid]
        q_bare, q_units = qnum
        if not q_bare and not q_units:
            return 0  # query specifies no numbers
        
        num_match_count, num_conflict, unit_conflict = numeric_features(qnum, nnum)
        if unit_conflict or num_conflict:
            return -1  # conflict
        if num_match_count > 0:
            return 1   # exact match
        return 0       # neutral

    rrf_max = sum(weights) / (rrf_k + 1)
    
    conf = {}
    for qid in qids:
        ces = sorted((cache[(qid, iid)] for iid in pools[qid]), reverse=True)
        conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0

    def blended(lam: float, eps_hi: float, use_num_guard: bool, eps: float = 0.015, conf_t: float = 0.10) -> dict[str, list[str]]:
        def group_key(qid: str):
            if use_num_guard:
                return lambda x: (-get_num_match_flag(qid, x[1]), -cont3(qid, x[1]), _id_key(x[1]))
            return lambda x: (-cont3(qid, x[1]), _id_key(x[1]))
            
        out = {}
        for qid in qids:
            scored = [
                (lam * cache[(qid, iid)] + (1 - lam) * priors[qid][iid] / rrf_max, iid)
                for iid in pools[qid]
            ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            e = eps
            if conf.get(qid, 1.0) < conf_t:
                e = eps_hi
            out[qid] = (eps_resort(scored, e, group_key(qid)) if e > 0
                        else [iid for _, iid in scored])
        return out

    qrels = load_candidates(BENCH / "candidates.jsonl")

    # 1. Run baseline (no num guard, lam 0.40, eps_hi 0.080)
    run_base = {qid: ids[:10] for qid, ids in blended(0.40, 0.080, False).items()}
    res_base = evaluate_run(run_base, qrels)
    print(f"\nBaseline (No Guard) nDCG@10: {res_base['nDCG@10']:.6f}")

    # 2. Run with num guard, same hyperparameters
    run_guard = {qid: ids[:10] for qid, ids in blended(0.40, 0.080, True).items()}
    res_guard = evaluate_run(run_guard, qrels)
    print(f"Num Guard (Same Params) nDCG@10: {res_guard['nDCG@10']:.6f}")
    print(f"Delta: {res_guard['nDCG@10'] - res_base['nDCG@10']:.6f}")

    # 3. Grid Search over lam and eps_hi with Num Guard
    print("\nStarting Hyperparameter Grid Search (Num Guard active)...", flush=True)
    import numpy as np
    lams = np.arange(0.35, 0.4501, 0.005)
    eps_his = np.arange(0.05, 0.1201, 0.005)
    
    best_score = -1.0
    best_lam = 0.0
    best_eps_hi = 0.0
    
    for lam in lams:
        for eps_hi in eps_his:
            run = {qid: ids[:10] for qid, ids in blended(lam, eps_hi, True).items()}
            res = evaluate_run(run, qrels)
            score = res['nDCG@10']
            if score > best_score:
                best_score = score
                best_lam = lam
                best_eps_hi = eps_hi
                print(f"New Best: lam={lam:.3f}, eps_hi={eps_hi:.3f} | nDCG@10={score:.6f} | delta={score - res_base['nDCG@10']:.6f}")

    print(f"\nBest Config: lam={best_lam:.3f}, eps_hi={best_eps_hi:.3f}")
    print(f"Best nDCG@10: {best_score:.6f}")
    print(f"Best Delta: {best_score - res_base['nDCG@10']:.6f}")

    # 4. Diagnostics: Print first 5 queries containing numbers where ranking changed
    best_run = {qid: ids[:10] for qid, ids in blended(best_lam, best_eps_hi, True).items()}
    import eval_ndcg
    print("\n--- Diagnostic check on numeric queries ---")
    count = 0
    for qid in qids:
        q_rels = qrels[qid]
        ndcg_base = eval_ndcg.ndcg_at_k(run_base[qid], q_rels, 10)
        ndcg_best = eval_ndcg.ndcg_at_k(best_run[qid], q_rels, 10)
        q_bare, q_units = q_nums[qid]
        if (q_bare or q_units) and ndcg_best != ndcg_base:
            diff = ndcg_best - ndcg_base
            print(f"QID: {qid} | Query: {qtext[qid]} | Delta: {diff:+.6f}")
            print(f"  Base top-5: {[names[i] for i in run_base[qid][:5]]}")
            print(f"  Best top-5: {[names[i] for i in best_run[qid][:5]]}")
            count += 1
            if count >= 5:
                break

if __name__ == "__main__":
    main()
