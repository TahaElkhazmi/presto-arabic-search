import json
import math
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import BENCH, DATA, load_jsonl, normalize_arabic
from eval_ndcg import evaluate_run, load_candidates
from reranker_common import format_document, format_query
from stacker_features import char_ngrams, containment

def _id_key(iid: str):
    try:
        return int(iid)
    except ValueError:
        return iid

def load_cache(path: Path) -> dict[tuple[str, str], float]:
    df = pd.read_parquet(path)
    return {(q, i): s for q, i, s in zip(df["qid"], df["iid"], df["score"])}

def main():
    # 1. Create dev_unseen2000.txt
    splits_dir = Path("splits")
    unseen_path = splits_dir / "dev_unseen2000.txt"
    rand2000_path = splits_dir / "dev_rand2000.txt"
    dev_qids_path = splits_dir / "dev_qids.txt"
    
    with open(dev_qids_path, "r", encoding="utf-8") as f:
        dev_all = [line.strip() for line in f if line.strip()]
        
    with open(rand2000_path, "r", encoding="utf-8") as f:
        rand2000 = {line.strip() for line in f if line.strip()}
        
    if not unseen_path.exists():
        available = sorted([q for q in dev_all if q not in rand2000])
        random.seed(42) # Deterministic
        unseen = random.sample(available, 2000)
        with open(unseen_path, "w", encoding="utf-8") as f:
            for q in unseen:
                f.write(q + "\n")
        print(f"Created dev_unseen2000.txt (2000 queries, 0 overlap with rand2000)")
    
    with open(unseen_path, "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    qids_set = set(qids)
    
    print(f"Loaded {len(qids)} queries for grid search.")
    
    # 2. Load required data for logic
    run_bm25 = {q["query_id"]: [i["item_id"] for i in q["retrieved"]] for q in load_jsonl("runs/bm25_all.jsonl")}
    run_dense = {q["query_id"]: [i["item_id"] for i in q["retrieved"]] for q in load_jsonl("runs/dense_lora_v1_all.jsonl")}
    
    # build pools
    depths = [100, 100]
    weights = [2.0, 1.0]
    rrf_k = 10
    
    pools = {}
    priors = {}
    for qid in qids:
        seen = {}
        pool = []
        for run, depth, w in zip([run_bm25, run_dense], depths, weights):
            for rank, iid in enumerate(run.get(qid, [])[:depth], start=1):
                if iid not in seen:
                    seen[iid] = 0.0
                    pool.append(iid)
                seen[iid] += w / (rrf_k + rank)
        pools[qid] = pool
        priors[qid] = seen

    print("Pools built.")
    
    # cache
    cache_path = DATA / "cache" / "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16.parquet"
    print("Loading cache...")
    cache = load_cache(cache_path)
    
    def rescale(raw: float, T: float) -> float:
        s = min(max(raw, 1e-12), 1.0 - 1e-12)
        logit = math.log(s / (1.0 - s))
        return 1.0 / (1.0 + math.exp(-logit / T))
    
    ce = {}
    for qid in qids:
        for iid in pools[qid]:
            raw_s = cache.get((qid, iid), 0.0)
            ce[(qid, iid)] = rescale(raw_s, 2.3)
            
    print("Cache loaded and T=2.3 applied.")
    
    # load queries/names for cont3
    queries = {str(q["query_id"]): q["query"] for q in load_jsonl(BENCH / "queries.jsonl")}
    names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}
    
    DIALECT_FILLERS = {normalize_arabic(w) for w in (
        "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
        "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

    def query_text_for_cont3(qid: str) -> str:
        qn = normalize_arabic(queries[qid])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        return " ".join(kept) if kept else qn
    
    qgrams = {qid: char_ngrams(query_text_for_cont3(qid), 3) for qid in qids}
    ngrams = {}
    
    def cont3(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in ngrams:
            ngrams[name] = char_ngrams(normalize_arabic(name), 3)
        return containment(qgrams[qid], ngrams[name])

    def group_key(qid: str):
        return lambda x: (-cont3(qid, x[1]), _id_key(x[1]))

    def eps_resort(scored: list[tuple[float, str]], eps: float, key) -> list[str]:
        out = []
        group = []
        for s, iid in scored:
            if group and (group[-1][0] - s) > eps:
                group.sort(key=key)
                out.extend(i for _, i in group)
                group = []
            group.append((s, iid))
        group.sort(key=key)
        out.extend(i for _, i in group)
        return out
        
    qrels = load_candidates(BENCH / "candidates.jsonl")
    
    # 3. Grid search
    lams = [0.35, 0.40, 0.45, 0.50]
    eps_vals = [0.010, 0.015, 0.020]
    eps_hi_vals = [0.06, 0.08, 0.10]
    conf_t_vals = [0.03, 0.05, 0.08, 0.10]
    
    best_score = -1
    best_params = None
    
    rrf_max = sum(weights) / (rrf_k + 1)
    
    configs = []
    for l in lams:
        for e in eps_vals:
            for e_hi in eps_hi_vals:
                for ct in conf_t_vals:
                    configs.append((l, e, e_hi, ct))
    
    print(f"Running grid search over {len(configs)} configs...")
    for lam, eps, eps_hi, conf_t in tqdm(configs):
        
        conf = {}
        for qid in qids:
            ces = sorted((ce[(qid, iid)] for iid in pools[qid]), reverse=True)
            conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0
            
        run = {}
        for qid in qids:
            scored = [
                (lam * ce[(qid, iid)] + (1 - lam) * priors[qid][iid] / rrf_max, iid)
                for iid in pools[qid]
            ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            
            cur_eps = eps
            if conf.get(qid, 1.0) < conf_t:
                cur_eps = eps_hi
                
            run[qid] = eps_resort(scored, cur_eps, group_key(qid))[:10]
            
        res = evaluate_run(run, qrels)
        score = res['nDCG@10']
        
        if score > best_score:
            best_score = score
            best_params = (lam, eps, eps_hi, conf_t)
            
    print("\n--- GRID SEARCH RESULTS ---")
    print(f"Best nDCG@10: {best_score:.4f}")
    print(f"Best Params:")
    print(f"  lam:    {best_params[0]:.2f}")
    print(f"  eps:    {best_params[1]:.3f}")
    print(f"  eps_hi: {best_params[2]:.2f}")
    print(f"  conf_t: {best_params[3]:.2f}")
    
    # Save best parameters to a file for easy reading
    with open("best_v6warm_params.json", "w") as f:
        json.dump({
            "lam": best_params[0],
            "eps": best_params[1],
            "eps_hi": best_params[2],
            "conf_t": best_params[3],
            "score": best_score
        }, f, indent=2)

if __name__ == "__main__":
    main()
