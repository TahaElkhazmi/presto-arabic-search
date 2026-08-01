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
from stacker_features import char_ngrams, containment

ROOT_AL_WORDS = {
    "العاب", "الوان", "الواح", "الغاز", "البان", "الترا", "الكتروني", "الكترونيه",
    "الماني", "المانيه", "الومنيوم", "المونيوم", "اليفه", "اليف", "الاف",
    "اليكسا", "الين", "اليت", "اليكسير", "الكسير", "الفاظ", "الفا",
    "الباين", "التهابات", "التيميت", "الوه", "اليريو", "الوه", "الايف",
    "الور"
}

def strip_arabic_prefixes(text: str) -> str:
    tokens = text.split()
    cleaned = []
    for t in tokens:
        if t in ROOT_AL_WORDS:
            cleaned.append(t)
            continue
        # Definite articles and coordinate/preposition prefixes
        if t.startswith("ال") and len(t) >= 4:
            t = t[2:]
        elif t.startswith("بال") and len(t) >= 5:
            t = t[3:]
        elif t.startswith("وال") and len(t) >= 5:
            t = t[3:]
        elif t.startswith("لل") and len(t) >= 4:
            t = t[2:]
        cleaned.append(t)
    return " ".join(cleaned)

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

    # 1. Baseline logic
    def query_text_baseline(qid: str) -> str:
        qn = normalize_arabic(qtext[qid])
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        return " ".join(kept) if kept else qn

    qgrams_baseline = {qid: char_ngrams(query_text_baseline(qid), 3) for qid in qids}
    ngrams_baseline = {}
    
    def cont3_baseline(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in ngrams_baseline:
            ngrams_baseline[name] = char_ngrams(normalize_arabic(name), 3)
        return containment(qgrams_baseline[qid], ngrams_baseline[name])

    # 2. Experiment: Prefix-stripping logic
    def query_text_exp(qid: str) -> str:
        qn = normalize_arabic(qtext[qid])
        # Strip fillers first
        kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
        clean_q = " ".join(kept) if kept else qn
        # Strip prefixes
        return strip_arabic_prefixes(clean_q)

    qgrams_exp = {qid: char_ngrams(query_text_exp(qid), 3) for qid in qids}
    ngrams_exp = {}
    
    def cont3_exp(qid: str, iid: str) -> float:
        name = names[iid]
        if name not in ngrams_exp:
            norm_name = normalize_arabic(name)
            # Strip prefixes from product name too
            clean_name = strip_arabic_prefixes(norm_name)
            ngrams_exp[name] = char_ngrams(clean_name, 3)
        return containment(qgrams_exp[qid], ngrams_exp[name])

    # Re-use evaluation structure
    rrf_max = sum(weights) / (rrf_k + 1)
    
    conf = {}
    for qid in qids:
        ces = sorted((cache[(qid, iid)] for iid in pools[qid]), reverse=True)
        conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0

    def blended(lam: float, cont3_func, eps: float = 0.015, eps_hi: float = 0.080, conf_t: float = 0.10) -> dict[str, list[str]]:
        def group_key(qid: str):
            return lambda x: (-cont3_func(qid, x[1]), _id_key(x[1]))
            
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
    
    # Eval baseline
    run_base = {qid: ids[:10] for qid, ids in blended(0.40, cont3_baseline).items()}
    res_base = evaluate_run(run_base, qrels)
    print(f"Baseline nDCG@10: {res_base['nDCG@10']:.6f}")
    
    # Eval prefix stripping
    run_exp = {qid: ids[:10] for qid, ids in blended(0.40, cont3_exp).items()}
    res_exp = evaluate_run(run_exp, qrels)
    print(f"Prefix-Stripping nDCG@10: {res_exp['nDCG@10']:.6f}")
    print(f"Delta: {res_exp['nDCG@10'] - res_base['nDCG@10']:.6f}")

    # Print first 10 regressed queries
    import eval_ndcg
    print("\n--- Regressed Queries ---")
    count = 0
    for qid in qids:
        q_rels = qrels[qid]
        ndcg_base = eval_ndcg.ndcg_at_k(run_base[qid], q_rels, 10)
        ndcg_exp = eval_ndcg.ndcg_at_k(run_exp[qid], q_rels, 10)
        if ndcg_exp < ndcg_base:
            print(f"QID: {qid} | Query: {qtext[qid]}")
            print(f"  Base top-5: {[names[i] for i in run_base[qid][:5]]}")
            print(f"  Exp top-5:  {[names[i] for i in run_exp[qid][:5]]}")
            # Show original and stripped forms of query and top products
            print(f"  Query stripped: {query_text_exp(qid)}")
            for i in run_base[qid][:5]:
                print(f"    Prod '{names[i]}' -> normalized: '{normalize_arabic(names[i])}' -> stripped: '{strip_arabic_prefixes(normalize_arabic(names[i]))}'")
            count += 1
            if count >= 10:
                break

if __name__ == "__main__":
    main()
