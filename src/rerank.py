"""Cross-encoder reranking of retrieval candidate pools with an RRF prior blend.

Pipeline per PHASE2_PLAN.md section 2:
  1. candidate pool per query = union of the top --depth ids from each input run
     (default intent: BM25 top-100 union dense top-100, ~200 dedup candidates).
  2. every (query, candidate) pair is scored by the seq-cls Qwen3 reranker with the
     official template + Arabic e-commerce instruction; scores are sigmoid in [0,1].
  3. final score = lam * ce + (1 - lam) * rrf / rrf_max, where the RRF prior is
     computed over the same runs at the same depths (default weights 2:1 and k=10,
     the tuned fusion settings) and rrf_max = sum(weights)/(k+1) scales it to ~[0,1].
     The prior guards against reranker blind spots; lam=1 is pure reranking.
  4. ties break by ascending integer id (the low-id prior from phase 1).

Scores are cached in data/cache/rerank_<model>_len<seq>_<dtype>.parquet keyed by
(query_id, item_id); lambda sweeps and repeat runs only pay for unseen pairs.

--lam-grid "0,0.2,..." skips run writing and instead reports nDCG@10 per lambda on
the selected query slice (tune on dev qids, verify once on val qids).
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from common import (BENCH, DATA, MODELS, RUNS, load_jsonl, normalize_arabic,
                    select_queries, setup_utf8, write_run_csv)
from eval_ndcg import evaluate_run, load_candidates, load_run_jsonl
from reranker_common import format_document, format_query


def _id_key(iid: str):
    try:
        return int(iid)
    except ValueError:
        return iid


def build_pools(
    runs: list[dict[str, list[str]]],
    depths: list[int],
    weights: list[float],
    rrf_k: int,
    qids: list[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Per query: deduped candidate union and the RRF prior over the same slices."""
    pools: dict[str, list[str]] = {}
    priors: dict[str, dict[str, float]] = {}
    for qid in qids:
        seen: dict[str, float] = {}
        pool: list[str] = []
        for run, depth, w in zip(runs, depths, weights):
            for rank, iid in enumerate(run.get(qid, [])[:depth], start=1):
                if iid not in seen:
                    seen[iid] = 0.0
                    pool.append(iid)
                seen[iid] += w / (rrf_k + rank)
        pools[qid] = pool
        priors[qid] = seen
    return pools, priors


def load_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {(q, i): s for q, i, s in zip(df["qid"], df["iid"], df["score"])}


def save_cache(path: Path, cache: dict[tuple[str, str], float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "qid": [q for q, _ in cache],
            "iid": [i for _, i in cache],
            "score": pd.array([cache[k] for k in cache], dtype="float32"),
        }
    )
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="candidate run jsonl files")
    p.add_argument("--depths", type=int, nargs="+", required=True, help="pool depth per run")
    p.add_argument("--weights", type=float, nargs="+", required=True, help="RRF prior weight per run")
    p.add_argument("--rrf-k", type=int, default=10)
    p.add_argument("--model", type=Path, required=True, help="seq-cls CrossEncoder model dir")
    p.add_argument("--cache-name", default=None,
                   help="override the score-cache stem (default rerank_<model>_len<seq>_<dtype>); "
                        "read/write a counterfactual cache without touching the live one")
    p.add_argument("--queries", type=Path, default=None,
                   help="query jsonl (default: benchmark queries.jsonl); e.g. training queries")
    p.add_argument("--qids", type=Path, default=None)
    p.add_argument("--score-only", action="store_true",
                   help="populate the score cache, write no runs and evaluate nothing")
    p.add_argument("--n-queries", type=int, default=0, help="0 = all (after qid filter)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=160)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--lam", type=float, default=0.5, help="blend: lam*ce + (1-lam)*rrf/rrf_max")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="rescale cached CE scores via delogit/T/resigmoid before use "
                        "(confidence margin AND the blend); the on-disk score cache is "
                        "never mutated, so any future T reuses the same raw scores")
    p.add_argument("--eps", type=float, default=0.0,
                   help="near-tie low-id re-sort: chain-group blend scores with gap <= eps, "
                        "sort each group by ascending id (0 = off)")
    p.add_argument("--eps-hi", type=float, default=0.0,
                   help="adaptive band: use this wider eps when the query's CE top1-top5 "
                        "margin is below --conf-t (0 = non-adaptive)")
    p.add_argument("--conf-t", type=float, default=0.10,
                   help="CE confidence threshold for --eps-hi")
    p.add_argument("--conf-topk", type=int, default=5,
                   help="confidence margin is CE top1 minus top-K (default 5). A sharper "
                        "margin (K=3) with a higher --conf-t routes a differently-shaped "
                        "set of queries into the wide band; measured +0.0009 benchmark "
                        "at K=3/conf-t 0.30 (FINDINGS 2026-07-26)")
    p.add_argument("--tie-key",
                   choices=["id", "cont3", "cont3prefix", "cont3pos", "cont3posft",
                            "cont3posftlt"],
                   default="id",
                   help="in-group order: ascending id; char-3gram query-in-name "
                        "containment desc then id (adaptive lexical fallback); or "
                        "cont3prefix, which inserts 'title starts with the stripped "
                        "query' between the two. Containment saturates at 1.0 as soon "
                        "as a title contains every query gram regardless of what else "
                        "it says, so on saturated queries the key was falling through "
                        "to ascending id across ~90 candidates that are NOT identical "
                        "strings (57.3 distinct titles on average). The prefix test is "
                        "binary, so it partitions the tie group and the low-id prior "
                        "still orders each half rather than being overwritten "
                        "(+0.0123 full benchmark, +0.0121 untouched); or cont3posft, "
                        "which appends FIRST-TOKEN position below match-position. "
                        "Match-position only fires on a CONTIGUOUS occurrence of the "
                        "query tokens, so every title holding them scattered scores 0 "
                        "and drops to raw id; first-token position asks the same "
                        "attribute-position question one level down and is worth "
                        "+0.0028 full benchmark, +0.0039 on the never-tuned slice")
    p.add_argument("--pos-bonus", type=float, default=0.0,
                   help="add this multiple of the token match-position score directly "
                        "to the BLEND, not just to the in-group tie-break. Match "
                        "position otherwise only orders candidates inside an eps "
                        "near-tie chain, so a strong signal is confined to a narrow "
                        "decision; a bounded additive term lets it also reorder "
                        "candidates whose blend scores are close but not in the same "
                        "chain. Requires --tie-key cont3pos or cont3posft. Swept to a "
                        "bracketed optimum at 0.35 (0.200 and 0.600 both worse)")
    p.add_argument("--idf-bonus", type=float, default=0.0,
                   help="add this multiple of the HIGHEST-IDF query token's position to "
                        "the blend. 'First token' is a positional accident of how the "
                        "query was typed; the most distinctive token is the one that "
                        "should locate the product, and it differs from the first token "
                        "on 49.5% of queries. Sub-gate alone (+0.0011) but part of a "
                        "near-additive trio")
    p.add_argument("--top-eps", type=float, default=0.0,
                   help="use this eps for the first --top-k-eps ranks instead of the "
                        "normal one. nDCG@10 weights rank 1 at 1.000 and rank 9 at 0.315, "
                        "so a re-sort near the top is worth about three times one near "
                        "the bottom, yet eps was uniform across positions. Sub-gate alone "
                        "(+0.0012)")
    p.add_argument("--top-k-eps", type=int, default=5,
                   help="how many leading ranks --top-eps applies to")
    p.add_argument("--ft-bonus", type=float, default=0.0,
                   help="same trick as --pos-bonus, applied to FIRST-TOKEN position: add "
                        "this multiple of it to the blend so it reorders candidates whose "
                        "scores are close but not in the same eps chain. Requires "
                        "--tie-key cont3posft. Note first-token is 3/(1+k) in [0,3] while "
                        "match-position is 1/(1+k) in [0,1], so the useful beta is ~7x "
                        "smaller; swept to a bracketed optimum at 0.05 (0.02 and 0.12 both "
                        "worse), worth +0.0020 full benchmark and +0.0026 on the "
                        "never-tuned slice")
    p.add_argument("--strip-fillers", action="store_true",
                   help="drop Libyan-dialect filler/request tokens from the query before "
                        "the cont3 key (ablated-safe set; +0.0032 benchmark, cont3-only)")
    p.add_argument("--lam-grid", default=None, help="comma list; evaluate instead of writing runs")
    p.add_argument("--name", default=None, help="output run name (required unless --lam-grid)")
    p.add_argument("--top-k", type=int, default=10, help="csv truncation depth")
    args = p.parse_args()
    assert len(args.runs) == len(args.depths) == len(args.weights), "one depth+weight per run"
    assert args.name or args.lam_grid or args.score_only, \
        "need --name to write runs, --lam-grid to evaluate, or --score-only"
    assert not args.pos_bonus or args.tie_key in ("cont3pos", "cont3posft", "cont3posftlt"), \
        "--pos-bonus needs a cont3pos* tie-key (it reuses that key's position score)"
    assert not args.idf_bonus or args.tie_key in ("cont3posft", "cont3posftlt"), \
        "--idf-bonus needs a cont3posft* tie-key (it shares that block's helpers)"
    assert not args.ft_bonus or args.tie_key in ("cont3posft", "cont3posftlt"), \
        "--ft-bonus needs --tie-key cont3posft (it reuses that key's first-token score)"

    run_files = [load_run_jsonl(path) for path in args.runs]
    queries = select_queries(load_jsonl(args.queries or (BENCH / "queries.jsonl")),
                             args.qids, args.n_queries)
    qids = [str(q["query_id"]) for q in queries]
    qtext = {str(q["query_id"]): q["query"] for q in queries}
    pools, priors = build_pools(run_files, args.depths, args.weights, args.rrf_k, qids)
    n_pairs = sum(len(v) for v in pools.values())
    print(f"{len(qids)} queries, {n_pairs} candidate pairs "
          f"(mean pool {n_pairs / max(len(qids), 1):.1f})")

    cache_stem = args.cache_name or f"rerank_{args.model.name}_len{args.max_seq}_{args.dtype}"
    cache_path = DATA / "cache" / f"{cache_stem}.parquet"
    cache = load_cache(cache_path)
    todo = [(qid, iid) for qid in qids for iid in pools[qid] if (qid, iid) not in cache]
    print(f"cache {cache_path.name}: {len(cache)} scores, {len(todo)} to compute")

    names: dict[str, str] = {}
    if todo or args.tie_key in ("cont3", "cont3prefix", "cont3pos", "cont3posft", "cont3posftlt"):
        names = {str(row["item_id"]): row["name"] for row in load_jsonl(BENCH / "corpus.jsonl")}

    if todo:
        from sentence_transformers.cross_encoder import CrossEncoder

        model = CrossEncoder(
            str(args.model),
            device="cuda",
            model_kwargs={"torch_dtype": getattr(torch, args.dtype)},
        )
        model[0].max_seq_length = args.max_seq
        pairs = [[format_query(qtext[qid]), format_document(names[iid])] for qid, iid in todo]
        t0 = time.time()
        scores = model.predict(pairs, batch_size=args.batch, show_progress_bar=True)
        print(f"scored {len(todo)} pairs in {(time.time() - t0) / 60:.1f} min")
        for (qid, iid), s in zip(todo, scores):
            cache[(qid, iid)] = float(s)
        save_cache(cache_path, cache)

    if args.score_only:
        print(f"score-only: cache now {len(cache)} scores, no runs written")
        return

    def rescale(raw: float, T: float) -> float:
        s = min(max(raw, 1e-12), 1.0 - 1e-12)
        logit = math.log(s / (1.0 - s))
        return 1.0 / (1.0 + math.exp(-logit / T))

    ce = cache if args.temperature == 1.0 else {
        k: rescale(v, args.temperature) for k, v in cache.items()
    }

    rrf_max = sum(args.weights) / (args.rrf_k + 1)

    conf: dict[str, float] = {}
    if args.eps_hi > 0:
        for qid in qids:
            ces = sorted((ce[(qid, iid)] for iid in pools[qid]), reverse=True)
            k = args.conf_topk
            conf[qid] = ces[0] - ces[k - 1] if len(ces) >= k else 1.0

    if args.tie_key in ("cont3", "cont3prefix", "cont3pos", "cont3posft", "cont3posftlt"):
        from stacker_features import char_ngrams, containment

        # ablated-safe Libyan-dialect filler/request tokens (post normalize_arabic);
        # inert-and-risky common words (من لو حق ...) deliberately excluded.
        DIALECT_FILLERS = {normalize_arabic(w) for w in (
            "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
            "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

        def query_text_for_cont3(qid: str) -> str:
            qn = normalize_arabic(qtext[qid])
            if not args.strip_fillers:
                return qn
            kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
            return " ".join(kept) if kept else qn  # never strip to empty

        qgrams = {qid: char_ngrams(query_text_for_cont3(qid), 3) for qid in qids}
        ngrams: dict[str, frozenset] = {}

        def cont3(qid: str, iid: str) -> float:
            name = names[iid]
            if name not in ngrams:
                ngrams[name] = char_ngrams(normalize_arabic(name), 3)
            return containment(qgrams[qid], ngrams[name])

        qtext_stripped = {qid: query_text_for_cont3(qid) for qid in qids}
        normed: dict[str, str] = {}

        def _norm(iid: str) -> str:
            name = names[iid]
            n = normed.get(name)
            if n is None:
                n = normed[name] = normalize_arabic(name)
            return n

        def starts_with_query(qid: str, iid: str) -> float:
            return 1.0 if _norm(iid).startswith(qtext_stripped[qid]) else 0.0

        qtoks_stripped = {qid: qtext_stripped[qid].split() for qid in qids}

        def match_position(qid: str, iid: str) -> float:
            """1/(1+k) for the query's TOKEN offset k in the title; 0 if absent.

            Strictly generalises the prefix test (which is only the k=0 case) and
            measures better: token position scores 0.683188 full-benchmark against
            the prefix test's 0.680892. Meilisearch, which built the judged pools,
            ranks partly by attribute position, so WHERE the query sits in a title
            is exactly the text dimension that survives containment saturation:
            dozens of titles all contain the query and differ only in placement.
            Token offset beats character offset (0.682680) because it does not
            depend on how long the preceding words happen to be.
            """
            toks = _norm(iid).split()
            q = qtoks_stripped[qid]
            n = len(q)
            if not n:
                return 0.0
            for k in range(len(toks) - n + 1):
                if toks[k:k + n] == q:
                    return 1.0 / (1.0 + k)
            return 0.0

        def first_token_position(qid: str, iid: str) -> float:
            """3/(1+k) for the FIRST query token's offset k in the title; 0 if absent.

            match_position above requires the query tokens to appear CONTIGUOUSLY, so
            every title that holds them scattered - or holds only the leading one -
            scores 0 and falls straight through to raw ascending id. This asks the same
            Meilisearch attribute-position question one level down and re-orders exactly
            that residue.

            It was missed for a long time because the standard screen compared each
            candidate feature's STANDALONE within-cell AUC against the low-id prior,
            where this scores 0.5194 against 0.6230 and is discarded. But the cascade
            sorts by (-feature, +id): the feature PARTITIONS and low-id still orders
            inside each class, so the quantity that matters is the AUC of the COMBINED
            key, which is 0.6368 / 0.6978 against low-id's 0.6230 / 0.6948. Measured the
            way it is actually used, it wins; measured standalone, it looks dead.

            Threshold-free by construction (same 1/(1+k) shape as match_position), so
            there are no bucket edges to overfit. Binary presence alone is worth only
            +0.0006, i.e. the signal really is the POSITION and not the mere match.
            """
            toks = _norm(iid).split()
            q = qtoks_stripped[qid]
            if not q or q[0] not in toks:
                return 0.0
            return 3.0 / (1.0 + toks.index(q[0]))

        def last_token_position(qid: str, iid: str) -> float:
            """Bucketed offset of the LAST query token; a 5th key level below first-token.

            Sub-gate on its own (+0.0012) but part of a near-additive trio: it acts on the
            tie-break key while the idf bonus acts on the blend and --top-eps acts on the
            band geometry, so the three barely overlap and sum to +0.0039.
            """
            toks = _norm(iid).split()
            q = qtoks_stripped[qid]
            if not q or q[-1] not in toks:
                return 0.0
            k = toks.index(q[-1])
            return 3.0 if k == 0 else (2.0 if k <= 2 else 1.0)

        # token IDF over UNIQUE normalized catalog names: catalog-derived, so it is legal
        # under the measurement-only rule on the judged file.
        _tok_df: dict[str, int] = {}
        for _n in {_norm(i) for i in names}:
            for _t in set(_n.split()):
                _tok_df[_t] = _tok_df.get(_t, 0) + 1
        _n_uniq = len({_norm(i) for i in names})
        _best_tok = {qid: (max(qtoks_stripped[qid],
                               key=lambda t: math.log(1.0 + _n_uniq / (_tok_df.get(t, 0) + 0.5)))
                           if qtoks_stripped[qid] else None)
                     for qid in qids}

        def idf_token_position(qid: str, iid: str) -> float:
            """Position of the query's most DISTINCTIVE token, same 3/(1+k) shape."""
            b = _best_tok[qid]
            if b is None:
                return 0.0
            toks = _norm(iid).split()
            return 3.0 / (1.0 + toks.index(b)) if b in toks else 0.0

    def group_key(qid: str):
        if args.tie_key == "cont3posftlt":
            return lambda x: (-cont3(qid, x[1]), -match_position(qid, x[1]),
                              -first_token_position(qid, x[1]),
                              -last_token_position(qid, x[1]), _id_key(x[1]))
        if args.tie_key == "cont3posft":
            return lambda x: (-cont3(qid, x[1]), -match_position(qid, x[1]),
                              -first_token_position(qid, x[1]), _id_key(x[1]))
        if args.tie_key == "cont3pos":
            return lambda x: (-cont3(qid, x[1]), -match_position(qid, x[1]),
                              _id_key(x[1]))
        if args.tie_key == "cont3prefix":
            return lambda x: (-cont3(qid, x[1]), -starts_with_query(qid, x[1]),
                              _id_key(x[1]))
        if args.tie_key == "cont3":
            return lambda x: (-cont3(qid, x[1]), _id_key(x[1]))
        return lambda x: _id_key(x[1])

    def eps_resort(scored: list[tuple[float, str]], eps: float, key) -> list[str]:
        """Chain-group descending scores whose consecutive gap is <= eps; each group
        sorts by `key` (the in-band prior beats sub-eps score noise).

        With --top-eps set, the first --top-k-eps ranks use that band instead. nDCG@10
        weights rank 1 at 1.000 and rank 9 at 0.315, so a re-sort near the head is worth
        roughly three times one near the tail, but eps was uniform across positions.
        """
        out: list[str] = []
        group: list[tuple[float, str]] = []
        for pos, (s, iid) in enumerate(scored):
            e = args.top_eps if (args.top_eps > 0 and pos < args.top_k_eps) else eps
            if group and (group[-1][0] - s) > e:
                group.sort(key=key)
                out.extend(i for _, i in group)
                group = []
            group.append((s, iid))
        group.sort(key=key)
        out.extend(i for _, i in group)
        return out

    # first_token_position only exists under --tie-key cont3posft, and --pos-bonus is
    # legal with plain cont3pos, so bind a zero stand-in rather than risking a NameError
    # in the blend. The conditional never evaluates the real name when ft_bonus is 0.
    ft_score = first_token_position if args.ft_bonus else (lambda qid, iid: 0.0)
    idf_score = idf_token_position if args.idf_bonus else (lambda qid, iid: 0.0)

    def blended(lam: float) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for qid in qids:
            if args.pos_bonus or args.ft_bonus or args.idf_bonus:
                scored = [
                    (lam * ce[(qid, iid)] + (1 - lam) * priors[qid][iid] / rrf_max
                     + args.pos_bonus * match_position(qid, iid)
                     + args.ft_bonus * ft_score(qid, iid)
                     + args.idf_bonus * idf_score(qid, iid), iid)
                    for iid in pools[qid]
                ]
            else:
                scored = [
                    (lam * ce[(qid, iid)] + (1 - lam) * priors[qid][iid] / rrf_max, iid)
                    for iid in pools[qid]
                ]
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            eps = args.eps
            if args.eps_hi > 0 and conf.get(qid, 1.0) < args.conf_t:
                eps = args.eps_hi
            out[qid] = (eps_resort(scored, eps, group_key(qid)) if eps > 0
                        else [iid for _, iid in scored])
        return out

    if args.lam_grid:
        qrels = load_candidates(BENCH / "candidates.jsonl")
        print(f"{'lam':>6s}  {'nDCG@10':>8s}")
        for lam in [float(x) for x in args.lam_grid.split(",")]:
            run = {qid: ids[:10] for qid, ids in blended(lam).items()}
            res = evaluate_run(run, qrels)
            print(f"{lam:6.2f}  {res['nDCG@10']:8.4f}   ({res['queries_evaluated']} queries)")
        return

    run = blended(args.lam)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in run.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{args.name}.csv", {q: ids[: args.top_k] for q, ids in run.items()})
    print(f"wrote runs/{args.name}.jsonl (full pool) and runs/{args.name}.csv (top{args.top_k})")


if __name__ == "__main__":
    main()
