"""Memory-resident harness mirroring the live v6warm transform byte-for-byte.

Mirrors src/grid_search_v6warm.py logic (which reproduced the shipped submission):
pools from bm25+dense top-100, weighted RRF prior (2:1, k=10), T=2.3 delogit
rescale of the v6warm CE cache, blend lam*CE + (1-lam)*prior/rrf_max, adaptive
eps near-tie chain re-sort keyed by (cont3 desc, id asc) with Set-D filler
stripping, low-id ties.

Extensions for the probe batch (all default OFF -> reproduces live):
  - key="cont3idf": IDF-weighted containment in-band key
  - znorm=True: per-query z-normalized CE in the blend term
  - lam_fn: callable qid -> lam (routing); lam used otherwise
  - id_bonus: score -= id_bonus * (numeric_id / MAX_ID)
  - extra_votes: {qid: {iid: rrf_votes}} added to the prior (third-voter injection)
  - extra_weight: weight of the extra voter (adds to rrf_max normalization)

Reproduction anchors: dev_unseen2000 = 0.6641, full val = 0.6639 at live params
(lam .45, eps .020, eps_hi .080, conf_t .10, key cont3, T 2.3).
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\LOQ\presto_v2\src")

import pandas as pd

from common import BENCH, DATA, ROOT, load_jsonl, normalize_arabic
from eval_ndcg import evaluate_run, load_candidates, ndcg_at_k
from stacker_features import char_ngrams, containment

DIALECT_FILLERS = {normalize_arabic(w) for w in (
    "للدار", "عطيني", "هاتلي", "ندور", "محتاج",
    "نبي", "ابي", "اريد", "بدي", "عايز", "ابغي")}

MAX_ID = 107041.0


def _id_key(iid: str):
    try:
        return int(iid)
    except ValueError:
        return iid


def rescale(raw: float, T: float) -> float:
    s = min(max(raw, 1e-12), 1.0 - 1e-12)
    logit = math.log(s / (1.0 - s))
    return 1.0 / (1.0 + math.exp(-logit / T))


_QUERIES = None
_NAMES = None
_QRELS = None


def shared_data():
    global _QUERIES, _NAMES, _QRELS
    if _QUERIES is None:
        _QUERIES = {str(q["query_id"]): q["query"] for q in load_jsonl(BENCH / "queries.jsonl")}
        _NAMES = {str(r["item_id"]): r["name"] for r in load_jsonl(BENCH / "corpus.jsonl")}
        _QRELS = load_candidates(BENCH / "candidates.jsonl")
    return _QUERIES, _NAMES, _QRELS


def stream_topk(path: Path, k: int, qid_set: set[str] | None = None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # cheap qid prefilter before full json parse
            if qid_set is not None:
                head = line[:60]
                qpos = head.find('"query_id"')
                if qpos >= 0:
                    colon = head.find(":", qpos)
                    qid_raw = head[colon + 1 : head.find(",", colon)].strip().strip('"')
                    if qid_raw not in qid_set:
                        continue
            obj = json.loads(line)
            qid = str(obj["query_id"])
            if qid_set is not None and qid not in qid_set:
                continue
            out[qid] = [str(e["item_id"]) for e in obj["retrieved"][:k]]
    return out


class Harness:
    def __init__(self, qids: list[str], T: float = 2.3, depth: int = 100,
                 cache_stem: str = "rerank_qwen3-reranker-arabic-ecom-rerank_v6warm_len160_bfloat16",
                 weights: list[float] | None = None, rrf_k: int = 10):
        self.qids = list(qids)
        qid_set = set(self.qids)
        self.queries, self.names, self.qrels = shared_data()

        self.run_bm25 = stream_topk(ROOT / "runs" / "bm25_all.jsonl", depth, qid_set)
        self.run_dense = stream_topk(ROOT / "runs" / "dense_lora_v1_all.jsonl", depth, qid_set)

        self.rrf_k = rrf_k
        self.weights = list(weights) if weights else [2.0, 1.0]
        self.pools: dict[str, list[str]] = {}
        self.priors: dict[str, dict[str, float]] = {}
        for qid in self.qids:
            seen: dict[str, float] = {}
            pool: list[str] = []
            for run, w in zip([self.run_bm25, self.run_dense], self.weights):
                for rank, iid in enumerate(run.get(qid, [])[:depth], start=1):
                    if iid not in seen:
                        seen[iid] = 0.0
                        pool.append(iid)
                    seen[iid] += w / (self.rrf_k + rank)
            self.pools[qid] = pool
            self.priors[qid] = seen

        cdf = pd.read_parquet(DATA / "cache" / f"{cache_stem}.parquet")
        cdf = cdf[cdf["qid"].isin(qid_set)]
        raw = {(q, i): s for q, i, s in zip(cdf["qid"], cdf["iid"], cdf["score"])}
        del cdf
        self.ce = {}
        self.missing_ce = 0
        for qid in self.qids:
            for iid in self.pools[qid]:
                r = raw.get((qid, iid))
                if r is None:
                    self.missing_ce += 1
                    r = 0.0
                self.ce[(qid, iid)] = rescale(r, T)

        self.conf = {}
        for qid in self.qids:
            ces = sorted((self.ce[(qid, iid)] for iid in self.pools[qid]), reverse=True)
            self.conf[qid] = ces[0] - ces[4] if len(ces) >= 5 else 1.0

        # per-query z-norm stats over the pool
        self.ce_mu = {}
        self.ce_sd = {}
        for qid in self.qids:
            vals = [self.ce[(qid, iid)] for iid in self.pools[qid]]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
            self.ce_mu[qid], self.ce_sd[qid] = mu, max(sd, 1e-9)

        # cont3 machinery (Set-D stripped query grams), plus IDF over unique names
        self.qgrams = {}
        for qid in self.qids:
            qn = normalize_arabic(self.queries[qid])
            kept = [t for t in qn.split() if t not in DIALECT_FILLERS]
            self.qgrams[qid] = char_ngrams(" ".join(kept) if kept else qn, 3)
        self._ngrams: dict[str, frozenset] = {}

        self._df: dict[str, int] | None = None
        self._n_unique_names: int | None = None

    def name_grams(self, iid: str) -> frozenset:
        name = self.names[iid]
        g = self._ngrams.get(name)
        if g is None:
            g = char_ngrams(normalize_arabic(name), 3)
            self._ngrams[name] = g
        return g

    def _build_idf(self):
        df: dict[str, int] = {}
        uniq = {normalize_arabic(n) for n in self.names.values()}
        for nn in uniq:
            for g in char_ngrams(nn, 3):
                df[g] = df.get(g, 0) + 1
        self._df = df
        self._n_unique_names = len(uniq)

    def idf(self, gram: str) -> float:
        if self._df is None:
            self._build_idf()
        d = self._df.get(gram, 0)
        return math.log(1.0 + self._n_unique_names / (d + 0.5))

    def cont3(self, qid: str, iid: str) -> float:
        return containment(self.qgrams[qid], self.name_grams(iid))

    def cont3idf(self, qid: str, iid: str) -> float:
        qg = self.qgrams[qid]
        if not qg:
            return 0.0
        ng = self.name_grams(iid)
        num = sum(self.idf(g) for g in qg & ng)
        den = sum(self.idf(g) for g in qg)
        return num / den if den > 0 else 0.0

    @staticmethod
    def eps_resort(scored, eps, key):
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

    def norm_name(self, iid: str) -> str:
        nm = getattr(self, "_normname", None)
        if nm is None:
            nm = self._normname = {}
        v = nm.get(iid)
        if v is None:
            v = nm[iid] = normalize_arabic(self.names[iid])
        return v

    def build_run(self, lam=0.45, eps=0.020, eps_hi=0.080, conf_t=0.10, key="cont3",
                  znorm=False, lam_fn=None, id_bonus=0.0, bonus_fn=None,
                  extra_votes=None, extra_weight=0.0,
                  dup_cap=0, dup_cap_wide_only=False,
                  pin_top=0, pin_wide_only=False, qids=None, topn=10) -> dict[str, list[str]]:
        qids = qids or self.qids
        rrf_max = (sum(self.weights) + extra_weight) / (self.rrf_k + 1)

        if key == "cont3":
            keyf = self.cont3
        elif key == "cont3idf":
            keyf = self.cont3idf
        elif callable(key):
            keyf = key
        else:
            raise ValueError(key)

        run = {}
        for qid in qids:
            l = lam_fn(qid) if lam_fn is not None else lam
            prior = self.priors[qid]
            ev = extra_votes.get(qid, {}) if extra_votes else {}
            scored = []
            for iid in self.pools[qid]:
                c = self.ce[(qid, iid)]
                if znorm:
                    c = (c - self.ce_mu[qid]) / self.ce_sd[qid]
                p = (prior[iid] + ev.get(iid, 0.0)) / rrf_max
                s = l * c + (1 - l) * p
                if id_bonus:
                    digits = "".join(ch for ch in iid if ch.isdigit())
                    if digits:
                        s -= id_bonus * (int(digits) / MAX_ID)
                if bonus_fn is not None:
                    s += bonus_fn(qid, iid)
                scored.append((s, iid))
            scored.sort(key=lambda x: (-x[0], _id_key(x[1])))
            wide = self.conf[qid] < conf_t
            cur_eps = eps_hi if wide else eps
            gk = lambda x, q=qid: (-keyf(q, x[1]), _id_key(x[1]))

            n_pin = pin_top if (pin_top and (wide or not pin_wide_only)) else 0
            if n_pin:
                prefix = [iid for _, iid in scored[:n_pin]]
                rest = self.eps_resort(scored[n_pin:], cur_eps, gk)
                order = prefix + rest
            else:
                order = self.eps_resort(scored, cur_eps, gk)

            cap = dup_cap if (dup_cap and (wide or not dup_cap_wide_only)) else 0
            if cap:
                counts: dict[str, int] = {}
                kept, overflow = [], []
                for iid in order:
                    nm = self.norm_name(iid)
                    if counts.get(nm, 0) >= cap:
                        overflow.append(iid)
                    else:
                        counts[nm] = counts.get(nm, 0) + 1
                        kept.append(iid)
                order = kept + overflow

            run[qid] = order[:topn]
        return run

    def score(self, run: dict[str, list[str]]) -> float:
        return evaluate_run(run, self.qrels)["nDCG@10"]

    def evaluate(self, **cfg) -> float:
        return self.score(self.build_run(**cfg))

    def per_query_ndcg(self, run: dict[str, list[str]]) -> dict[str, float]:
        return {qid: ndcg_at_k(run[qid], self.qrels[qid], 10) for qid in run if qid in self.qrels}
