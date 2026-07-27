"""PLAY B probe, mark 2: listwise re-ranking under CONSTRAINED decoding.

Mark 1 (`src/probe_listwise.py`, free generation) measured -0.029, but inspection of
the generations showed the number was dominated by FORMAT failure, not ranking
judgement. Four distinct pathologies on 16 queries:
  - runaway counting ("14, 16, 17, ... 42") on a query whose list stopped at 20;
  - identity copy-out ("1,2,3,...,20"), i.e. the input order handed back, a no-op;
  - restating the list ("1-10" then a numbered re-print), which parsed to a 2-item
    order and destroyed a 0.847 query;
  - out-of-range indices, then English self-correction ("Wait, but the original
    list only goes up to 20").
That is what a checkpoint fine-tuned to emit one of two tokens (yes/no) does when
asked for free-form structured output. It says nothing about whether SEEING THE
CANDIDATES TOGETHER helps.

This version removes the failure mode entirely. Candidates are labelled with
single-token letters and the ordering is built by greedy CONSTRAINED selection: at
each step the logits are masked to the letters not yet chosen, so every output is a
valid permutation by construction. The model can no longer fail at formatting - it
can only make choices, which is exactly the capability under test.

Only the top-10 are selected (nDCG@10 needs no more); the remainder keep their live
order. Cost is one prompt forward plus ten single-token forwards per query.
"""

import argparse
import json
import random
import string
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import BENCH, MODELS, ROOT, load_jsonl, load_qid_file, setup_utf8
from eval_ndcg import ndcg_at_k
from probe_harness import Harness

setup_utf8()

V7 = "rerank_qwen3-reranker-arabic-ecom-rerank_v7synth_len160_bfloat16"
LIVE = dict(lam=0.60, eps=0.010, eps_hi=0.040, conf_t=0.30, key="cont3")
SPLIT = ROOT / "splits" / "dev_listwise200.txt"

PROMPT = """أنت خبير في البحث عن المنتجات في متجر إلكتروني.

استعلام البحث: "{query}"

المنتجات المرشحة:
{candidates}

رتب المنتجات من الأكثر صلة إلى الأقل صلة بالاستعلام، الأكثر صلة أولا.

الترتيب:"""


def mint_split(n: int, seed: int = 47) -> list[str]:
    if SPLIT.exists():
        return sorted(load_qid_file(SPLIT), key=int)
    all_qids = {str(q["query_id"]) for q in load_jsonl(BENCH / "queries.jsonl")}
    used = load_qid_file(ROOT / "splits" / "val_qids.txt")
    for f in ("dev_rand2000.txt", "dev_unseen2000.txt", "dev_probe2000.txt",
              "dev_recal2000.txt", "dev_ens2000.txt", "dev_final2000.txt"):
        used |= load_qid_file(ROOT / "splits" / f)
    remainder = sorted(all_qids - used, key=int)
    picked = sorted(random.Random(seed).sample(remainder, n), key=int)
    SPLIT.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"minted {SPLIT.name}: {n} of {len(remainder)} untouched queries (seed {seed})")
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20, help="candidates shown to the judge")
    p.add_argument("--select", type=int, default=10, help="positions to fill")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", type=Path, default=MODELS / "qwen3-reranker-4b-seq-cls")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "probe_listwise_constrained.json")
    args = p.parse_args()

    qids = mint_split(args.n)
    if args.limit:
        qids = qids[: args.limit]
    print(f"probe on {len(qids)} queries, K={args.k}, select={args.select}\n")

    # ---- baseline: the live shipped stack (LB 0.66861) ----
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:  # live margin is top1-top3
        v = sorted((h.ce[(qid, i)] for i in h.pools[qid]), reverse=True)
        h.conf[qid] = v[0] - v[2] if len(v) >= 3 else 1.0
    cand = h.build_run(topn=args.k, **LIVE)
    base = {q: ids[:10] for q, ids in cand.items()}
    wide = {q for q in qids if h.conf[q] < LIVE["conf_t"]}
    print(f"wide band (conf < {LIVE['conf_t']}): {len(wide)}/{len(qids)} queries")

    def mean_ndcg(run, qs):
        v = [ndcg_at_k(run[q], h.qrels[q], 10) for q in qs if q in run and q in h.qrels]
        return sum(v) / len(v) if v else 0.0

    print(f"baseline nDCG@10 (live stack): {mean_ndcg(base, qids):.6f}\n")

    # ---- judge ----
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), quantization_config=qc, dtype=torch.bfloat16)
    model.eval()

    letters = string.ascii_uppercase[: args.k]
    letter_ids = []
    for ch in letters:
        ids = tok(f" {ch}", add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"' {ch}' is not a single token: {ids}"
        letter_ids.append(ids[0])
    sep_ids = tok(" >", add_special_tokens=False)["input_ids"]
    print(f"label tokens: {len(letter_ids)} single-token letters, sep={sep_ids}\n")
    letter_t = torch.tensor(letter_ids, device="cuda")

    ls: dict[str, list[str]] = {}
    records = []
    n_top1_changed = 0
    t0 = time.time()
    for n, qid in enumerate(qids, 1):
        cands = cand[qid][: args.k]
        k = len(cands)
        lines = "\n".join(f"[{letters[i]}] {h.names[iid]}" for i, iid in enumerate(cands))
        prompt = PROMPT.format(query=h.queries[qid], candidates=lines)
        enc = tok(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            out = model(**enc, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]

            chosen: list[int] = []
            confs: list[float] = []
            used = torch.zeros(k, dtype=torch.bool, device="cuda")
            n_sel = min(args.select, k)
            for step in range(n_sel):
                cl = logits[letter_t[:k]].float()
                cl = cl.masked_fill(used, float("-inf"))
                probs = cl.softmax(-1)
                pick = int(cl.argmax())
                chosen.append(pick)
                confs.append(float(probs[pick]))
                used[pick] = True
                if step == n_sel - 1:
                    break
                feed = torch.tensor([[letter_ids[pick]] + sep_ids], device="cuda")
                out = model(input_ids=feed, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]

        rest = [i for i in range(k) if i not in set(chosen)]
        new_ids = [cands[i] for i in chosen + rest]
        ls[qid] = new_ids[:10]
        if new_ids[:1] != base[qid][:1]:
            n_top1_changed += 1
        records.append({
            "qid": qid, "query": h.queries[qid], "wide": qid in wide,
            "chosen": chosen, "conf": [round(c, 3) for c in confs],
            "base_names": [h.names[i] for i in base[qid][:3]],
            "ls_names": [h.names[i] for i in ls[qid][:3]],
            "base_ndcg": ndcg_at_k(base[qid], h.qrels[qid], 10),
            "ls_ndcg": ndcg_at_k(ls[qid], h.qrels[qid], 10),
        })
        if n % 20 == 0 or n == len(qids):
            el = time.time() - t0
            print(f"  {n}/{len(qids)}  {el / 60:.1f} min, "
                  f"eta {el / n * (len(qids) - n) / 60:.1f} min", flush=True)

    print(f"\ntop-1 changed on {n_top1_changed}/{len(qids)} queries")

    wide_only = {q: (ls[q] if q in wide else base[q]) for q in qids}
    narrow = [q for q in qids if q not in wide]
    narrow_only = {q: (base[q] if q in wide else ls[q]) for q in qids}

    print("\n" + "=" * 62)
    print(f"{'slice':<34s}{'baseline':>10s}{'listwise':>10s}{'delta':>9s}")
    print("=" * 62)
    for label, qs, run in (
        ("ALL queries", qids, ls),
        (f"wide band only ({len(wide)} q)", qids, wide_only),
        (f"narrow band only ({len(narrow)} q)", qids, narrow_only),
        ("-- within wide band", sorted(wide), ls),
        ("-- within narrow band", narrow, ls),
    ):
        b, c = mean_ndcg(base, qs), mean_ndcg(run, qs)
        print(f"{label:<34s}{b:>10.6f}{c:>10.6f}{c - b:>+9.6f}")
    print("=" * 62)

    win = sum(1 for r in records if r["ls_ndcg"] > r["base_ndcg"] + 1e-9)
    loss = sum(1 for r in records if r["ls_ndcg"] < r["base_ndcg"] - 1e-9)
    print(f"per-query: {win} better, {loss} worse, {len(records) - win - loss} unchanged")

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nper-query records -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
