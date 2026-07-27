"""PLAY B probe: does a zero-shot LISTWISE re-ranker beat our pairwise ordering?

Our cross-encoder scores each (query, document) pair in ISOLATION, so it can never
say "this candidate is better than that one" - it assigns independent scores that we
sort. The dominant remaining error (loss autopsy + the v8graded result) is
same-brand / size-variant confusion, which is inherently COMPARATIVE. A listwise
re-ranker sees the top-K candidates TOGETHER and orders them jointly, attacking
exactly that failure mode.

Judge: `models/qwen3-reranker-4b-seq-cls` reloaded as Qwen3ForCausalLM. The config
sets tie_word_embeddings, so lm_head ties back to embed_tokens and we recover the
original Qwen3-Reranker-4B language-model head with no download (see
src/smoke_4b_causal.py, which also established that RAW completion works and the
chat template wastes tokens on an English <think> monologue).

Protocol notes:
  - Baseline is the LIVE shipped stack (LB 0.66861): v7synth CE at T=2.6, RRF k=60
    at 3:1, lam 0.60, eps 0.010/0.040, top1-top3 confidence margin at conf_t 0.30,
    cont3 in-band key with Set-D filler stripping.
  - The listwise model re-orders the live top-K; the top-10 of that ordering is
    scored. K > 10 lets it PROMOTE items our stack left below the cut, so this
    measures more than a permutation of the existing head.
  - Queries come from a FRESH split drawn from the untouched remainder (the six
    named dev splits and val are all excluded), per protocol rule 4.
  - Reported three ways: applied to all queries, applied only to wide-band
    (low-confidence) queries, and applied only to narrow-band queries. The
    wide-band figure is the one that matters, because that is how it would ship.
"""

import argparse
import json
import random
import re
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

رتب المنتجات من الأكثر صلة إلى الأقل صلة بالاستعلام.
أجب فقط بالترتيب بهذا الشكل: [رقم] > [رقم] > [رقم]

الترتيب:"""


def mint_split(n: int, seed: int = 47) -> list[str]:
    """n queries sampled from the untouched remainder (excl. six dev splits + val)."""
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


def parse_order(text: str, k: int) -> list[int] | None:
    """'2 > 4 > 3 > 1' -> [1,3,2,0] (0-based). None if nothing parseable."""
    head = text.strip().split("\n")[0]
    nums = [int(x) for x in re.findall(r"\d+", head)]
    seen, order = set(), []
    for n in nums:
        if 1 <= n <= k and n not in seen:
            seen.add(n)
            order.append(n - 1)
    return order or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=20, help="candidates shown to the judge")
    p.add_argument("--n", type=int, default=200, help="probe queries")
    p.add_argument("--limit", type=int, default=0, help="0 = all (dry-run smaller)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new", type=int, default=110)
    p.add_argument("--model", type=Path, default=MODELS / "qwen3-reranker-4b-seq-cls")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "probe_listwise.json")
    args = p.parse_args()

    qids = mint_split(args.n)
    if args.limit:
        qids = qids[: args.limit]
    print(f"probe on {len(qids)} queries, K={args.k}, judge={args.model.name}\n")

    # ---- baseline: the live shipped stack ----
    h = Harness(qids, T=2.6, cache_stem=V7, weights=[3.0, 1.0], rrf_k=60)
    for qid in h.qids:  # live margin is top1-top3, not the harness default top1-top5
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

    # ---- listwise judge ----
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(str(args.model))
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), quantization_config=qc, dtype=torch.bfloat16)
    model.eval()

    prompts = []
    for qid in qids:
        cands = cand[qid][: args.k]
        lines = "\n".join(f"[{i}] {h.names[iid]}" for i, iid in enumerate(cands, 1))
        prompts.append(PROMPT.format(query=h.queries[qid], candidates=lines))

    gens: list[str] = []
    t0 = time.time()
    for s in range(0, len(prompts), args.batch):
        chunk = prompts[s : s + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        gens.extend(tok.batch_decode(new, skip_special_tokens=True))
        done = min(s + args.batch, len(prompts))
        el = time.time() - t0
        print(f"  {done}/{len(prompts)}  {el / 60:.1f} min elapsed, "
              f"eta {el / done * (len(prompts) - done) / 60:.1f} min", flush=True)

    # ---- apply the ordering ----
    ls: dict[str, list[str]] = {}
    n_fail = n_partial = n_top1_changed = 0
    records = []
    for qid, text in zip(qids, gens):
        cands = cand[qid][: args.k]
        order = parse_order(text, len(cands))
        if order is None:
            n_fail += 1
            ls[qid] = base[qid]
            new_ids = base[qid]
        else:
            if len(order) < len(cands):
                n_partial += 1
            rest = [i for i in range(len(cands)) if i not in set(order)]
            new_ids = [cands[i] for i in order + rest]
            ls[qid] = new_ids[:10]
        if new_ids[:1] != base[qid][:1]:
            n_top1_changed += 1
        records.append({
            "qid": qid, "query": h.queries[qid], "wide": qid in wide,
            "gen": text.strip()[:200], "parsed": order,
            "base_ndcg": ndcg_at_k(base[qid], h.qrels[qid], 10),
            "ls_ndcg": ndcg_at_k(ls[qid], h.qrels[qid], 10),
        })

    print(f"\nparse: {n_fail} failed, {n_partial} partial, "
          f"top-1 changed on {n_top1_changed}/{len(qids)}")

    # ---- verdict ----
    wide_only = {q: (ls[q] if q in wide else base[q]) for q in qids}
    narrow_only = {q: (base[q] if q in wide else ls[q]) for q in qids}
    narrow = [q for q in qids if q not in wide]

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

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nper-query records -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
