"""Kaggle kernel: constrained-decoding LISTWISE judge over the low-confidence band.

Runs on 2x T4. Reads the exported band pools, asks a generative judge to re-order the
top-K candidates of each query, and writes back ONLY permutations. No labels are
uploaded and the kernel cannot evaluate itself; scoring happens locally.

Constrained decoding: candidates are labelled with single-token letters A..T and the
ordering is built by greedy selection with the logits MASKED to letters not yet
chosen, so every output is a valid permutation by construction. This matters because
a free-generation version of this probe measured a large FALSE negative locally: the
model failed at output format (runaway counting, identity copy-out, list restatement)
rather than at ranking, and constrained decoding flipped the sign.

Two judges on identical queries, because the open question is whether world knowledge
helps at all on this task: a 4B instruct model and a 14B. Every zero-shot model tried
so far has been domain-weak, so the scaling comparison is the actual experiment.

T4 note: fp16 only, no bf16. That is acceptable here because the output is a
permutation, not a score fed into a calibrated blend, so there is no dtype confound
of the kind that rules out T4 training.
"""

import json
import os
import random
import string
import subprocess
import sys
import time

import pandas as pd
import torch

INPUT_GLOB = "listwise_probe_input.parquet"
OUT = "/kaggle/working/listwise_orders.parquet"
SELECT = 10           # only the top-10 matter for nDCG@10

# Presentation order. "live" shows candidates in our own ranking, which converts the
# judge's position bias into a safety prior. "shuffle" is the CONTROL: a judge with
# real order-independent signal should still surface the good items, while one that
# merely echoes the order it was shown collapses. That distinction decides whether a
# positive live-order result is genuine judgement or an artifact, and a pilot on our
# own reranker checkpoint failed it badly (re-found the live top-1 on 9% of queries
# against a 5% chance rate), so it is not a hypothetical concern.
PRESENTATION = os.environ.get("PRESENTATION", "live")
MODELS = [
    ("qwen3-4b-instruct", "Qwen/Qwen3-4B-Instruct-2507", True),
    ("qwen3-14b", "Qwen/Qwen3-14B", True),
]

PROMPT = """أنت خبير في البحث عن المنتجات في متجر إلكتروني.

استعلام البحث: "{query}"

المنتجات المرشحة:
{candidates}

رتب المنتجات من الأكثر صلة إلى الأقل صلة بالاستعلام، الأكثر صلة أولا.

الترتيب:"""


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)


def find_input():
    """Locate the parquet anywhere under /kaggle/input.

    Hardcoding /kaggle/input/<slug>/<file> killed the first run: the dataset was
    still processing when the kernel started, so the path did not exist yet. Search
    instead, and print the tree when it fails so the next diagnosis is immediate.
    """
    import glob
    hits = glob.glob(f"/kaggle/input/**/{INPUT_GLOB}", recursive=True)
    if hits:
        print(f"input: {hits[0]}", flush=True)
        return hits[0]
    print("!! input not found. /kaggle/input contains:", flush=True)
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print(f"   {os.path.join(root, f)}", flush=True)
    raise SystemExit("input parquet not attached; is the dataset still processing?")


def load_input():
    df = pd.read_parquet(find_input())
    groups = []
    for qid, g in df.groupby("qid", sort=False):
        g = g.sort_values("rank")
        groups.append({
            "qid": str(qid),
            "query": g["query"].iloc[0],
            "items": [str(x) for x in g["item_id"].tolist()],
            "names": [str(x) for x in g["name"].tolist()],
        })
    print(f"loaded {len(groups)} queries, K={len(groups[0]['items'])}", flush=True)
    return groups


def run_model(tag, repo, use_chat, groups):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    print(f"\n=== {tag} ({repo}) ===", flush=True)
    t0 = time.time()
    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # T4: no bf16
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=qc, dtype=torch.float16, device_map="auto")
    model.eval()
    dev = next(model.parameters()).device
    print(f"loaded in {time.time() - t0:.0f}s on {dev}", flush=True)

    k = len(groups[0]["items"])
    letters = string.ascii_uppercase[:k]
    letter_ids = []
    for ch in letters:
        ids = tok(f" {ch}", add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise SystemExit(f"' {ch}' not single-token for {repo}: {ids}")
        letter_ids.append(ids[0])
    sep_ids = tok(" >", add_special_tokens=False)["input_ids"]
    letter_t = torch.tensor(letter_ids, device=dev)

    rows = []
    t0 = time.time()
    for n, g in enumerate(groups, 1):
        items, names = g["items"], g["names"]
        if PRESENTATION == "shuffle":
            # deterministic per query so the control is reproducible
            perm = list(range(len(items)))
            random.Random(1000 + int(g["qid"])).shuffle(perm)
            items = [items[i] for i in perm]
            names = [names[i] for i in perm]
        lines = "\n".join(f"[{letters[i]}] {nm}" for i, nm in enumerate(names))
        prompt = PROMPT.format(query=g["query"], candidates=lines)
        if use_chat:
            try:  # Qwen3 hybrids default to thinking; force it off
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(dev)

        with torch.no_grad():
            out = model(**enc, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
            chosen, confs = [], []
            used = torch.zeros(k, dtype=torch.bool, device=dev)
            for step in range(min(SELECT, k)):
                cl = logits[letter_t].float().masked_fill(used, float("-inf"))
                probs = cl.softmax(-1)
                pick = int(cl.argmax())
                chosen.append(pick)
                confs.append(float(probs[pick]))
                used[pick] = True
                if step == min(SELECT, k) - 1:
                    break
                feed = torch.tensor([[letter_ids[pick]] + sep_ids], device=dev)
                out = model(input_ids=feed, past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1]

        rest = [i for i in range(k) if i not in set(chosen)]
        order = chosen + rest
        rows.append({
            "model": tag, "qid": g["qid"], "presentation": PRESENTATION,
            "order": json.dumps(order),
            # item ids, so the local evaluator never depends on presentation order
            "items": json.dumps([items[i] for i in order]),
            "shown": json.dumps(items),
            "conf": json.dumps([round(c, 4) for c in confs]),
        })
        if n % 50 == 0 or n == len(groups):
            el = time.time() - t0
            print(f"  {n}/{len(groups)}  {el / 60:.1f} min, "
                  f"eta {el / n * (len(groups) - n) / 60:.1f} min", flush=True)

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    pip("-U", "bitsandbytes", "accelerate")
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
          f"gpus {torch.cuda.device_count()}", flush=True)
    groups = load_input()

    all_rows = []
    for tag, repo, use_chat in MODELS:
        try:
            all_rows.extend(run_model(tag, repo, use_chat, groups))
            pd.DataFrame(all_rows).to_parquet(OUT, index=False)  # checkpoint per model
            print(f"checkpointed {len(all_rows)} rows -> {OUT}", flush=True)
        except Exception as e:
            import traceback
            print(f"!! {tag} FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if all_rows:
        pd.DataFrame(all_rows).to_parquet(OUT, index=False)
        print(f"\nWROTE {OUT}: {len(all_rows)} rows", flush=True)
    else:
        raise SystemExit("no model produced output")


if __name__ == "__main__":
    main()
