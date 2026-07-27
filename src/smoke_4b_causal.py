"""Can the banked seq-cls 4B be reloaded as a GENERATIVE model? (gate for PLAY B)

`models/qwen3-reranker-4b-seq-cls` is a Qwen3ForSequenceClassification conversion of
Qwen3-Reranker-4B: the index holds `score.weight` and NO `lm_head.weight`. But the
config sets `tie_word_embeddings: true`, and the original reranker was tied too, so
loading the same directory as Qwen3ForCausalLM ties lm_head back to embed_tokens and
recovers the ORIGINAL language-model head exactly. No download, no extra disk.

This script proves (or disproves) three things before any probe is built:
  1. the checkpoint loads as a causal LM in 4-bit within budget;
  2. lm_head is genuinely tied to embed_tokens (not randomly initialised);
  3. it can still follow an instruction and emit a listwise ordering in our format.

Point 3 is the real risk: this checkpoint was fine-tuned to emit one of two tokens
(yes/no), so its general instruction-following may have been trained away.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import MODELS, setup_utf8

setup_utf8()

MODEL = MODELS / "qwen3-reranker-4b-seq-cls"


def vram():
    free, tot = torch.cuda.mem_get_info()
    return (tot - free) / 1e9, tot / 1e9


PROMPT = """أنت خبير في البحث عن المنتجات في متجر إلكتروني.

استعلام البحث: "حليب اطفال نيدو"

المنتجات المرشحة:
[1] شامبو للاطفال جونسون - 200 مل
[2] حليب نيدو للاطفال مرحلة 2 - 400 غ
[3] حليب المراعي طازج - 1 لتر
[4] حليب نيدو بودرة - 1800 غ

رتب المنتجات من الأكثر صلة إلى الأقل صلة بالاستعلام.
أجب فقط بالترتيب بهذا الشكل: [رقم] > [رقم] > [رقم] > [رقم]

الترتيب:"""


def main():
    print(f"model: {MODEL.name}\n")
    t0 = time.time()

    qc = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), quantization_config=qc, dtype=torch.bfloat16)
    model.eval()
    used, tot = vram()
    print(f"1. LOADED as causal LM in 4-bit: {used:.2f}/{tot:.2f} GB  [{time.time() - t0:.0f}s]")

    # 2. is lm_head actually tied to embed_tokens?
    lm_w = model.get_output_embeddings().weight
    emb_w = model.get_input_embeddings().weight
    tied = lm_w.data_ptr() == emb_w.data_ptr()
    print(f"2. lm_head tied to embed_tokens: {tied}  "
          f"(lm_head {tuple(lm_w.shape)}, embed {tuple(emb_w.shape)})")
    if not tied:
        print("   !! NOT tied - the head may be randomly initialised; probe is invalid")

    # sanity: a tied, real LM head should give a confident sensible next token
    ids = tok("عاصمة فرنسا هي", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    top = torch.topk(logits.float().softmax(-1), 5)
    print("   next-token sanity for 'عاصمة فرنسا هي':")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"     {p:.3f}  {tok.decode([i])!r}")

    # 3. can it follow a listwise instruction?
    print("\n3. LISTWISE INSTRUCTION TEST")
    for use_chat in (True, False):
        if use_chat and tok.chat_template is None:
            print("   (no chat template on this tokenizer)")
            continue
        if use_chat:
            text = tok.apply_chat_template(
                [{"role": "user", "content": PROMPT}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = PROMPT
        enc = tok(text, return_tensors="pt").to("cuda")
        t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        dt = time.time() - t
        n_new = out.shape[1] - enc["input_ids"].shape[1]
        print(f"   --- chat_template={use_chat} ({n_new} tok in {dt:.1f}s = "
              f"{n_new / dt:.1f} tok/s) ---")
        print(f"   {gen!r}")

    print("\n   expected if healthy: an ordering putting [2] and [4] (نيدو) first")


if __name__ == "__main__":
    main()
