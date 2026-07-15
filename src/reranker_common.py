"""Shared pieces for the Qwen3 seq-cls cross-encoder reranker (phase 2 step A).

The checkpoint is tomaarsen/Qwen3-Reranker-0.6B-seq-cls: the original causal-LM
reranker converted so score = sigmoid(single logit), drop-in for the
sentence-transformers CrossEncoder. Inputs must be formatted with the official
chat template below; queries carry a custom instruction (Qwen reports +1-5% from
task-specific instructions, ours mirrors the retriever's stored query prompt).

Measured on real data: the template alone is 75 tokens, full pairs p50=96,
p99=118, max~130, hence the 160 default max length everywhere.
"""

from common import MODELS

BASE_CE = MODELS / "qwen3-reranker-0.6b-seq-cls"

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given an Arabic e-commerce search query, retrieve the product that best matches it"


def format_query(query: str) -> str:
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n"


def format_document(doc: str) -> str:
    return f"<Document>: {doc}{SUFFIX}"
