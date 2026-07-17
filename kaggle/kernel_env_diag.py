"""Kaggle environment diagnostic: internet, GPUs, transformers/qwen3 support."""

import glob
import socket

print("== env diag ==", flush=True)
for host in ("huggingface.co", "pypi.org"):
    try:
        socket.getaddrinfo(host, 443)
        print(f"dns {host}: OK", flush=True)
    except Exception as e:
        print(f"dns {host}: FAIL ({e})", flush=True)

try:
    import torch

    print("torch", torch.__version__, "| cuda devices:", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        print("  gpu", i, torch.cuda.get_device_name(i), flush=True)
except Exception as e:
    print("torch err:", e, flush=True)

try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    print("transformers", transformers.__version__,
          "| qwen3 registered:", "qwen3" in CONFIG_MAPPING_NAMES, flush=True)
except Exception as e:
    print("transformers err:", e, flush=True)

try:
    import sentence_transformers

    print("sentence_transformers", sentence_transformers.__version__, flush=True)
except Exception as e:
    print("sentence_transformers err:", e, flush=True)

print("inputs:", glob.glob("/kaggle/input/*"), flush=True)
print("DIAG DONE", flush=True)
