"""Shared utilities: paths, UTF-8 console setup, Arabic normalization, data loaders."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BENCH = DATA / "benchmark" / "data"
BENCH_ROOT = DATA / "benchmark"
TRAIN = DATA / "training"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def setup_utf8():
    """Windows console crashes on Arabic text without this; call at every entrypoint."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


setup_utf8()

# Arabic normalization: diacritics (tashkeel), superscript alef, tatweel.
_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")
# Token = Arabic letter run, or latin/digit run (keeps model codes like "t800" whole).
_TOKEN = re.compile(r"[ء-ي]+|[a-z0-9]+")
_AR_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_AR_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})


def normalize_arabic(text: str) -> str:
    t = text.translate(_AR_DIGIT_MAP)
    t = _AR_DIACRITICS.sub("", t)
    t = re.sub("[أإآٱ]", "ا", t)  # alef variants -> bare alef
    t = t.replace("ى", "ي")  # alef maksura -> yeh
    t = t.replace("ة", "ه")  # teh marbuta -> heh
    t = t.replace("ؤ", "و")  # waw hamza -> waw
    t = t.replace("ئ", "ي")  # yeh hamza -> yeh
    return t.lower()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_arabic(text))


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """qrels.tsv rows: query_id <tab> iter <tab> item_id <tab> relevance."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("query_id\t"):
                continue
            qid, _, item_id, rel = line.split("\t")
            qrels.setdefault(str(qid), {})[str(item_id)] = int(rel)
    return qrels


def load_run_csv(path: Path) -> dict[str, list[str]]:
    """Submission-format CSV: header query_id,product_id; ids space-separated, best first.

    Duplicates within a row are dropped after first occurrence, list truncated to 10.
    """
    import csv

    run: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["query_id"]).strip()
            ids = str(row["product_id"]).split()
            seen: set[str] = set()
            deduped = []
            for pid in ids:
                if pid not in seen:
                    seen.add(pid)
                    deduped.append(pid)
            run[qid] = deduped[:10]
    return run


def write_run_csv(path: Path, run: dict[str, list[str]]):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "product_id"])
        for qid, ids in run.items():
            writer.writerow([qid, " ".join(ids)])
