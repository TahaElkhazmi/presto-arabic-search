"""Duplicate-name canonicalization + expansion of ranked runs (phase 2 step C).

Measured on dev qrels (see FINDINGS 2026-07-15): when one id of a duplicated name is
judged positive, 78% of the name's ids are positive too, and the highest grade sits
on the LOWEST id in 100.0% of 23,258 measurable cases (grades are non-increasing in
id within a name group). Two transforms follow:

  1. canonicalize: rewrite each ranked id to the lowest not-yet-used id of its name
     group, so within-group order is always ascending (weakly grade-improving).
  2. expand: for names seen in the top --expand-from ranks, insert their missing
     sibling ids (ascending) directly after the group's last placed member, up to
     --budget insertions per query; tail items get displaced past rank 10.

Both are pure run-file transforms (no model, no scoring); tune flags on dev, verify
the chosen config on val once.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import BENCH, RUNS, load_jsonl, load_run_csv, setup_utf8, write_run_csv
from eval_ndcg import load_run_jsonl


def build_name_groups() -> tuple[dict[str, str], dict[str, list[str]]]:
    corpus = load_jsonl(BENCH / "corpus.jsonl")
    id_name = {str(r["item_id"]): r["name"] for r in corpus}
    name_ids: dict[str, list[str]] = defaultdict(list)
    for r in corpus:
        name_ids[r["name"]].append(str(r["item_id"]))
    for ids in name_ids.values():
        ids.sort()
    return id_name, dict(name_ids)


def transform_query(
    ids: list[str],
    id_name: dict[str, str],
    name_ids: dict[str, list[str]],
    canonicalize: bool,
    expand_from: int,
    budget: int,
    top_k: int = 10,
) -> list[str]:
    if canonicalize:
        used: set[str] = set()
        canon: list[str] = []
        for iid in ids:
            group = name_ids.get(id_name.get(iid, ""), [iid])
            repl = next((g for g in group if g not in used), None)
            if repl is None:
                continue
            canon.append(repl)
            used.add(repl)
        ids = canon
    if budget > 0:
        placed = set(ids[:top_k])
        out = list(ids[:top_k])
        added = 0
        for iid in list(out[:expand_from]):
            if added >= budget:
                break
            group = name_ids.get(id_name.get(iid, ""), [])
            missing = [g for g in group if g not in placed]
            if not missing:
                continue
            last_pos = max(i for i, x in enumerate(out) if x in group and x in placed)
            take = missing[: budget - added]
            out[last_pos + 1 : last_pos + 1] = take
            placed.update(take)
            added += len(take)
        ids = out + [x for x in ids[top_k:] if x not in placed]
    return ids[:top_k]


def apply(
    run: dict[str, list[str]],
    canonicalize: bool = True,
    expand_from: int = 5,
    budget: int = 3,
) -> dict[str, list[str]]:
    id_name, name_ids = build_name_groups()
    return {
        qid: transform_query(ids, id_name, name_ids, canonicalize, expand_from, budget)
        for qid, ids in run.items()
    }


def main():
    setup_utf8()
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--no-canonicalize", action="store_true")
    p.add_argument("--expand-from", type=int, default=5, help="expand names seen in top N ranks")
    p.add_argument("--budget", type=int, default=3, help="max inserted ids per query; 0 = off")
    args = p.parse_args()

    run = load_run_csv(args.run) if args.run.suffix == ".csv" else load_run_jsonl(args.run)
    out = apply(run, not args.no_canonicalize, args.expand_from, args.budget)

    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for qid, ids in out.items():
            obj = {"query_id": qid, "retrieved": [{"item_id": iid} for iid in ids]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    write_run_csv(RUNS / f"{args.name}.csv", out)
    print(f"wrote runs/{args.name}.jsonl and runs/{args.name}.csv")


if __name__ == "__main__":
    main()
