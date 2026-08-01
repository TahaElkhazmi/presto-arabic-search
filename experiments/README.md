# Experiments

One-off measurement scripts. **Nothing here is on the reproduction path** - the submitted
score is regenerated entirely by `src/`, per section 1 of the repository README.

These files are the evidence behind the experimental record in `TECHNICAL_REPORT.md`,
including every negative result. They are kept because a claim like "we measured this
mechanism and it failed for this reason" should be checkable, not taken on trust.

Each script states its hypothesis, its method and usually its outcome in the module
docstring, so the docstrings are the fastest way to read the experimental history.

## What is here

| prefix | count | what it does |
|---|---|---|
| `probe_*` | 48 | screens a candidate mechanism, usually before any end-to-end run |
| `validate_*` | 13 | full multi-slice validation of a change that passed its screen |
| `grid_*` / `retune_*` | 11 | parameter grids and post-model-swap re-fits |
| `verify_*` / `test_*` | 8 | checks of a stated fact or an implementation detail |
| `eval_*` / `screen_*` / `size_*` | 9 | scoring harnesses and cost/headroom estimates |
| everything else | 28 | data export, autopsies, one-off diagnostics |

## Running one

They expect to be run from the repository root with the project virtualenv, and they add
`src/` to `sys.path` themselves:

```bash
./.venv/Scripts/python.exe experiments/probe_headroom_autopsy.py --run submission_v7synth_stack5.csv
```

Most read `data/benchmark/data/candidates.jsonl` for **measurement only**, which is the use
competition rule 2.4.b permits; none of them fits model weights. Many also need the
cross-encoder score cache from the Hugging Face audit bundle.

## A note on reading them

These were written during a two-week competition, one per question, and they are not a
library: there is duplication between them, and a few reference intermediate model
generations that were later superseded. They are preserved as run rather than tidied up,
because rewriting them after the fact would make the record less trustworthy, not more.
