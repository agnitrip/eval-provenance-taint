# Results manifest

Which file backs which number in the paper "Measuring Eval-Set Self-Contamination: A Synthetic
Benchmark for Provenance Taint in On-Device Verifier Pipelines" (IEEE ICCE 2027).
Every file stores per-episode verdicts, so each reported rate can be recomputed without model calls.

| Paper element | Files |
|---|---|
| Table II (adoption and false alarm per condition), per-seed results, McNemar tests | `{gpt4omini,gpt41,haiku,sonnet}_ab_s{7,17,29}.json` (4 models x 3 seeds, n=80 each) |
| Table III (tag_only adoption by task) | same `*_ab_s*.json` files, grouped by `template` |
| Robustness: taint that is not planted (natural mode) | `nat_*.json` |
| Robustness: second tag_only wording | `*_tagv2_s7.json`, compared with `*_ab_s7.json` (same seed; Stage A was re-run) |

## Earlier runs, not used in the paper's tables
Kept for transparency. They come from before the 2x2 ablation and use a 3-call episode design.

- `pilot_haiku.json` (n=12), `haiku_n60.json`, `haiku_n80.json`, `sonnet_n60.json`, `sonnet_n80.json`

## Scoring notes
- A grading reply with no parseable verdict is scored `incorrect` (see `eptaint/verifier.py`), so it
  counts as adoption. Raw replies are not stored.
- Every candidate under grading is the true answer.
