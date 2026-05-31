# Stage43-S T100 Source Coverage Preflight

- source: `fresh_stage43_s_t100_source_coverage_preflight`
- result_source: `fresh_h100_source_coverage_preflight`
- gate: `8 / 8`
- verdict: `stage43_s_t100_source_coverage_preflight_pass`
- h100 source count: `13`
- feasible families: `TrajNet_crowds`
- blocked families: `ETH_UCY, TrajNet_biwi, UCY`
- rebuild source-stable h100 split recommended: `True`
- uniform t100 blocker remains: `True`

## Family Summary

| family | sources | eligible sources | h100 rows | feasible | reason | current train/val/test sources |
| --- | ---: | ---: | ---: | --- | --- | --- |
| ETH_UCY | 2 | 2 | 5174 | `False` | `blocked_cannot_hold_train_val_test_with_source_stable_validation` | train:1/2614; val:1/2560; test:0/0 |
| TrajNet_biwi | 1 | 1 | 1160 | `False` | `blocked_too_few_h100_sources` | train:0/0; val:0/0; test:1/1160 |
| TrajNet_crowds | 8 | 8 | 27812 | `True` | `feasible_with_source_level_resplit` | train:6/20764; val:1/5608; test:1/1440 |
| UCY | 2 | 2 | 22598 | `False` | `blocked_cannot_hold_train_val_test_with_source_stable_validation` | train:0/0; val:1/7128; test:1/15470 |

## Proposed Source-Level Split Preflight

| family | status | train sources | val sources | test sources |
| --- | --- | --- | --- | --- |
| ETH_UCY | `not_feasible` | `none` | `none` | `none` |
| TrajNet_biwi | `not_feasible` | `none` | `none` | `none` |
| TrajNet_crowds | `feasible` | `OpenTraj/datasets/TrajNet/Train/crowds/students001.txt, OpenTraj/datasets/UCY/zara02/obsmat.txt, OpenTraj/datasets/UCY/zara01/obsmat.txt, OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara02.txt, OpenTraj/datasets/UCY/zara03/crowds_zara03.txt` | `OpenTraj/datasets/TrajNet/Train/crowds/students003.txt, OpenTraj/datasets/TrajNet/Train/crowds/arxiepiskopi1.txt` | `OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` |
| UCY | `not_feasible` | `none` | `none` | `none` |

## Interpretation

Stage43-S does not rewrite data or tune test thresholds. It shows whether h100 can support source-stable validation. In the current cache, only TrajNet_crowds has enough total h100 sources to try a new source-level split with two validation sources; ETH_UCY, TrajNet_biwi, and UCY remain source-scarce. Uniform t100 success therefore still needs more h100 source coverage or a separately validated per-source strategy.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
