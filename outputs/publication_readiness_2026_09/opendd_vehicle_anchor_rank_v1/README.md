# OpenDD: ranking alternatives to the additive two-edit proposal

Most of the [exact relative-motion headroom](../opendd_vehicle_relative_ceiling_v1/README.md)
is available without changing both recipients. I kept at least one recipient
from the additive proposal and trained a ranker to look for a better pair.
The two new rankers share the same inputs, candidate set, architecture,
initialization and training budget. Both learn risk and utility against CV
and OLS4; the second also predicts conservative trajectory and
relative-motion gains against the anchor and uses them to rank.

| Fixed two-edit comparison | Pass original checks and beat CV/OLS4 | Also improve on additive without conservative trajectory loss |
| --- | ---: | ---: |
| Archived additive | 2,975 | Not applicable |
| Archived unrestricted joint | 2,952 | 242 |
| Anchored, CV/OLS4 only | 3,134 | 373 |
| Anchored, with relative targets | 3,071 | 408 |

Each arm has 8,874 views from the same 493 recordings. The 35 extra
full-condition wins over the equal-capacity head are split +26, +6 and
+3 by seed. The extra wins do not imply higher average utility. All arms
retain two edits at the same location and peak radii.

| Relative-target head, direct paired gain | vs CV/OLS4-only head | vs archived unrestricted joint |
| --- | ---: | ---: |
| Available-step trajectory (pp) | +0.000705 [+0.000367, +0.000978] | +0.000475 [-0.000142, +0.001236] |
| Conservative trajectory (pp) | -0.003130 [-0.005627, -0.001217] | -0.005362 [-0.010279, -0.001560] |
| Conservative relative motion (native units) | -0.02713 [-0.03805, -0.01764] | -0.04207 [-0.06312, -0.02475] |

Against the frozen additive anchor itself, the relative-target head
loses **0.006904 pp** of conservative trajectory gain
[-0.013405, -0.001933] and **0.05793 native units** of conservative
relative-motion gain [-0.09130, -0.03063]. The direct comparisons
recompute disagreement for each pair of predictions; they are not
differences of two lower bounds. Intervals resample seven location
means 3,000 times after averaging seeds and calibration choices.

## Frozen calibration

| Held recording views | CV/OLS4-only head | Relative-target head |
| --- | ---: | ---: |
| Admitted to replace additive | 3 | 14 |
| Missing easy reference denominator | 2 | 7 |
| Pass all original risk checks | 0 | 6 |
| Also beat CV and OLS4 | 0 | 5 |
| Also improve on additive with no conservative trajectory loss | 0 | 1 |
| Active directions with all replacements meeting the full condition | 0 / 3 | 0 / 4 |

The observed-step improvement is small and does not survive the
conservative comparison. The new head finds more opportunities but
cannot reliably recognize them before seeing outcomes. Donor anchors
are randomly chosen from the past-only bank, while evaluation uses
learned additive anchors. That difference could hurt transfer and
was not isolated here. The 2% rule and current CV fallback for this
vehicle study stay unchanged. No Songdo or DroneCrowd confirmation
outcomes were opened.

This comparison concerns whether to substitute paths in an unchanged
forecast bank. It does not generate jointly consistent trajectories
as [ScePT](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_ScePT_Scene-Consistent_Policy-Based_Trajectory_Predictions_for_Planning_CVPR_2022_paper.html)
does, nor does its empirical score prefix give the expected-loss
guarantee of [conformal risk control](https://openreview.net/pdf?id=33XGfHLtZg).

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_anchor_rank.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_anchor_rank.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_anchor_rank_paired.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_anchor_rank.py tests/test_m3w_vehicle_relative_ceiling.py tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_opendd_pair_disagreement.py tests/test_m3w_reviewer_evidence.py
```

`fresh_run`: 126 arm64 PyTorch fits, 25,200 optimizer updates, 315 donor
batches with 3,051,933 sampled pair instances, and 126 held scoring
directions. Four compute threads, no data workers, checkpoints every
100 steps and per-cell PID receipts. Twenty-five focused tests passed.
`cached_verified`: the existing donor forecasters, cohorts and additive
and joint proposals, verified against their stored hashes and roster.
Weights and packets are local. No manuscript edit or deployment change.

[Protocol](protocol.md) | [Registration](registration.json) |
[Freeze](freeze.json) | [Scoring](readout.json) | [Direct contrasts](paired_readout.json).
