# DroneCrowd: Learning Which Predictions Improve on OLS4

The [previous coverage bound](../dronecrowd_observation_augmentation_v1/README.md)
showed that rare edits to CV cannot match the full OLS4 forecast, even
with perfect predictions at the frozen counts. Here I start both models
at OLS4 and test whether changing the gain target helps identify useful
replacements. OLS4 is a comparator and a default path for this diagnostic,
not a certified fallback under the CV-referenced risk contract.

## Design

The two models see the same native histories and candidate paths, start
with identical weights, and use the same minibatches in all three seeds.
Training uses 125,540 actor-query rows from 14 screened MovingDroneCrowd++
donor groups. The control learns signed available-label gain relative to
CV; the alternative learns the same quantity relative to OLS4. A wholly
unlabelled donor row receives the appropriate path-disagreement debit in
each gain target. Both versions still learn CV-referenced positive all
and easy harm, and CV reference mass. The control's training reproduces
the original CV model weights bit-for-bit. The training objective uses
Huber for signed gain; it does not certify a conditional mean or risk.

At inference both models subtract their estimated gain of retaining OLS4.
The available replacements are median slope, fitted origin and pooled
origin, from the same past eight observations. The primary comparison
matches the number of actual OLS4 replacements in each query and seed
to the CV-target control. A separate joint rule maximizes predicted
gain among assignments from a fixed two-price grid that meet the two
projected CV risk constraints. In its risk sums even the unchanged
OLS4 paths count when they differ from CV. Unsupported queries keep
OLS4 for diagnosis. The joint rule is also compared with independent
gain ordering at exactly its own replacement counts, not matched actors,
peak offsets or realized harm.

All scoring uses the same previously opened 40 DroneCrowd recordings
from 15 calibration source groups, eight observed and twelve forecast
annotation steps, and native image coordinates. The original CV reference,
2% tolerances and missing-label debit are unchanged. The joint criterion
requires conservative positive gain against both CV and OLS4 plus passing
the original risk checks on the entire recording. Differences use 3,000
paired source-group bootstrap draws across three seeds. These are
development results and do not change the original sparse two-edit study.

## Results

| Rule (all start at OLS4) | OLS4 replacements | Original risk passes / 120 | Easy harm > 2% / 120 | Both floors and risk / 120 | Conservative gain vs OLS4 (pp) |
|---|---:|---:|---:|---:|---:|
| CV-target independent | 3,824,730 | 24 | 90 | 24 | +0.144 |
| OLS4-target, matched count | 3,824,730 | 27 | 74 | 6 | -0.380 |
| OLS4-target, own count | 1,304,576 | 28 | 76 | 6 | -0.236 |
| OLS4-target joint | 0 | 24 | 84 | 0 | 0.000 |
| OLS4-target independent, joint count | 0 | 24 | 84 | 0 | 0.000 |
| OLS4 alone | 0 | 24 | 84 | 0 | 0.000 |

The direct, equal-count effect of changing the gain reference is
**-0.529 pp [-0.954, -0.070]**, positive in only 4/15 groups. The CV
control's +0.144 pp vs OLS4 has a 95% interval of [-0.070, +0.354].
Available-label ADE and available-endpoint FDE were 1.4241 and 2.5527 px
for the CV control, 1.4471 and 2.5778 px for the equal-count OLS4 target,
and 1.4295 and 2.5462 px for OLS4 alone. The OLS4-target result does
not beat the [deterministic fitted-origin control](../dronecrowd_motion_floor_v1/README.md).

The joint controller makes no replacements. Its matched control is empty;
the exact trajectory tie is not evidence of joint-policy benefit or safety.
The OLS4 risk pass count represents eight recordings repeated across
three seeds, not 24 independent passes. An input-only risk relaxation
minimizes each predicted excess over every actor's full menu separately,
ignoring gain restrictions and allowing the two constraints to disagree
on the chosen path. Both minima are positive in all 11,240 queries in
each seed. A larger price grid could not fix these predictions.

The hindsight checks identify two separate limits. At the control's
same replacement counts, the exact analytic-menu maximum is **+4.304 pp
[+4.020, +4.603] vs OLS4** with the missing-label debit and without risk
constraints. All 15 groups have headroom. The poor gain result is not
explained by a lack of choices at those counts. The zero-count joint
rule has zero OLS4 headroom by construction.

With outcomes available, OLS4 itself meets both query-local risk ratios
with nonzero support in **4,199 of 11,240 queries**. The separate-minima
relaxation still rules out **6,007 queries** for every path assignment
from the menu under that query-local restriction. The donor harm head
rejects many viable queries, but a substantial fraction is genuinely
blocked under the more restrictive query rule. These are diagnostics;
the approved deployment criterion remains at recording level.

The next useful comparison is a recording-level risk model calibrated
on the approved calibration sources, with a fixed candidate bank and a
matched independent control. The score must still enforce both original
2% risk limits and easy support on each recording. A larger model, wider
query price grid or changed risk limit is not justified by this result.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_ols_relative_gain.py train
.venv-pytorch/bin/python scripts/run_m3w_ols_relative_gain.py freeze
.venv-pytorch/bin/python scripts/run_m3w_ols_relative_gain.py score
.venv-pytorch/bin/python scripts/run_m3w_ols_relative_gain.py replay
.venv-pytorch/bin/python scripts/audit_m3w_ols_predicted_feasibility.py
.venv-pytorch/bin/python scripts/audit_m3w_ols_relative_gain_ceiling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_ols_relative_gain.py \
    tests/test_m3w_observation_augmentation.py tests/test_m3w_full_scene_harm_transfer.py \
    tests/test_m3w_dronecrowd*.py tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 PyTorch, four compute threads, zero workers. The six fits
took 7.62 seconds with data prep. Freeze took 47.64 seconds, scoring
227.73 seconds (2.15 GB maximum RSS), and the predicted-risk audit
20.87 seconds; the outcome-based ceiling took 15.01 seconds.
Checkpoints include optimizer and
sampler state every 100 of 600 updates; freeze and score resume from
recording receipts. Replay reads score summaries, not target labels.
The focused suite passed 190 tests. No deployment, independent
confirmation access, new CREATE job, paper edit, Stage5C or SMC.

[Registration](registration.json) · [Training](training.json) ·
[Forecast freeze](freeze.json) · [Scoring](analysis.json) ·
[Predicted feasibility](predicted_feasibility.json) · [Menu ceiling](ceiling.json)
