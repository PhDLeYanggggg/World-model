# AV2: Explicit Forecast Costs Improve Some Risk Counts, Not Joint Release

I kept the [history ranker](../av2_history_rank_v1/README.md),
the candidate trajectories and the additive reference fixed.
The question was whether predicting where observed gains are
lost to missing labels and easy-case harm would let the same
ranker choose a better pair of edits.

## Design

The new head predicts five terms: observed trajectory gain,
missing trajectory disagreement, observed pair-relative gain,
missing pair-relative disagreement, and signed easy excess
above the original 2% allowance. The first four compare to
the same additive path, and the easy term compares to CV.
Observed gains and missing disagreement use the same full-grid
denominators. Geometric displacement bounds normalize the
targets; the bounds and all features use past observations
and candidate paths, not future targets or availability masks.
Future labels enter only supervision, calibration and scoring.

The three reranking variants share one fitted cost head. Starting from
the unchanged history score, I successively apply soft factors
for predicted observed gain, gain net of missing disagreement,
and signed easy excess. These are ranking factors, not safety
certificates. The full model was designated primary before
scoring. The release rule still tests every nonempty score
prefix with the original dual-baseline, pair comparison,
and 2% checks for both arms.

Six city folds, seeds 17/29/43, 896 added development scenarios,
eight observed steps and twelve forecast steps. For each cost
training donor, a nested additive model, its EqMotion forecast
producer, and the cutoff used by the action model exclude that
donor city as well as the outer calibration and held cities.
The outer additive reference is not retrained: calibration
and held inference use the identical frozen comparator.
The stored history choices and scores replay exactly.

I stopped an initial preflight after identifying that the
nested action cutoff still included the recipient donor.
The registered v2 run recomputes it on the three remaining
donor cities; no preflight fits or outputs enter this result.
All 72 final inner checkpoint identities and cutoffs replayed.

## Matched Evaluation

| Forced decision | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / 1,932 | Signed easy violations | Both arms qualify / 48 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | 0 | No edits | No edits | Not applicable |
| OLS4 | 0.27864 | 0.61665 | -22.03 | -10.91 | Different budget | Different budget | Not applicable |
| Additive, fixed | 0.24574 | 0.56860 | -1.07 | +0.56 | 143 | 242 | Reference |
| History, fixed | 0.24761 | 0.57486 | -1.60 | +0.44 | 137 | 383 | 5 |
| Observed gain only | 0.24711 | 0.57410 | -1.43 | +0.52 | 156 | 349 | 5 |
| Add missing-step debit | 0.24603 | 0.57120 | -1.23 | +0.67 | 157 | 346 | 2 |
| Add signed easy, primary | 0.24602 | 0.57121 | -1.21 | +0.62 | 158 | 346 | 3 |

All matched arms made 2,952 edits. The primary selects the
same path as additive in 339 of 1,020 two-slot views, compared
with 25 for the history ranker. The full model reduces some
easy violations relative to history, but still exceeds the
additive count by 104. Its edited-easy ratio is undefined
in 1,343 views; those are not passes.

| Primary head-to-head contrast | Full-grid paired estimate [95% CI] |
|---|---:|
| Pair lower gain vs history, native | -0.001051 [-0.001509, -0.000635] |
| Pair lower gain vs additive, native | -0.004251 [-0.006211, -0.002479] |
| Trajectory lower gain vs history, pp | -0.146 [-0.524, +0.176] |
| Trajectory lower gain vs additive, pp | -0.607 [-1.120, -0.024] |

The primary's direct trajectory gain over history on observed
steps is +0.321 pp, before charging a 0.466 pp missing-step
disagreement debit. The complete-scenario contrast against
history is +0.196 pp [-0.213, +0.612]. Intervals use 3,000
paired city bootstrap draws, averaging seeds within scenario
first. These are six-city development results, not independent
confirmation and not claims about errors on unobserved labels.

## Where Calibration Fails

The exact audit counts 315 of 1,020 calibration views with
a qualifying joint option and 57 with an eligible fixed
additive path. Their intersection has **44 views, 30 unique
scenarios**, containing 823 qualified options. The full
model selects a qualifying path in only **1 of those 44**,
versus **2** for history. No arm passed any of its 1,020
nonempty calibration prefixes. The released forecast is CV;
this is no deployment gain.

On calibration the predicted missing **pair** debit reduces
normalized MAE from 0.0897 (always zero) to 0.0814. Missing
**trajectory** debit is worse than zero (0.0328 vs 0.0285),
and signed easy error nearly doubles (0.0160 vs 0.0081).
The extra factors therefore do not reliably forecast the
rare outcomes that decide whether both arms can be released.
The next test should learn additive eligibility as a
scenario-level support target and exact relative qualification
as a candidate-level target, holding the comparator, menu,
and thresholds protocol fixed. Compare top retrieval on the
44 opportunities before interpreting any release gain.

## Reproduce

Requires the local inputs and checkpoints from the
[city-excluded study](../av2_oof_selection_v1/README.md)
and [history ablation](../av2_history_rank_v1/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_forecast_cost.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_forecast_cost.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_forecast_cost_calibration.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

72 nested reference fits and 18 cost fits ran on native arm64
PyTorch with four compute threads and no data workers. Training
and freeze: 307.24 s, peak RSS 1.31 GB; scoring: 10.60 s;
calibration audit: 39.51 s. Checkpoints every 40 steps retain
optimizer and sampler state; reissue `fit` to resume. A replay
verified all completed receipts in 7.20 s without new updates.
The local heartbeat records PID, model and step. Forty focused
tests passed. No CREATE job, no reserved confirmation access,
no Stage5C or SMC, and no manuscript or deployment change.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Calibration capacity](calibration_capacity.json)
