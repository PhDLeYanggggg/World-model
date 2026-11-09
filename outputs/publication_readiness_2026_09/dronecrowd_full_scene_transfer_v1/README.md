# DroneCrowd: Can a Source-Separated Harm Head Preserve Useful Coverage?

The [full-scene motion controls](../dronecrowd_motion_floor_v1/README.md)
found that fitting the origin improves on OLS4, yet only 8/40
recordings meet both accuracy and the original risk checks.
A hindsight witness on the same analytic paths passes 40/40.
This tests whether learning the twelve-step gain and harm on
different sources can recover that gap without using target labels.

## Fixed Paths, Different Decisions

I train three seeds on the existing MovingDroneCrowd++ A-only
donor population: 125,540 requested rows, 34 recordings and
14 source groups. The original image screen covers the fitting
roster; it does not prove physical independence. The target is the
same 40 previously exposed DroneCrowd calibration recordings,
15 source groups, 1,757,584 requested rows and 11,240 queries.
No target outcomes train the models, fit a normalizer, or tune a
threshold. The three heads freeze with the choices before scoring.

All decision arms use the same CV, OLS4, componentwise median slope,
fitted origin and pooled origin. The head predicts twelve-step
available-label gain, positive all/easy harm including the
unknown-actor debit, and nonnegative CV reference masses.
The reference output is shared across an actor's candidates.
Scale and features come from the observed eight positions,
candidate geometry and retrospective one-step history probes.
Nonnegative terms use an untransformed Poisson mean loss;
signed gain uses Huber loss. The predictions are converted
back into native units before allocating a query's budget.

- **Gain:** choose a positive predicted gain for each actor.
- **Individual:** add projected all/easy positive-harm limits per actor.
- **Scene budget:** rank candidate edits by gain minus predicted harm
  excess and take the largest prefix with positive reference mass and
  predicted positive-harm ratios within 2%. No interaction term.
- **Matched gain:** rank by gain alone but copy the scene rule's exact
  number of changed paths in every query. This matches counts, not peaks
  or realized risk. It does not replace the prior exact two-edit study.

The forecast-side gate is not a safety certificate. Evaluation
still uses the original CV-referenced risk scorer, both motion
floors, and missing-label accounting. Source-level intervals
average the three seeds and use 3,000 paired source draws.
All data are annotation-step and image-space only.

## Transfer Result

| Policy | Edits / 5,272,752 row-seed instances | Defined original risk passes / 120 | Easy reference undefined | Both floors and risk | Lower gain vs CV (pp) | Lower gain vs OLS4 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Gain | 3,956,997 | 24 | 0 | 22 | +49.909 | -1.384 |
| Individual | 13,580 | 36 | 57 | 0 | +0.063 | -51.484 |
| Scene budget | 25 | 9 | 107 | 0 | +0.00017 | -51.548 |
| Matched gain | 25 | 12 | 107 | 0 | +0.00033 | -51.547 |

The gain-only model has coverage but loses to OLS4
by **-1.384 pp [-2.022, -0.759]**. It has mean available
ADE/FDE 1.4670/2.6328 px and a worst recording p95 ADE of
7.0390 px, all worse than the fitted-origin comparator
(1.4157/2.5406/6.6690 px). Its worst signed easy change
is still an improvement (-26.83%), but the positive
easy-harm ratio exceeds 2% in 90/120 views. Those
metrics answer different questions.

The scene-budget rule switches only in seed 17. It loses
three risk passes relative to the matched gain control
and has four positive easy-harm violations instead of one.
The count-matched direct trajectory contrast is
**-0.00016 pp [-0.00064, +0.00046]**. This does not
demonstrate a benefit of the scene-level allocation;
107/120 views lack a defined easy reference in both arms.
The 120 views are 40 recordings repeated across three
seeds, not 120 independent trials.

### Observed-input mismatch

An independent input-only pass uses the exact frozen target roster,
with no target labels or model changes:

| Quantity (actor-query weighted) | Donor median | Calibration median | Calibration outside donor 1st–99th percentile |
|---|---:|---:|---:|
| Query size | 103 | 177 | 49.0% |
| Mean step in pixels, floored at 0.05 | 15.59 | 0.53 | 87.3% |
| Past CV probe error divided by actor scale | 0.210 | 1.340 | 64.0% |
| Peak OLS4–CV difference divided by actor scale | 1.603 | 8.400 | 45.2% |

These are descriptive differences in exported observation
inputs, not physical speed or evidence that any one difference
caused the failure. Matching the eight/twelve step counts did
not make the trajectory signal or forecast disagreement comparable.

### The small predicted budget is not viable

I count the frozen path changes in each query, then let an
impossible oracle perfectly predict any chosen actors at that
count. As in the earlier bound, I ignore all risk constraints
and omit the nonnegative missing-label debit in the OLS4 contrast.
This only raises the upper bound; it does not produce an eligible
policy. At the scene rule's count, the upper bound is
**-51.417 pp [-56.014, -46.593] vs OLS4**, negative in every
source group. The individual rule's upper bound is also negative
in 15/15 groups (-50.679 pp). In contrast, gain-only has positive
headroom in 15/15 groups; its failure is not explained by count.

The evidence points to two separate fixes: a better predictor of
twelve-step gains under different observed sampling/noise regimes,
and an allocation that retains enough coverage to compete with
OLS4. Relaxing a test threshold would not demonstrate either.
Keep the fitted-origin baseline, keep the original 2% rule, and
test a pre-registered donor-only observation augmentation before
trying another release gate. Do not tune on the 15 target groups
or open the independent confirmation partition to select a model.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_full_scene_harm_transfer.py train
.venv-pytorch/bin/python scripts/run_m3w_full_scene_harm_transfer.py freeze
.venv-pytorch/bin/python scripts/run_m3w_full_scene_harm_transfer.py score
.venv-pytorch/bin/python scripts/run_m3w_full_scene_harm_transfer.py replay
.venv-pytorch/bin/python scripts/audit_m3w_full_scene_harm_input_shift.py
.venv-pytorch/bin/python scripts/audit_m3w_full_scene_harm_coverage.py
.venv-pytorch/bin/python scripts/audit_m3w_full_scene_harm_coverage.py --replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_full_scene_harm_transfer.py \
    tests/test_m3w_dronecrowd*.py tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 PyTorch, four compute threads, zero workers.
Three 600-step donor fits took 5.69 s including data prep;
freeze took 23.48 s and scoring 103.66 s (2.28 GB maximum
RSS). The input pass and count bound took 15.11 and 10.72 s.
The focused suite passed 183 tests.
Atomic checkpoint every 100 updates saves optimizer and
sampler; per-recording receipts support resuming freeze, score
and coverage. Replay recomputes the same public scalars without
reopening labels. No CREATE job, manuscript change, deployment,
Stage5C, SMC or independent confirmation.

[Registration](registration.json) · [Training](training.json) ·
[Forecast freeze](freeze.json) · [Scoring](analysis.json) ·
[Input shift](input_shift.json) · [Coverage bound](coverage.json)
