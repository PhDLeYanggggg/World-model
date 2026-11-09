# DroneCrowd: Donor-Only Observation Augmentation at Fixed Edit Counts

I test whether an observed motion mismatch explains the earlier
failure to transfer a twelve-step gain and harm model. The prior
[input audit](../dronecrowd_full_scene_transfer_v1/README.md)
found that 87.3% of calibration actor-query observations fell outside
the donor 1st–99th percentile of mean observed step size. That is an
image-space mismatch, not a difference in verified physical speed.

## Comparison

I fit the same architecture on the same 125,540 requested training
rows from 14 MovingDroneCrowd++ source groups. For each of three seeds,
the plain, scale-only and scale-plus-noise models start from the same
weights and use the same donor minibatch draws and update count.
Half of the donor queries are left unchanged. On the others, a shared
query factor is sampled log-uniformly from 1/64 to 1 and applied to the
observed and future coordinates. The noise arm also adds zero-mean
Gaussian jitter with standard deviation 0.5 image-coordinate units
to each observed coordinate, then recomputes the future offsets
relative to the perturbed final observation. Future positions and
validity masks are not jittered. No target statistics set the factors.

The unaugmented target roster is the previously opened DroneCrowd
calibration partition: 40 recordings, 15 source groups, 1,757,584
requested rows, eight observed and twelve forecast annotation steps.
For the primary comparison, each augmented model picks exactly as many
changed paths in each query and seed as the original gain-only model.
Paths come from the same CV, OLS4, median slope, fitted origin and
pooled origin bank. Matching counts does not match actors, peak
displacements or realized harm. The natural gain-only arms and the
noise arm with the original projected scene budget are secondary.

The model weights and target choices were frozen before scoring.
Scoring uses the original CV-referenced 2% risk rules and missing-label
accounting, and requires improvement over both CV and OLS4 for the
joint criterion. Intervals use 3,000 paired source-group bootstrap
draws after averaging seeds; 120 recording-seed views are not 120
independent observations. This is an exploratory calibration comparison,
not independent confirmation or a new release gate.

## What the Augmentations Change

The input-only replay measures the fraction of target actor-query rows
outside the donor 1st–99th percentile. Ranges cover the three training
seeds, not confidence intervals. Query size is unchanged (49.0%).

| Observed statistic | Plain | Scale only | Scale + noise |
|---|---:|---:|---:|
| Mean step size | 87.31% | 14.34% | 66.32–66.50% |
| CV history-probe error / observed step scale | 63.99% | 64.09–64.23% | 31.08–31.11% |
| OLS4–CV peak disagreement / observed step scale | 45.25% | 45.27–45.86% | 33.89–33.90% |

Scaling brings step sizes into range without changing much of the
relative forecast geometry. Adding independent noise brings more
relative probe errors into range, but worsens step-size overlap
compared with scaling alone. Neither augmentation reproduces both
input distributions or changes the observed actor roster.

## Results

| Policy | Changed paths | Original risk passes / 120 | Easy harm > 2% / 120 | Both accuracy floors and risk / 120 | Conservative gain vs OLS4 (pp) |
|---|---:|---:|---:|---:|---:|
| Plain gain | 3,956,997 | 24 | 90 | 22 | -1.384 |
| Scale, matched count | 3,956,997 | 24 | 89 | 19 | -1.933 |
| Scale + noise, matched count | 3,956,997 | 24 | 90 | 0 | -0.932 |
| Scale, own gain choices | 3,649,408 | 23 | 88 | 1 | -6.854 |
| Scale + noise, own gain choices | 3,962,145 | 24 | 90 | 0 | -0.859 |
| Scale + noise, projected scene budget | 28,019 | 88 | 28 | 0 | -50.824 |

At equal counts, scale-only loses **-0.551 pp [-0.672, -0.419]**
against the original gain model. The prespecified noise comparison
estimates **+0.450 pp [-0.117, +1.013]**. That interval includes
zero and the OLS4 contrast is still negative, **-0.932 pp
[-1.217, -0.708]**, with no positive source group. Mean available
ADE improves from 1.4670 to 1.4622 px; available-endpoint FDE
worsens from 2.6328 to 2.6373 px. The previously evaluated fitted-origin
control remains stronger: +0.391 pp vs OLS4, although only 8/40
recordings pass both floors and risk for that deterministic control.

The projected scene budget goes from 25 to 28,019 edits and from
9/120 to 88/120 original risk passes with noise augmentation.
Three views still lack easy-risk support and no view passes both
accuracy floors and risk. This is a different-count comparison,
not evidence that scene allocation beats independent selection at
the same budget. Signed easy degradation and positive harm on
selected easy rows are different measures; the latter still
exceeds 2% in 28 views. Nothing is released.

An exact-count hindsight bound explains why 88 risk passes are
not enough. Only 0.53% of actor-seed rows change. Even if the rule
could pick any actors within its frozen count in every query and
forecast them perfectly, its OLS4 gain could not exceed
**-50.179 pp [-54.547, -45.551]** on average, negative in all 15
source groups. This calculation ignores risk and omits the OLS4
comparison's missing-label debit to give the rule every advantage.
The bottleneck is not just ranking within that budget.

Keep the fitted-origin comparator and the original risk rule.
The next useful development comparison is to learn which edits improve
on the stronger OLS4 path, and to check attainable accuracy at the
resulting coverage before calibrating a release. More threshold search
on the same low-coverage policy cannot fix its ceiling.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_observation_augmentation_transfer.py train
.venv-pytorch/bin/python scripts/run_m3w_observation_augmentation_transfer.py freeze
.venv-pytorch/bin/python scripts/run_m3w_observation_augmentation_transfer.py score
.venv-pytorch/bin/python scripts/run_m3w_observation_augmentation_transfer.py replay
.venv-pytorch/bin/python scripts/audit_m3w_observation_augmentation_shift.py
.venv-pytorch/bin/python scripts/audit_m3w_observation_augmentation_coverage.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_observation_augmentation.py \
    tests/test_m3w_full_scene_harm_transfer.py tests/test_m3w_dronecrowd*.py \
    tests/test_m3w_recording_forecast_comparison.py tests/test_m3w_perfect_switch_ceiling.py
```

Native arm64 PyTorch, four compute threads, zero workers. The two
augmented fits for each of three seeds took 13.01 s total including
data prep. Inference and freezing took 31.81 s, scoring took 180.79 s
(1.95 GB maximum RSS), the input-only replay took 17.72 s, and the count
bound took 9.92 s. Atomic checkpoints
save optimizer and sampler every 100 of 600 updates per seed.
Per-recording receipts resume freeze and scoring. Replay reads only
previously computed recording summaries. The focused suite passed
187 tests. No new CREATE job, deployment, manuscript edit, Stage5C,
SMC or access to the 30 independent confirmation groups.

[Registration](registration.json) · [Training](training.json) ·
[Forecast freeze](freeze.json) · [Scoring](analysis.json) ·
[Input-only audit](input_shift.json) · [Exact-count upper bound](coverage.json)
