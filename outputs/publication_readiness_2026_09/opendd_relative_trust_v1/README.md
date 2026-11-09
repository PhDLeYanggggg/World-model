# OpenDD: Smaller Joint Edits Lose Useful Alternatives

I tested whether joint selection can preserve its interaction benefit
while reducing changes where future relative positions are missing. This
is an 8-observation, 12-prediction-step OpenDD development experiment.
Distances are in native coordinates.

## What Changed

For each window, measure the mean Euclidean difference between the
candidate and the common additive control in pair-relative coordinates,
over all pairs and forecast steps. Keep only candidates whose difference
is at most half that of the original joint policy's selected forecast.
Neither the bound nor the original selection reads future labels or
masks. The legal additive candidate has zero difference, so the shortlist
is never empty. Retrieval and ranking use the same two frozen heads
within this restricted set. Fewer than eight eligible candidates are
allowed; none are duplicated to fill the list.

All 21 location-seed folds replayed the uncapped shortlist, forecast and
score exactly. Both arms make two edits with the same sorted peak
displacements from CV in each eligible window, totaling 966 edits across
159 recording-seed views. The eligible recipient choices are unchanged,
but the selected recipients need not be identical. Training locations,
separate calibration locations, held development locations and the
original recording-level 2% rule all stay fixed. No model was refit.

## What the Restriction Cost

The restriction leaves 10,932 of 192,801 held candidates. In 168 of 483
query-seed views fewer than eight survive; 53 selections are exactly
the additive path. Counts below are query-seed views, not independent
recordings. Hindsight qualification is an audit, not an inference input.

| Opportunity or selection | Calibration | Held |
|---|---:|---:|
| Any qualifying candidate in the original menu | 141 | 150 |
| Any qualifying candidate after the cap | 42 | 36 |
| Uncapped / capped shortlist retains a qualifying candidate | 56 / 20 | 51 / 23 |
| Uncapped / capped policy selects a qualifying candidate | 20 / 6 | 22 / 5 |

| Forced arm | Known ADE | Available FDE | Mean recording p95 ADE | Risk passes | Easy over 2% | Undefined edited easy |
|---|---:|---:|---:|---:|---:|---:|
| Uncapped joint | 0.54767 | 1.25386 | 1.39108 | 13/159 | 43 | 51 |
| Half-cap joint | 0.54814 | 1.25497 | 1.38947 | 15/159 | 48 | 44 |

CV remains the tested causal floor (ADE 0.54726, FDE 1.25329).
Hard gain versus CV falls from -0.026% to -0.165%. The count of
recording-seed views with qualified positive full-grid gain over
additive falls from three to zero. Neither arm passes any of its
483 nonempty calibration breakpoints. Returning CV with no intervention
does not count as a risk pass.

## What the Cap Fixed and What It Did Not

| Contrast | Known term | Missing-step debit | Full-grid lower gain |
|---|---:|---:|---:|
| Uncapped vs additive | +0.003167 | 0.010484 | -0.007317 |
| Half-cap vs additive | -0.000492 | 0.004359 | -0.004850 |
| Half-cap vs uncapped, directly paired | -0.003658 | 0.007707 | -0.011365 |

All three terms use the same full-grid denominator. Against the common
additive control, the realized debit falls by 58.4%, but the known
component also falls by 0.003658. The difference between the two bounds
relative to additive is +0.002467 (95% interval [-0.006250, +0.012067]);
it is **not** a lower bound for the head-to-head improvement. In the
direct comparison, moving back toward the additive forecast also
changes unknown relative positions. The directly paired lower gain is
**-0.011365 [-0.023943, -0.002163]**. The worst location is rdb5
(-0.029907). The paired conservative trajectory gain also falls
**-0.937 pp [-1.943, -0.095]**. Intervals use 3,000 location bootstrap
draws after averaging seeds within recording; only four locations have
both required pair denominators. This is not independent confirmation.

## Implication for the Paper

Lowering the geometric upper bound on missing-step debit is possible
without spending fewer edits, but it removed most of the useful slates
in this menu. This rules out the tested half-cap as a fix, not all
geometry-aware policies. The next discriminating experiment is to
measure headroom and train on better-verified pair labels in a
development source, then test whether a learned support signal can
preserve known benefit at the same edit peaks. A stronger joint
forecaster is still a necessary reference: [Scene Transformer](https://arxiv.org/abs/2106.08417)
and [ScePT](https://research.nvidia.com/publication/2022-06_scept-scene-consistent-policy-based-trajectory-predictions-planning)
address the joint forecast itself. The question here is when to use
such a forecast instead of a strong motion baseline at a matched
budget. This empirical calibration is not the expected-risk guarantee
proved under the conditions of [Conformal Risk Control](https://arxiv.org/abs/2208.02814).

In order of priority: (1) demonstrate positive relative benefit with
the original risk rule on separate development calibration recordings;
(2) compare the frozen policy against a public joint predictor and the
matched additive control; (3) run untouched external confirmation only
after a nonempty policy qualifies. Nothing in this experiment changes
the DroneCrowd holdout, prior WEPDTOF result or manuscript status.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_relative_trust.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_relative_trust.py score
```

Native arm64 CPU, four Torch threads, no workers. Fresh freeze and score
took 32.52 s and 77.21 s, respectively, under 837 MB RSS. The initial
receipt write failed because two role keys collided; the fix separated
role identifiers from payloads. The measured run started from a new
registration and completed. Fold receipts are resumable and a PID/phase
heartbeat is written locally. Twenty-six related tests passed. No CREATE
job, Stage5C, SMC or reserved outcome access. [Registration](registration.json),
[freeze](freeze.json), [calibration](calibration.json), [metrics](summary.json).
