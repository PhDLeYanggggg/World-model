# Why did the learned velocity adjustment stay close to OLS4?

I held the near-neighbor model fixed and varied only two parts of the
loss: whether to debit missing steps by trajectory disagreement and
whether to double the weight on positive per-row harm. Masked steps
contribute zero to the masked losses, but the denominator stays at
twelve. All fits start at OLS4.

There are 81 fresh fits (three loss variants, three seeds, nine held
fitting sources) and 27 verified cached fits for the original loss.
The original and new checkpoint records agree on sampling counts and
generator state at the same 800-update budget. Evaluation uses all
69,746 fitting windows, or the 45,330 with every step labeled for the
complete-label diagnostic. The held source is not used to train its
forecaster. These repeatedly used fitting sources are not independent
test data. See the [registered design](protocol.md).

## The training loss changes the forecast, but does not fix it

Numbers are CV-error-normalized points versus OLS4. The complete-label
column has its own CV denominator. The full-grid lower score debits
missing steps by forecast disagreement. Intervals use 3,000 source
resamples after averaging the three seeds.

| Loss or Control | Median Endpoint Move, Pixels Across Seeds | Available Labels | Complete Labels [95% CI] | Full-Grid Lower [95% CI] |
|---|---:|---:|---:|---:|
| Debit missing steps, extra harm weight | 0.018–0.027 | +0.026 | +0.017 [+0.004, +0.029] | -0.018 [-0.029, -0.007] |
| Debit missing steps, symmetric | 3.040–3.695 | +1.334 | +1.281 [+0.094, +2.584] | -2.378 [-3.865, -1.008] |
| Mask missing steps, extra harm weight | 18.242–20.728 | +1.554 | +0.306 [-8.909, +7.795] | -14.067 [-19.691, -9.006] |
| Mask missing steps, symmetric | 30.703–32.575 | +1.920 | +0.593 [-13.566, +13.884] | -27.454 [-36.367, -19.135] |
| Fixed near-neighbor rule | 1.567 | +2.241 | +1.422 [+0.475, +2.357] | -1.021 [-2.424, +0.255] |

Removing extra harm weighting while retaining the missing-step debit
increases the complete-label gain by +1.264 points in the direct
paired comparison [ +0.089, +2.557 ]. That isolates one reason the
original model made almost no adjustment. It does not show the model
works on incomplete tracks, and on the full-grid lower score the
same paired comparison is -2.377 [-3.875, -1.009]. Masking missing
steps lets the model make much larger changes; their apparent gains
on available labels do not survive the full-grid comparison.

One fitting source still has no easy denominator in every seed.
All non-CV arms show large known positive easy harm on at least one
supported source (e.g., worst source 242.9–251.5% for the symmetric
debit arm). This is an unrestricted forecast diagnostic, not a
formal 2% policy safety test.

## One edit per query

On 1,171 multi-agent queries where every arm has a nonidentity edit,
I compared one hindsight-selected edit with one random edit. Both
use the same intervention count and the full-grid lower score.

| Candidate | Hindsight Gain [95% CI] | Random Gain |
|---|---:|---:|
| Original loss | +0.0035 [+0.0016, +0.0066] | -0.0004 |
| Symmetric with debit | +0.3076 [+0.1657, +0.5184] | -0.0639 |
| Masked with extra harm weight | +0.7095 [+0.4310, +1.0234] | -0.3412 |
| Masked symmetric | +1.0334 [+0.6082, +1.4681] | -0.6156 |
| Fixed near-neighbor rule | +0.1117 [+0.0635, +0.1596] | -0.0211 |

There are potentially useful proposals, but the gain requires choosing
well; hindsight is not a learned result. For the next selector test,
the held source must be excluded from **both** the forecast and the
selector training pipeline. Predictions for selector training require
inner source folds that do not train on the outer held source.
[Budgeted diagnostic](capacity_protocol.md)

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_velocity_objectives.py train
.venv-pytorch/bin/python scripts/run_m3w_velocity_objectives.py score
.venv-pytorch/bin/python scripts/run_m3w_velocity_objectives.py replay
.venv-pytorch/bin/python scripts/report_m3w_velocity_objective_capacity.py
```

Use `train --resume` with existing checkpoints. Training used native
arm64 CPU, four compute threads and no loader subprocesses.
Eleven focused tests passed. [Scores](analysis.json) and
[budgeted diagnostics](capacity_diagnostic.json) reproduced exactly.
The forecasts and checkpoints remain private.

This is 8 observed and 12 forecast annotation steps in image coordinates.
No new calibration or confirmation outcomes were opened. The original
2% rule and 15/30 roles are unchanged. No deployment, metric or seconds
claim, Stage5C or SMC. The work does not establish submission readiness.
