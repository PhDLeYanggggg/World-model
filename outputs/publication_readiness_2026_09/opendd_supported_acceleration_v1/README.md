# OpenDD: Predicting Easy Reference Support Does Not Recover Joint Gains

The Argoverse 2 study found more feasible hindsight improvements with
a decaying acceleration path, but the frozen policy still fell short
of constant velocity. I took that path back to OpenDD and retrained
the gain and harm predictors. I added a separate predictor for
whether an edited actor would contribute an observable, positive
easy-reference error. A tiny nonzero expected denominator on its
own is not good evidence that this reference will exist.

## Experiment

Seven location-excluded fits for each of seeds 17, 29 and 43 gave
21 real CPU fits, each with 600 AdamW steps. Each fit used six donor
sites. The existing neural candidate producer for a donor excluded
both the donor and the scored location; easy cutoffs for training
also excluded both. The only new candidate is decaying acceleration
from the same eight past positions. Twelve-step targets are used only
for training and scoring. All held-location forecasts were frozen
before scoring. OpenDD locations were previously exposed to
development, so this is not independent confirmation.

The refit predicts two missing-step-debited gain increments, six
harm and reference moments, and a binary easy-support event. Each
of four rules compares additive and joint selection on the same
candidate set. Their edit count and exact peak offsets match at
every query. The first rule forces all prior slots. The other rules
can leave slots unused: they require estimated gain and the original
three 2% constraints on a chronological ledger, with both arms
feasible on their own ledger. Two predeclared variants additionally
require an edited actor with evidence score at least 0.25 or 0.50.
These thresholds are diagnostic, not calibrated risk guarantees.
Abstaining returns CV and does not count as passing an undefined
edited-view check. This is a combined refit and menu change relative
to the previous generation, not an isolated estimate of the
acceleration path's effect on OpenDD.

## Held-Location Results

202 recordings, 606 recording-seed views. The table reports the joint
arm; additive uses the same slots in each row.

| Rule | Edits | Edited views | Missing edited-easy reference | Easy degradation >2% | All original risk checks | Qualified positive pair wins |
|---|---:|---:|---:|---:|---:|---:|
| Forced | 3690 | 439 | 144 | 131 | 33 | 0 |
| Moment gate | 267 | 121 | 56 | 3 | 35 | 0 |
| Moment gate + support 0.25 | 222 | 101 | 36 | 2 | 30 | 0 |
| Moment gate + support 0.50 | 40 | 7 | 2 | 1 | 2 | 0 |

The 0.25 evidence gate reduces missing denominators among edited
views from 46% to 36%, but loses five risk passes. The 0.50 variant
releases only at rdb1, where it edits four unique recordings across
the three seeds. The evidence head has mean held-fold AUROC 0.698,
Brier 0.166 and prevalence 22.36%. Of 299 actor-seed predictions at
or above 0.50, only 47.2% actually had the required easy support.
This separates a useful but imperfect evidence signal from a
reliable admission rule.

Forced joint changes 247 query choices relative to additive.
The full-grid pair gain is +0.000198 in native units, with a
3,000-draw paired location interval of [−0.000168, +0.000526]
over six estimable locations. The moment, 0.25 and 0.50 variants
change only one, two and one query choices respectively.
Their matched pair effects are effectively zero, with zero
qualified positive gains. The forced budget itself is concentrated:
3,527 of 3,690 edits are in rdb1 and rdb6, and rdb7 has no slots.

| Joint arm | Known ADE | Available FDE | Mean recording p95 ADE | Conservative CV gain (pp), 95% interval |
|---|---:|---:|---:|---:|
| Forced | 0.54786 | 1.25337 | 1.39092 | −0.0320 [−0.2961, +0.2542] |
| Moment gate | 0.54725 | 1.25329 | 1.38783 | −0.00126 [−0.00288, +0.00034] |
| Support 0.25 | 0.54726 | 1.25330 | 1.38783 | −0.00159 [−0.00324, −0.00020] |
| Support 0.50 | 0.54725 | 1.25329 | 1.38783 | +0.00006 [0, +0.00018] |

The last row is virtually a CV fallback. Its tiny gain is not
positive deployment evidence. The [structured summary](summary.json)
includes the OLS4, hard-subset, additive and tail results.

## Implication

Neither enlarging the candidate menu nor adding a per-actor support
threshold makes the joint selector reliable. The narrowing happens
before the pair score can help: after admission, additive and joint
almost always choose the same paths. A more informative next
diagnostic is to measure how often two distinct gain-positive
assignments are available *with adequate support* at the same
budget, then learn support at the full decision level only where
such alternatives actually exist. Do not take the 0.50 result
as a reason to tune a cutoff on these held views.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_supported_acceleration.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_supported_acceleration.py score
.venv-pytorch/bin/python scripts/report_m3w_opendd_supported_acceleration.py
```

The arm64 run used four Torch threads and no loader workers. The
21 fits plus inference took 56.36 seconds (1.14 GB maximum RSS);
scoring took 3.78 seconds. Training saves optimizer and sampler state
every 50 steps and resumes through the same freeze command.
The [registration](registration.json) binds the donor and
evaluation inputs. Eighteen focused tests passed. No CREATE job
was needed. No independent outcomes were opened; deployment,
Stage5C and SMC remain unchanged.
