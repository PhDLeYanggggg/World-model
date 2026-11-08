# Sparse Support on Pedestrian Development

To separate source concentration from the strong-baseline gap, this
experiment uses the existing Moving A bank: 34 recordings from 14
source groups, evaluated with three forecaster seeds (102 repeated
views), native eight observations and twelve prediction steps.
Forecasts exclude all A sources; the new head fits the original
seven training sources per fold, excluding the six reserved and
one scored source. This is cross-fit development, not a transfer
claim or independent confirmation.

## One Edit, Same Query and Count

The original gain ranking and two learned readouts each replace CV
once per recording at the first legal query. The head predicts the
exact event of a positive conservative gain and a risk pass against
CV, and the stricter event of also beating OLS4. Training samples
source, recording, query and action uniformly in that order. The
dense direct policy and unconditional OLS4 are contextual controls,
not matched interventions. The evaluation still uses the original
unweighted 2% recording-risk numerators/denominators and the original
query-weighted paired utility on available labels with missingness debit.

| Arm | Edits | Risk pass | CV joint | Both joint | Undefined |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense direct | 6,952 | 2 | 2 | 1 | 35 |
| First query, gain rank | 102 | 3 | 2 | 2 | 89 |
| First query, CV head | 102 | 10 | 9 | 6 | 78 |
| First query, dual head | 102 | 8 | 7 | 5 | 76 |
| First query, CV oracle | 87 | 87 | 87 | 27 | 15 |
| First query, dual oracle | 27 | 27 | 27 | 27 | 75 |
| Any query, CV oracle | 99 | 99 | 99 | 30 | 3 |
| Any query, dual oracle | 30 | 30 | 30 | 30 | 72 |

The CV head versus matched gain ranking is +0.01980 pp
[-0.04773, +0.11028] conservative paired utility. Versus OLS4 it
is -9.81617 pp [-21.98417, -0.55340]. These intervals average
the three seeds before 3,000 paired source draws. The oracle may
abstain; its edit counts are not comparable to forced one-edit
arms. The 30 dual opportunities occur in seven source groups,
with nine at the largest. The learner does not reliably improve
on matched ranking or beat the strong comparator.

## Exact Scope of the OLS4 Deficit

OLS4 has lower observed error on 42 views and is certified better
than CV in the paired completion interval on 36. The class of
known-easy, individually CV-improving edits, at most one per
query, never exceeds 30 dual passes even with no recording cap.
Requiring one easy anchor while allowing other known CV-improving
edits gives this exact within-class result:

| Max edits per recording (one per query) | 1 | 2 | 4 | 8 | No recording cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy-only dual pass | 30 | 30 | 30 | 30 | 30 |
| One easy anchor, other known positive gains | 30 | 30 | 30 | 38 | 48 |
| Latter, where OLS4 is certified better than CV | 0 | 0 | 0 | 0 | 0 |

The bridge enumerates the forced easy query and chooses the
largest remaining query gains. Every positive witness is rescored.
This is a label-aware bound within the specified zero-harm action
classes, not a bound for all allowable 2% policies. It shows that
adding more zero-harm edits at this query budget does not repair
the confirmed OLS4 deficits. The next decisive test is whether a
broader forecast bank or a larger per-query menu provides enough
gain while retaining a measured easy reference. Do not spend
independent confirmation on the current head.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_sparse_support.py
.venv-pytorch/bin/python scripts/run_m3w_moving_sparse_support.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_moving_sparse_support.py
.venv-pytorch/bin/python scripts/audit_m3w_moving_support_bridge.py
.venv-pytorch/bin/python scripts/build_m3w_sparse_support_replay.py
```

`fresh_run`: 42 fits, 21,000 updates, PID 82110, 38.1 seconds;
native arm64 Torch, four threads, zero workers, checkpoint and
heartbeat every 100 steps. Both exact audits are fresh and do not
train. `cached_verified`: existing forecast bank, source roles,
dense baseline, and zero-update checkpoint replay. `not_run`:
new CREATE work or independent confirmation. DroneCrowd keeps
15 calibration groups and 30 sealed confirmation groups. No
change to deployment, metric/seconds claim, submission, Stage5C
or SMC.
