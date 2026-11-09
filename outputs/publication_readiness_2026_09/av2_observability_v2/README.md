# AV2: Past Context Does Not Fix Release, but the Menu Has Capacity

I tested whether the short observed tracks around an actor
help decide when a fixed joint forecast should replace CV.
The [paired qualification experiment](../av2_paired_qualification_v1/README.md)
had found a single nonempty calibration threshold that did
not transfer to its held city.

## What Changed

I read only observed steps 42-49 from the existing publisher
train objects, keeping the 8 observed/12 forecast grid.
The scored roster is unchanged. The wider observed roster
supplies local neighbor count, distance, entry and exit
within a past-defined neighborhood, and partial-track counts.
The [AV2 schema](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html)
distinguishes observed states from track-level scoring
categories; no category, inferred object type, stored velocity,
future state, or map enters the model.

The input audit reproduces the scored histories exactly and
counts 219,390 observed states and 4,855 partial tracks across
947 of 1,024 development scenarios. A partial track here lacks
at least one of the eight observed steps; it is not the
publisher's track-fragment scoring category.

The gate predicts the existence of edited-easy label support
(S) and exact additive eligibility (A), either directly or as
a product of an S score and a conditional A score. Per-edited
actor heads also predict future label coverage and easy
membership given an observed label. A is checked to imply S
under the original risk implementation. These are ranking
scores, not a certificate. The motion control masks only the
new context channels; architectures, initialization, optimizer
recipe and sampled batches are matched.

I reused the checked city-excluded donor references and the
outer additive and candidate choices. The fitted gates change
only scene order. The old direct and query-weighted calibration
replay exactly. Six city roles, three seeds, 896 held development
scenarios, and every nonempty threshold under the original
both-arm 2% rule are unchanged. Held forecasts and thresholds
were sealed before scoring.

## Eligibility and Release

| Gate | Eligibility AUROC | Eligibility AP | Eligibility Brier | Calibration passes / 1,020 | Held both pass / released |
|---|---:|---:|---:|---:|---:|
| Fixed direct candidate score | Not an eligibility head | — | — | 0 | No release |
| Fixed query-weighted | 0.716 | 0.153 | 0.05963 | 1 | 0 / 1 |
| Motion, direct | 0.702 | 0.140 | 0.06036 | 1 | 0 / 1 |
| Motion, factored | 0.718 | 0.140 | 0.06081 | 1 | 0 / 1 |
| Context, direct | 0.696 | 0.136 | 0.06013 | 1 | 0 / 1 |
| Context, factored (primary) | 0.702 | 0.138 | 0.06046 | 1 | 0 / 1 |

Eligibility diagnostics pool 1,020 seed views with 63 positive
labels. The paired city-mean Brier change of the primary
relative to the fixed query-weighted head is +0.00028
[−0.00265, +0.00310], where smaller is better. The interval
uses 3,000 paired city bootstrap draws after averaging seeds
within scenario. Every nonempty calibration occurs in the
Pittsburgh-to-Palo Alto fold for seed 29. Each resulting held
readout releases once and fails the original joint risk check.
The primary released trajectory lower gain vs CV is −0.00203 pp
[−0.00609, 0]. There is no deployment change.

Context reduces the label-coverage MAE from 0.1336 to 0.1113,
but an always-full-coverage diagnostic is better (0.0603):
23,003 of 24,480 required edited-actor steps have labels.
Easy-membership AP among actors with labels changes from
0.639 to 0.646, while scene support AP stays near 0.766.
Predicting easy-label support is much easier than deciding
whether both forecasts pass the gain and harm criteria.

## Same Menu, Different Reference: Hindsight Capacity

**Correction:** The [later replay](../av2_margin_assignment_v2/README.md)
found that the old vectorized enumeration dropped views with
a wholly unlabelled actor. The corrected ceilings are 382
eligible-reference views and 344 pair views; fixed-reference
and Palo Alto pair counts are unchanged. The table below
preserves the original readout.

I evaluated the exact same menu and peak budget with outcomes
only to count possible references. The relaxed comparison
requires both paths to meet the original accuracy and risk
criteria and one path to improve the other on full-grid pair
error without conservative trajectory degradation. It uses
no new forecasts and is **not** a deployable comparator.

| Held city | Two-slot views | Fixed reference eligible | Any eligible reference | Pair with fixed reference | Pair with any eligible reference |
|---|---:|---:|---:|---:|---:|
| Austin | 252 | 24 | 115 | 18 | 101 |
| Dearborn | 99 | 7 | 21 | 6 | 21 |
| Miami | 309 | 13 | 119 | 9 | 107 |
| Palo Alto | 54 | 0 | 18 | 0 | 16 |
| Pittsburgh | 186 | 11 | 69 | 10 | 62 |
| Washington, D.C. | 120 | 8 | 27 | 5 | 27 |
| Total | 1,020 | 63 | 369 | 48 | 334 |

The 48 and 334 pair views represent 35 and 121 distinct
scenarios, respectively. Fixed edited-easy support is defined
in 382 views; an option with that support exists in 745.
In Palo Alto, no scene gate can make the existing reference
qualify on these views, even though other lawful options
exist. This points to improving reference assignment before
retraining another gate. I would test a donor-trained
support-aware independent-action assignment, keeping the old
reference as a control, and re-estimate matched joint utility
only after its calibration passes. No hindsight option is
used to change any forecast or threshold.

## Reproduce

Requires the [paired qualification assets](../av2_paired_qualification_v1/README.md).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_observability.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_observability.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_observability_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

The initial input preflight stopped on a shared-scale broadcast
bug, before training; a multi-actor regression test covers
the correction. The v2 run completes 36 gate fits (two per
fold) with native arm64 PyTorch, four threads and zero loader
workers. Fit/freeze: 183.39 s and 1.24 GB peak RSS; score:
5.54 s; capacity audit: 16.37 s. Atomic checkpoints every
40 steps retain optimizer and sampler state. Receipt replay
took 9.98 s without new fits. 48 focused tests passed.
No CREATE job, manuscript change, independent confirmation
access, Stage5C, or SMC.

[Registration](registration.json) · [Freeze](freeze.json) ·
[Results](summary.json) · [Reference capacity](reference_capacity.json)
