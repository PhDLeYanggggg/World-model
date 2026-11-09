# OpenDD vehicles: learning feasibility helps, but joint scoring is not better

The [fixed-menu study](../opendd_vehicle_pair_support_v1/README.md)
found a qualifying two-edit pair in hindsight in 8,062 of 8,874
recording views, but the causal rule found only 1,979. I kept the
two edits, source roles, query, paths and radii fixed and trained
the selector directly on examples of feasible and infeasible pairs.

## What changed

Both trained heads predict the original risk pass, the risk-and-dual
pass, and conservative incremental gains against CV and OLS4.
Each donor's candidates come from a producer that leaves out the
donor, calibration and held locations. The same past-only per-action inputs,
labels and 200-update budget are used for the learned controls.
The additive head combines separate actor scores; the joint head
combines both embeddings and relative motion features. The latter
also has more parameters (6,180 versus 4,964); this comparison does
not isolate the geometry features from the architecture.
The same-family arm reuses the joint head and restricts the search.
All choices and calibration thresholds were frozen at `612b1df99`.

## Fixed-count comparison

| Outcome | Archived joint | Learned additive | Learned same family | Learned joint |
| --- | ---: | ---: | ---: | ---: |
| Recording views | 8,874 | 8,874 | 8,874 | 8,874 |
| Missing easy denominator | 5,624 | 3,730 | 4,210 | 4,247 |
| Pass all risk checks | 2,371 | 4,379 | 3,835 | 3,743 |
| Also beat CV and OLS4 | 1,979 | 2,975 | 3,012 | 2,952 |

The learned joint rule finds 973 more qualifying pairs than the archived
rule (+7.72 per direction averaged across sites, 95% site interval
+3.42 to +13.25). This does not establish a joint advantage. The learned
additive arm finds 996 more than archived, and the learned joint-minus-
additive count contrast is -0.18 per direction [-1.71, +1.38].

| Direct paired learned joint contrast | Gain (95% site interval) |
| --- | ---: |
| Conservative trajectory vs learned additive (pp) | -0.00533 [-0.00916, -0.00241] |
| Available-step trajectory vs learned additive (pp) | +0.00025 [-0.00045, +0.00096] |
| Conservative relative motion vs learned additive (native units) | -0.04435 [-0.06402, -0.02759] |
| Conservative trajectory vs same family (pp) | -0.00100 [-0.00157, -0.00047] |
| Conservative trajectory vs archived joint (pp) | -0.01174 [-0.01989, -0.00542] |

Trajectory gains are divided by the common CV error mass. Conservative
contrasts debit the actual difference between the two predictions on
missing steps. This is not proof of the sign on unobserved futures.
The 8,874 views repeat 493 recordings, three seeds and six calibration
directions. Intervals resample the seven site means 3,000 times.

## Frozen release on held sites

| Released outcome | Archived joint | Learned additive | Learned same family | Learned joint |
| --- | ---: | ---: | ---: | ---: |
| Released recordings, two edits each | 61 | 65 | 65 | 66 |
| Missing easy denominator | 43 | 23 | 22 | 20 |
| Pass all risk checks | 17 | 32 | 30 | 37 |
| Also beat CV and OLS4 | 13 | 19 | 24 | 29 |
| Active directions with every release passing both | 1 / 16 | 0 / 16 | 4 / 18 | 4 / 16 |

Twenty of the learned joint rule's 29 passes come from rdb1.
After calibration, the learned joint rule has more passing releases than the archived heuristic,
but neither new head meets the full gate. These releases are not
count-matched. CV remains the fallback.

**Next comparison:** test whether actual observed vehicle spacing can
improve the same learned joint model, with identical parameters, targets
and budgets in the no-spacing control. The
[observed geometry export](../opendd_vehicle_observed_geometry_v1/README.md)
recovers the missing spacing on all existing inputs; it was not used
here. This experiment learns pair outcome surrogates and enumerates the
decision afterwards. It does not train through the solver as in
[Wilder et al.](https://ojs.aaai.org/index.php/AAAI/article/view/3982).

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_learning.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_pair_learning.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_pair_learning.py tests/test_m3w_opendd_vehicle_observed_geometry.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_vehicle_pair_hindsight.py tests/test_m3w_vehicle_support_rank.py tests/test_m3w_vehicle_easy_factor.py tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: 126 fits, 25,200 updates, frozen choices and held scoring.
The 315 donor batches contain 4,731,290 sampled pair instances, with
repeated source rows across fits. Native arm64 PyTorch, four compute
threads, no data workers, checkpoints every 100 steps, per-cell receipts
and PID heartbeats. Twenty-three focused tests passed. `cached_verified`:
the forecast bank, old cohort and archived heuristic. No CREATE job,
external confirmation, manuscript or deployment change.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Per-site readout](readout.json).
