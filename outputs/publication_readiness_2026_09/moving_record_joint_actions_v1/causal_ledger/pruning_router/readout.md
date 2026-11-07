# Can A Causal Prefix Identify Repairable Recordings?

The earlier deletion teacher touched few repairable recordings and
lost one existing pass by removing selected easy reference. This
experiment keeps its imitation action head, the exact original
direct stream, and the original 2% and reference-support rules.
The task is exposed Moving A, eight observed and twelve forecast
annotation steps in native image coordinates.

## Detecting Repairable Failures

The prefix score estimates whether direct will fail yet has a
non-loss deletion solution. Training choices are regenerated
on each fitting source using the other six fitting sources,
and calibration and held sources are excluded. Only current
and past proposed direct actions are input. Checked optima
provide positive labels; an unresolved solve is not a negative.
The 42 fitted heads use 175 positive and 434 negative repeated
fitting-record occurrences, not independent samples.

At the first active query, 28 of 84 active views have a repairable
failure under the previously verified deletion-only optimum.
The fitted score has **AUROC 0.632 [0.440, 0.766]** and **AP 0.442
[0.209, 0.615]**, versus positive prevalence 0.333. The intervals
resample 14 source groups 3,000 times, retaining their repeated
seed views together. The estimate is suggestive, not reliable
separation: the AUROC interval includes chance. The 18 zero-action
views have no routing decision and cannot be repaired by deletion.

## Does Routing Improve The Policy?

The routed arm uses the same imitation score; the reserve arm
also retains at least half the original *predicted* easy-reference
mass seen so far. This does not certify observed support. Both
calibrate on the original six donors before held scoring.

| Calibrated policy | Deletions | Changed views | Repairable failures touched / 28 | Complete passes / 102 | Signed easy >2% |
|---|---:|---:|---:|---:|---:|
| Direct | 0 | 0 | 0 | 2 | 8 |
| Imitation | 75 | 3 | 1 | 2 | 8 |
| Predicted-gain guard | 75 | 9 | 4 | 1 | 7 |
| Prefix routing | 44 | 5 | 2 | 2 | 8 |
| Routing and support reserve | 1 | 1 | 0 | 2 | 8 |

Routing calibrates in 6/42 cells; routing with a reserve in 4/42.
Neither adds a pass or loses selected easy support. Both have 35
undefined easy upper bounds, unchanged from direct. Paired
completion-lower effects versus direct are -0.00025 pp
[-0.00074, 0] and +0.00001 pp [0, +0.00002], respectively.
These intervals resample sources after averaging the three
forecaster seeds within each source. Neither policy improves
the registered gate, and their different coverage is not an
equal-count ranking comparison.

Holding the original routing and action cutoffs fixed, turning
on the reserve changes 44 deletions to 43, with the same two
passes and eight signed easy violations. Thus the collapse to
one deletion happens with recalibration, not because the reserve
directly vetoes 43 actions at the old thresholds. The reserve
diagnostic does not tune thresholds or change frozen decisions.

Current prefix descriptors do not identify repairable records
reliably enough to help the policy. This result does not establish
that more data or a better representation cannot help. The next
question is whether a broader but still causal action set can
acquire reference support without giving up paired utility, not
whether a different deletion threshold can mask the limitation.

## Reproduction

Registration `a0db8b02` precedes the 42 new routing fits. The
forecaster bank, imitation head and prior capacity receipts are
verified parents. With those private artifacts available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_pruning_router.py fit
.venv-pytorch/bin/python scripts/run_m3w_pruning_router.py score
.venv-pytorch/bin/python scripts/run_m3w_pruning_router.py replay
.venv-pytorch/bin/python scripts/audit_m3w_pruning_router_reserve.py
```

Fit resumes from source checkpoints. Score replay reproduces the
aggregate bytes. No independent outcomes were opened, and no
deployment, metric/seconds, physical-safety, Stage5C or SMC claim
follows.
