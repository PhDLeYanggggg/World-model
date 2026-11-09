# Broader Coverage Needs Better Risk and Gain Estimates

I expanded the frozen policy from sparse edits to all legal proposals
within the current query. An action with predicted easy reference
support starts the pool; other actions may use its predicted all/easy
risk slack. The controller uses no later query. I compared it with the
same library ranked by gain, with and without matching the pooled
controller's intervention counts. Predictions were frozen before scoring.

| Held diagnostic | Edits | Original risk passes /72 | Undefined easy /72 | Worst known signed easy change |
|---|---:|---:|---:|---:|
| Prior sparse policy | 2,317 | 14 | 38 | +0.84% |
| Relaxed independent screen, no cap | 12,674 | 10 | 30 | +111.79% |
| Strict independent screen, no cap | 197 | 5 | 66 | +0.014% |
| Pooled risk, cap eight | 229 | 5 | 66 | +0.014% |
| Pooled risk, cap 32 or no cap | 244 | 5 | 66 | +0.014% |
| Gain only, matched to the uncapped pool in each query | 244 | 0 | 66 | +0.26% |
| Gain only, no cap | 190,226 | 0 | 6 | +880.96% |
| Calibration-selected fallback | 0 | 0 | 72 | Undefined edited support |

Pooling admits 98 edits that would fail a strict per-action budget or
the per-action easy probability screen. That did not deliver broader
coverage. The paired lower utility versus count-matched gain is
-0.0040 pp [-0.0120, 0.0000]; the paired lower neighbor contrast is
-0.0016 pixels [-0.0049, 0.0000]. The independent relaxed arm increases
coverage but trails OLS4 by -55.439 pp on the conservative CV scale.
The 18-setting calibration qualifies **0/9** cells. No setting passes
more than one calibration recording in a cell. The deployable decision
remains fallback.

## Which prediction is blocking coverage?

Of 3,528 repeated query views, 2,453 have an actually eligible easy
seed in the library, but the predicted screen fires in only 73 views.
All 73 have a true seed somewhere in the query; 72 have a true seed
that also satisfies the predicted criteria on that **same action**.
On the 2,453 true-seed queries, the probability condition by itself
retains 2,195, the predicted all-excess sign retains 572, and the
predicted easy-excess sign retains only 72. Thus the dominant loss is
the predicted sign of the easy excess, not the probability cutoff.

I then replaced predictions with realized terms in the same greedy
algorithm, solely for outcome-aware attribution:

| Information given to the pool | Edits | Original passes /72 | Conservative gain vs. CV (95% source interval) |
|---|---:|---:|---:|
| Predicted risk and gain | 244 | 5 | +0.0003 [0.0000, +0.0010] pp |
| Realized risk and support, predicted gain | 148,073 | 66 | -10.187 [-31.204, +1.451] pp |
| Realized risk, support, and gain | 146,809 | 66 | +16.491 [+5.835, +28.403] pp |

Neither outcome-aware greedy arm establishes a lower gain over OLS4;
even the last row is -29.532 pp [-80.859, +12.567]. Correcting the
support screen alone can recover interventions without recovering
accuracy. Intervals use 3,000 source-group draws after averaging seeds.

## What can the full bank achieve in hindsight?

Allowing any outcome-aware recording-level selection on the same
library initially yields 19/72 verified dual-reference witnesses
with cap eight, 21/72 with cap 32, and 21/72 with no cap. The first
two counts still have 8 and 24 unresolved views. I continued all
32 unresolved uncapped cells using a separable Lagrangian upper
bound. Every dual trial combines the two gains with convex weights,
prices the original risk and support inequalities, and relaxes the
recording to independent agent choices between one path or CV.
Each dual feasible value is an upper bound. The resulting primal
proposals are scored independently, without assuming dual tightness.

The continuation finds **32 additional verified witnesses**.
Final uncapped accounting: **53/72 verified**, 13 ruled out by the
upper bound, six undefined for lack of easy support, zero unresolved.
This is not a 21-to-53 gain due solely to removing the cap: the
uncapped solver also received a continuation. It shows that useful,
in-budget combinations exist for much of the roster, but the present
scores cannot identify them before outcomes. A direct signed
easy-excess loss and a conservative pairwise gain target are the
next matched training comparisons. A calibration sweep alone will
not close the measured gap.

This empirical recording rule is not
[conformal risk control](https://arxiv.org/abs/2208.02814): a bound on
expected monotone loss would not by itself establish the original
per-recording requirement with potentially undefined support.

## Reproduction

`fresh_run`: 24 fitting recordings in nine sources, three frozen
seeds, 50.30 s for selection and scoring on native arm64 CPU, plus
outcome attribution and 216 original solver cells. No optimizer
updates. `cached_verified`: heads, donor paths, sparse controls,
exact score replay, and independent rescoring of all positive
witnesses. Thirty-six focused tests pass; the legacy suite was
not rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_pooled_coverage.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_support.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_support_funnel.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_pooled_coverage_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_uncapped_dual_certificate.py
.venv-pytorch/bin/python scripts/build_m3w_cv_pooled_coverage_replay.py
```

The local anonymous scalar replay archive SHA-256 is
`6c8be40650c0127e5fc0cc7bc35bce9df6dd5bc1d272ae03dc99bd6e45fa9433`.
It reproduces the scalar risk and utility tables after clean extraction
without original tracks or Torch; it does not reproduce geometry,
training or the solver.

`not_run`: new A or DroneCrowd calibration outcome access, independent
confirmation, deployment, Stage5C and SMC. The 30 confirmation groups
remain sealed. The 2% rule is unchanged. Native 8/12 annotation steps
and local image pixels are not verified seconds or physical units.
This is not a 3D or foundation model.

[Protocol](protocol.md) | [Readout](readout.json) |
[Attribution](support_attribution.json) | [Support funnel](support_funnel.json) |
[Capacity](hindsight_capacity.json) | [Continuation](uncapped_continuation.json) |
[Manuscript v130](../evidence_manuscript_v130/README.md)
