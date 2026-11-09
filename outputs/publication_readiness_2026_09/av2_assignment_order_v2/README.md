# AV2: learning within-scene assignment order did not improve release

The [separate-constraints experiment](../av2_assignment_constraints_v1/README.md)
found a ranking failure even after removing its probability cutoff. I tested
whether adding a loss over the whole candidate list would improve decisions.
I also computed the exact number of useful alternatives when the comparator
must qualify on the same view.

## Controlled comparison

All arms keep the same two displacement profiles at each step, the same
ordered recipient assignments, 8 observed and 12 predicted native steps,
CV and OLS4 floors, validity masks, and original 2% conditions.

| Arm | How it ranks |
|---|---|
| Frozen soft cost | Unchanged learned additive costs |
| Pointwise | Predict the continuous assignment score by regression |
| Listwise | Same regression plus cross-entropy over the entire within-scene list |

The donor target puts candidates that satisfy support, risk, and both gain
conditions above those that do not. Bounded utility and risk margins order
candidates within each group. The two new arms share network architecture,
initialization and minibatches, sampling whole scenes equally across donor
cities. Listwise supervision uses temperature 0.25 and weight 0.5. Padded
options are ignored. Prediction uses causal history and frozen baseline
rollouts. No outcome or relative gain label enters the prediction.

Each rotation uses four donor cities, a separate calibration city, and one
held development city. The original nonempty all-qualified prefix is applied
to each arm's scores; there is no minimum probability cutoff. The forecasts
and thresholds were frozen before scoring. Three seeds (17, 29, 43) and six
rotations yielded 36 new fits (400 updates each). Local arm64 CPU training and
freezing took 47.49 s, with 792 MB peak RSS, four Torch threads and zero
loader workers. An initial role check rejected valid metadata before any
training; v2 corrects the check without changing any city role or data.

## Held development results

There are 340 distinct scenarios and 1,020 seed views, 465 with complete
labels. Seed views are not independent.

| Arm | Forced qualifies | Complete-label qualifies | Held releases / passes | Known ADE / available FDE | Worst view p95 ADE | Observed signed easy harm >2% |
|---|---:|---:|---:|---:|---:|---:|
| Frozen soft cost | 89 | 57 | 2 / 1 | 0.25091 / 0.56398 | 7.390 | 171 |
| Pointwise | 53 | 26 | 4 / 0 | 0.26304 / 0.58781 | 7.432 | 401 |
| Listwise | 48 | 27 | 0 / 0 | 0.26134 / 0.58461 | 7.432 | 344 |

All errors use native coordinates. Listwise minus pointwise qualification
is -0.96 percentage points (95% paired city CI [-2.66, +0.43]).
Listwise minus soft qualification is -4.31 pp [-6.87, -1.62].
The direct conservative trajectory contrast against soft is -6.08 pp
[-6.83, -5.40]. Listwise improves known ADE and FDE over pointwise, but
worsens conservative relative-trajectory gain by 0.01311 native units
(CI for listwise minus pointwise [-0.01997, -0.00482]).
Zero held releases does not establish safety or a deployment benefit.

## Exact matched opportunity

This diagnostic uses outcomes after scoring and enumerates the entire
unchanged bank. It never chooses or calibrates a policy. The comparator is
the same frozen learned additive assignment, not another hindsight choice.

| Opportunity on the same fixed bank | Seed views |
|---|---:|
| Some candidate independently qualifies | 317 |
| A qualifying candidate preserves trajectory and improves relative motion on known steps | 248 |
| Same, after the missing-step disagreement debit | 212 |
| Same, and the additive comparator itself qualifies | 47 |
| Same, with complete labels | 38 |

The pointwise and listwise choices realize only 6 and 3 of the 47 strict
matched opportunities. A standalone ceiling of 317 therefore does not
describe the achievable matched claim. Before fitting another ranker,
test whether a different causal reference on donor/calibration data raises
two-arm opportunity at the same edit budget. Keep the frozen soft reference
as a comparator and keep the original risk rule. If the ceiling remains
small, use this branch as evidence about the limits of matched intervention,
not as the manuscript's main performance result.

## Reproduction

Requires the [frozen additive experiment](../av2_assignment_cost_v1/README.md)
and its hash-checked inputs. Run `fit` again to resume: the checkpoint retains
both models, both optimizer states and sampling RNG every 50 updates; the
heartbeat records PID, fold and step. `replay` reads only scored receipts;
the exact-capacity diagnostic reads already scored development outcomes.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_order.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_order.py score
.venv-pytorch/bin/python scripts/run_m3w_av2_assignment_order.py replay
.venv-pytorch/bin/python scripts/audit_m3w_av2_assignment_order_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Fresh scoring](summary.json), [exact matched capacity](matched_capacity.json),
and [freeze receipt](freeze.json). No AV2 official validation/test or
DroneCrowd independent confirmation outcomes opened. No CREATE job was
needed. Manuscript, submission, deployment, Stage5C and SMC unchanged.
