# Candidate Coverage Does Not Ensure Learnable Allocation

I kept the candidate bank, all 125,540 rows and the original recording risk
rules fixed. For each of three actual producer seeds, I fit cost moments on
13 A sources and applied them to the omitted source. The producers had never
fitted any A source. These are 42 deterministic allocator fits on an already
exposed development set, not 42 independent experiments or new neural fits.

| Policy | Switches across seeds | Original risk passes / 102 | Undefined easy risk |
|---|---:|---:|---:|
| CV only | 0 | 0 | 102 |
| Independent | 9,675 | 4 | 15 |
| Common action per query | 9,675 | 2 | 17 |
| Pair marginals only | 9,675 | 3 | 15 |
| Joint with mixed term | 9,675 | 3 | 16 |
| Joint then budget prefix | 12 | 3 | 99 |

Counts match exactly within each query for the four middle rows, not just
in aggregate. The joint policy reduces observed relative-pair displacement
error by 0.231 pixels [0.139, 0.345] versus independent selection. But against
the otherwise identical marginal policy the reduction is only 0.013 pixels
[-0.009, 0.045], and direct paired completion-lower ADE utility is **-0.061
pp [-0.157, -0.006]**. The joint term changes 418 of 4,839 seed-query views.
The relative-pair comparison is on common observed pair steps, not a
collision or physical safety claim. The marginal component accounts for
most of the observed gain; the mixed term does not earn its place here.

Versus CV, joint known-label gain is +0.782 pp [0.158, 1.381], but its
completion lower is -0.340 pp [-1.871, 0.770]. Estimated recording budgets
reduce this to 12 switches and no positive utility (-0.00084 pp, CI crosses
zero). Those 12 decisions use only constant position, across nine repeated
views; they are not an example of a neural intervention. Only three views
meet all original conditions; non-switches do not count as risk passes.
Intervals resample 14 sources 3,000 times after averaging the seeds.

I replayed the actions from frozen coefficients without labels. In **83 of
93** zero-switch views there is a positive-utility action somewhere in the
full bank that individually meets both predicted budgets. In 43, all
positive joint proposals violate the estimated easy budget. Only two
zero-switch views have no nonpositive easy-budget excess among any distinct
action. This identifies a concrete search restriction in addition to any
cost calibration problem. [Filter-first follow-up](filter_first/protocol.md).

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_record_joint_actions.py fit
.venv-pytorch/bin/python scripts/run_m3w_record_joint_actions.py score
.venv-pytorch/bin/python scripts/run_m3w_record_joint_actions.py replay
.venv-pytorch/bin/python scripts/diagnose_m3w_record_joint_actions.py
```

The initial run stopped before fitting when it looked for the original job
list in the wrong manifest. The v2 registration (`2eadff11`) resolves the
hash-bound parent; methods, budgets and outcomes do not change. The three
seed caches resume atomically. Fit, score and replay all exit 0. The original
record rules, 8/12 image-coordinate task and external reservations remain
unchanged. No deployment, independent confirmation, Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) | [Budget diagnosis](budget_diagnosis.json)
