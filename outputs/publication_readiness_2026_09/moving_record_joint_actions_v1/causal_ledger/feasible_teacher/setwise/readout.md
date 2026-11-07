# Whole-Set Supervision And The Limits Of The Causal Slate

I regenerated the same 32,297 rejected/repaired donor pairs from the
double-excluded teacher. Two new models fit the difference between
complete sets: one uses additive action features; the other adds mixed
relative motion of the two substituted forecasts. The membership
student is the previous fitted control. All three choose from the
same slate. Every option has the original count and no more predicted
all or easy excess than the original set in that query. This is a
prediction constraint, not an observed safety certificate.

| Policy | Switches | Completion Lower vs CV (pp) | Easy Risk Above 2% | Signed Easy Net Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|
| Original ledger | 6,184 | +0.10154 | 65 | 2 | 34 | 3 |
| Membership ranker on this slate | 6,184 | +0.09783 | 64 | 3 | 35 | 3 |
| Additive set difference | 6,184 | +0.06563 | 64 | 0 | 36 | 2 |
| Set difference with interaction | 6,184 | +0.06523 | 64 | 0 | 36 | 2 |

Of 3,323 active query views, 2,721 admit more than one option and
2,580 admit more than one two-switch option. The interaction model
changes 264 queries and 591 row actions relative to the additive
model. Its primary paired completion lower is -0.00062 pp
[-0.00215, +0.00130]. The additive model also loses against the same
slate membership control: -0.03978 pp [-0.06602, -0.01669].
The observed local relative-position error reduction from interaction
is +0.000194 image units per supported query
[-0.000830, +0.001447]. No measured interaction improvement or
original safety improvement follows. Intervals use 3,000 paired draws
of 14 sources, after averaging the three forecast banks.

## Can A Better Ranker Pass On This Slate?

The posthoc observed-outcome check re-enumerates the exact 198,964
options. It proves **75/102 views cannot satisfy the original risk
rules on this slate**, even if the ranker knows the outcomes. In 34
views no available option supplies positive easy-reference mass.
Summing each query's minimum observed all-risk excess rules out 28
views; the analogous easy bound rules out 39. These counts overlap,
and include ten views without an active switch. For each budget
separately, a sum of minima above zero makes that budget infeasible
under any selection of one option per query. The other 27 views are
**not ruled out**, not proved feasible. This does not find an optimum
or export any outcome-aware action.

**Decision.** The two learned set scores fail. More importantly,
improving the scorer alone cannot repair the screened slate. Next
compare wider candidate support while preserving the same original
observed rules, instead of fitting a larger interaction model on
the same options. The strict predicted-spend restriction is specific
to this experiment, not a scientific requirement of the 2% gate.

This is the exposed Moving A cohort: 34 recordings over three actual
forecast seeds, 8 observed and 12 future annotation steps, image
coordinates. Not independent confirmation, physical safety, or a
deployment. Original risk/support rules, Stage5C and SMC are unchanged.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_setwise_repair.py fit
.venv-pytorch/bin/python scripts/run_m3w_setwise_repair.py score
.venv-pytorch/bin/python scripts/run_m3w_setwise_repair.py replay
.venv-pytorch/bin/python scripts/audit_m3w_setwise_slate_support.py
```

Registration `0a72ad3f` precedes fitting. Eighty-four new source-excluded
fits complete in 43.42 s on native arm64, with per-source checkpoints.
The original control and score replay are exact. Fifty scoped tests pass.
[Protocol](protocol.md) | [Replay receipt](replay_execution.json) |
[Observed support bound](slate_support.json)
