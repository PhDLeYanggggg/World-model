# Exact Capacity Against Two Strong Motion References

The [prior audit](../native_exact_count_joint_capacity_v1/README.md)
maximized conservative gain against constant velocity (CV) and checked
its witnesses against the four-position least-squares velocity baseline
(OLS4). It did not establish whether a different assignment could beat
both. This experiment directly tests that question on the exposed
development data at the same exact count in every query.

## Results

All 102 views (34 recordings repeated across three frozen forecaster
seeds) were resolved at the same 9,675 genuine switches. Counts are
not independent recordings or sites.

| Forecast options | Original risk feasible | Original CV optimum beats both | Maximin witness beats both | Negative maximin bound | Risk/count infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One frozen path per person | 93 | 44 | 44 | 49 | 9 | 0 |
| Full bank | 96 | 51 | 51 | 45 | 6 | 0 |

All negative bounds came from solver status 0 (optimal). Optimizing
against both references did not find any additional views beyond the
CV-maximizing witnesses. For 45 risk-feasible views, no assignment in
the full bank can beat both references under the registered conservative
criterion and exact query counts. A better ranker alone cannot remove
that limitation of the fixed menu and budget. It is not proof that
all missing-label completions lose, or that a different forecast
bank or intervention budget would fail. For the 51 remaining views,
existence does not establish that a past-only policy can recover
the witness.

See [scalar analysis](analysis.json), [score receipt](score_execution.json)
and [replay receipt](replay_execution.json). Six focused tests passed.

## Formulation

For each eligible row, start with the CV path. The coefficient for a
non-CV action is the change from that fallback's conservative gain
against each reference. On an observed row the change is the reduction
in available-step ADE from CV to that action; it is the same for both
references. On a wholly unobserved row the change against a particular
reference is the fallback's maximum path disagreement with that
reference minus the action's maximum disagreement. Thus the OLS4
contrast has a nonzero constant even if nothing switches. All gains
share the same observed CV denominator and original within-query
weights. Neither the denominator nor the partial rows are imputed.

Choose at most one nonidentity action per person and exactly the frozen
number per query. Keep the original CV-based positive all/easy harm,
signed easy degradation and easy support constraints. Maximize a
continuous margin that is at most each of the two gain masses.
On an observed row, one lowest-ADE
action dominates other nonidentity alternatives. For unknown rows, the
solver keeps every legal action because their CV and OLS4 disagreements
can order them differently.

A witness is accepted only after verifying the exact genuine switch
counts, original recording rule and both gain signs with the original
scorer. Infeasibility and negative objective upper bounds are separate
outcomes. An unresolved view is neither a pass nor a proved failure of
the bank. This optimizes hindsight utility, not a past-only policy.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_dual_reference_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_dual_reference_capacity.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dual_reference_capacity.py \
  tests/test_m3w_exact_count_joint_capacity.py tests/test_m3w_recording_risk_attribution.py
```

The [protocol](protocol.md) and [registration](registration.json)
precede real-data scoring. Scoring resumes by recording and seed;
replay re-scores cached witnesses and reproduces the aggregate but does
not independently rerun infeasibility solves. Private witnesses, data
and forecast arrays are not published. The unit is native image pixels
at 8 observed and 12 forecast annotation steps. No confirmation access,
deployment, Stage5C execution or SMC.
