# Forecast Geometry At Fixed Intervention Recipients

I kept the four original whole-query choices, then supplied two alternate
forecast shapes for their exact same recipients. This tests whether the
option bank limits performance even without changing whom to intervene on.

| Hindsight options | Risk-feasible views | Infeasible views | Positive conservative gain vs. OLS4 |
| --- | ---: | ---: | ---: |
| Four frozen choices | 61 | 41 | 27 |
| Plus same-recipient OLS4 | 70 | 32 | 32 |
| Plus same-recipient common translation | 68 | 34 | 32 |
| Both additions | 71 | 31 | 33 |

The combined slate restores feasibility in ten views and adds six
positive OLS4 witnesses across three source groups. Where the original
slate was feasible, the median increase in optimum conservative utility
is 0.236 percentage points on the same CV denominator. No solve is
unresolved; every feasible witness passes the original recording scorer.
Both alternatives help, but neither alone explains all the gains.

This is a capacity diagnostic on **34 exposed recordings, 14 source
groups and three forecast seeds**, not 102 independent recordings. The
solver uses development labels; neither the individual choices nor the
recording optima are attainable results for a learned policy. The common
translation preserves relative paths only between replaced agents in the
same query. Forecasting is eight observed and twelve predicted annotation
steps in image pixels, with no verified seconds or metric scale.

The geometry change is useful but insufficient: 31 views are still
infeasible, and 38 more are feasible without a positive conservative
gain over OLS4. The next experiment should design past-only proposals
that improve over OLS4, not assume that another confidence threshold can
make the current bank work. DroneCrowd calibration is unchanged and no
confirmation data were opened. No deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Aggregate result](development.json) |
[Execution receipt](development_execution.json) |
[Working supplement](../evidence_manuscript_v10/slate_geometry_appendix.tex)

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_slate_geometry_capacity.py tests/test_m3w_joint_slate_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_slate_geometry_capacity.py development
```

The first readout uses the existing frozen development banks and freshly
solves the recording programs. Re-running with the same local private
checkpoints re-verifies the aggregate without redoing the solves.
