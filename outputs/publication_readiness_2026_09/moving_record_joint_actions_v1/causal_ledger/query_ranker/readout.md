# Learning Relative Action Scores Within A Query

I tested whether the model can choose better agents by learning
differences between legal interventions in the same frame. The new
models use the same forecasts, descriptors, source weights, ridge and
gain-minus-harm target. The pointwise model learns absolute scores. The
centered model learns within-query differences, removing shared offsets.
Both use the original risk estimates and feasibility checks when making
their choices. Neither inherits other sources' learned selections as
training labels. This is a new ranking experiment, not a forecast refit.

| Ranking Objective | Switches | Lower Utility vs CV (pp) | Hard Known Gain vs CV (%) | Easy Risk Upper Above 2% | Easy Net Degradation Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original ledger | 6,184 | +0.10154 | +0.05464 | 65 | 2 | 34 | 3 |
| Pointwise | 6,184 | +0.05826 | +0.02022 | 66 | 0 | 34 | 2 |
| Within-query centered | 6,184 | +0.04214 | +0.01487 | 66 | 0 | 33 | 2 |

The number of switches matches at **every** current query. The centered
model replaces 2,877 of 6,184 selected agent occurrences relative to the
pointwise model, and changes the forecast for 291 shared occurrences.
The primary paired completion-lower contrast is -0.02030 pp
[ -0.04727, +0.00018 ]; the known-label contrast is -0.01741 pp
[ -0.04062, +0.00107 ]. Neither establishes improved ranking.
Both new rankings lose conservative utility against the original ledger:
-0.05586 pp [ -0.08378, -0.02987 ] for pointwise and -0.07108 pp
[ -0.11468, -0.03259 ] for centered. Intervals use 3,000 paired draws of
the 14 sources, after averaging the three actual forecaster banks.
The 102 views represent 34 recordings over three banks, not independent
recordings. Hard gains are a known-label source-macro diagnostic.

**The repair does not pass.** Neither new policy exceeds 2% signed easy
net degradation in any view. But both have 66 views exceeding the easy
positive-harm completion upper limit, and only two pass the full original
rules. Lower signed harm is not the same as controlling positive harm.
The centered-minus-pointwise reduction of observed local-pair relative
position error is +0.00167 image-coordinate units per supported query
[ -0.00476, +0.00762 ]. There is no established interaction benefit and
the allocator does not optimize physical collisions.

The common scalar objective and the centering comparison fail for this
bank. The experiment does not prove that all within-query learning fails,
or that source offsets uniquely explain the result. It motivates training
on joint feasible choices from donor outcomes rather than assigning
arbitrary equal weight to risk and gain in a scalar ranker. Any resulting
policy still needs the original observed rules and independent evaluation.
No independent outcomes were opened and deployment is unchanged. The
task remains eight observed to twelve forecast annotation steps in image
coordinates, not physical time or scale. Stage5C and SMC are off.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_query_centered_ranker.py fit
.venv-pytorch/bin/python scripts/run_m3w_query_centered_ranker.py score
.venv-pytorch/bin/python scripts/run_m3w_query_centered_ranker.py replay
```

Design `9688de69` precedes fitting. Eighty-four new source-excluded
fits and frozen allocations run in 10.22 s on native arm64 with seed
checkpoints. Scoring and replay take 10.94 and 11.06 s. Static control
metrics and final analysis SHA-256 are exact. Forty-five scoped tests
pass, including explicit pairwise verification of the centered loss.
[Protocol](protocol.md) | [Exact score replay receipt](replay_execution.json)
