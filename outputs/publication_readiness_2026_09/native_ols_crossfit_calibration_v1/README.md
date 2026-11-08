# A positive OLS4 replacement score was not a reliable activation rule

The previous support-aware model intervened at nearly every eligible query. I ran leave-one-source-out calibration inside the original nine fitting sources: 54 fresh Torch fits across three seeds, two objectives and nine held groups. This produced out-of-fold scores for 69,746 agent windows in 1,176 queries. Each pair uses identical initialization and draws from the other eight sources. The constant-probability control uses the same conditional heads and the mean training-source missing rate. The seven thresholds and decision rule were fixed before fitting. None qualified.

| Policy / score threshold | Interventions | Conservative gain vs. OLS4 (pp) | 95% source-bootstrap interval | Positive source means |
| --- | ---: | ---: | ---: | ---: |
| Direct / 0 | 6,583 | +0.0610 | [-0.0638, +0.1798] | 8/9 |
| Direct / 0.04 | 1,988 | +0.0528 | [+0.0051, +0.1237] | 6/9 |
| Neighborhood factorization / 0 | 6,904 | +0.0106 | [-0.1174, +0.1339] | 6/9 |
| Neighborhood factorization / 0.04 | 2,136 | +0.0513 | [-0.0020, +0.1339] | 4/9 |
| Fixed probability / 0 | 6,879 | -0.1050 | [-0.2960, +0.0451] | 3/9 |
| Fixed probability / 0.04 | 1,516 | +0.0413 | [-0.0050, +0.1073] | 6/9 |

The registered internal rule needs nonzero intervention, a positive interval lower endpoint and positive means in at least seven sources. The direct arm at 0.04 passes the interval screen but not the source requirement. These source intervals describe the fixed ladder; they are not formal inference after threshold selection. The common denominator is observed CV ADE mass, not OLS4 ADE. Counts are repeated agent windows across seeds, not independent people or recordings.

The score itself is optimistic. Among nominated neighborhood-factorized rows scoring between 0.02 and 0.04, the weighted mean predicted gain-minus-harm is +0.0296 but the realized mean is -0.0078 (2,525 nominations). Above 0.08 the predicted mean is +0.1033 versus +0.0211 realized (493 nominations). The 0.04 threshold removes many optimistic nominations, but the remaining gain is too concentrated to promote this policy. This is evidence against a reliable global margin with the current features, not against every way of selecting replacements.

The application receipt records the abstention decision and no access to development targets. The fitting-source selection and fallback branch replay exactly; they do not re-train the 54 models during replay. Nine related tests passed. No change to the original CV risk rule, DroneCrowd result, confirmation access, deployment, Stage5C or SMC. Eight observations and twelve predictions are annotation steps; no seconds or metric claim.

```sh
runner=scripts/run_m3w_ols_crossfit_calibration.py
.venv-pytorch/bin/python "$runner" oof
.venv-pytorch/bin/python "$runner" select
.venv-pytorch/bin/python "$runner" apply
.venv-pytorch/bin/python "$runner" replay
```

Use `oof --resume` to resume atomic fold checkpoints. [Protocol](protocol.md) | [Full ladder and calibration bins](analysis.json) | [Application](application.json) | [Replay](replay_execution.json)
