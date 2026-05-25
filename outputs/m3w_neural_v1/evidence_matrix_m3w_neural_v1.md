# M3W-Neural v1 Evidence Matrix

- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.
- package_input_hash: `44071931ceb325e067ce5ce68c0bced79e25983915bbcd20aed45668856ed4b8`
- git_commit: `52ba61a`

| Evidence | Value | Gate interpretation |
| --- | --- | --- |
| Stage41 gates | `41 / 41` | pass if all gates true |
| endpoint geometry pass | `True` | required |
| all improvement vs Stage37 floor | `21.03%` | must be positive |
| t+50 improvement vs Stage37 floor | `13.65%` | must be positive |
| t+100 raw-frame diagnostic | `14.69%` | diagnostic only |
| hard/failure improvement | `20.38%` | must improve |
| easy degradation | `0.00%` | must be <= 2% |
| switch rate | `34.10%` | reported for deployment risk |
| positive external domains | `3` | must be >= 2 for cross-domain evidence |
| bootstrap evidence pass | `True` | required for statistical support |
| multiseed replication pass | `True` | required for replication support |
| strict delta vs teacher repair pass | `True` | required for latest-policy contribution |
| all-agent composite world-state pass | `True` | required for full active-agent waypoint evidence |
| all-agent composite ADE all/t50/t100 | `21.03% / 13.65% / 14.69%` | protected full-waypoint rollout |
| all-agent composite FDE all/t50 | `19.82% / 17.39%` | endpoint check over same full rollout |
| all-agent composite multi-agent ADE all/t50 | `20.82% / 13.80%` | same-frame multi-agent rows |
| pure UCY source-heldout gate | `True` | required for UCY held-out support |
| pure UCY-only retrain/select/test gate | `False` | reported blocker, not claimed |
| strict pure UCY neural retrain gate | `True` | source-only neural retrain with validation-selected conservative residual policy |
| strict pure UCY neural best trial/mode | `pure_ucy_transformer / bounded_endpoint_residual` | source-only neural retrain protocol |
| strict pure UCY neural all/t50/hard/easy | `9.01% / 8.80% / 9.36% / 0.00%` | bounded residual policy selected on validation; raw no-fallback neural remains unsafe |
| strict pure UCY neural bootstrap stable | `True` | 2000-bootstrap lower bounds positive for all/t50/t100/hard |
| strict pure UCY neural bootstrap lows all/t50/t100/hard | `8.89% / 8.63% / 8.07% / 9.23%` | strict pure-UCY source-only neural statistical evidence |
| endpoint-to-full bridge gate | `True` | positive full-waypoint bridge evidence, not learned shape |
| endpoint-to-full bridge positive domains | `['ETH_UCY', 'TrajNet']` | ETH_UCY and TrajNet if pass |
| calibrated learned-shape meta-policy gate | `True` | positive learned-shape residual evidence under fallback |
| calibrated learned-shape positive domains | `['ETH_UCY', 'TrajNet']` | ETH_UCY and TrajNet if pass |
| JEPA deployable path | `disabled` | JEPA had no deployable downstream lift |
| fixed-prior source switch beats fixed composer | `False` | negative branch audit |
| residual source-switch oracle headroom | `False` | negative branch audit |
| Stage5C executed | `False` | must remain false |
| SMC enabled | `False` | must remain false |

## Per-Domain Metrics

| Domain | all | t+50 | t+100 diagnostic | hard/failure | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | 19.06% | 13.58% | 13.46% | 18.71% | 0.00% | 39.38% |
| TrajNet | 22.95% | 16.62% | 14.02% | 21.96% | 0.00% | 27.65% |
| UCY | 23.27% | 9.14% | 19.26% | 22.60% | 0.00% | 33.34% |
