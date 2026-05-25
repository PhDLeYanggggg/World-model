# M3W-Neural v1 Evidence Matrix

- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.
- package_input_hash: `578b8aeb131f93e30f8b2ea681b63a8116114f0213ae0cca4bdb6640384e4bf5`
- git_commit: `e8b477d`

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
| pure UCY source-heldout gate | `True` | required for UCY held-out support |
| pure UCY-only retrain/select/test gate | `False` | reported blocker, not claimed |
| JEPA deployable path | `disabled` | JEPA had no deployable downstream lift |
| Stage5C executed | `False` | must remain false |
| SMC enabled | `False` | must remain false |

## Per-Domain Metrics

| Domain | all | t+50 | t+100 diagnostic | hard/failure | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | 19.06% | 13.58% | 13.46% | 18.71% | 0.00% | 39.38% |
| TrajNet | 22.95% | 16.62% | 14.02% | 21.96% | 0.00% | 27.65% |
| UCY | 23.27% | 9.14% | 19.26% | 22.60% | 0.00% | 33.34% |
