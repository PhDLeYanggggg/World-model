# M3W-Neural v1 Evidence Matrix

- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.
- package_input_hash: `ebd6b8b7e1ea7076d120ab50248e93abd1815812cea8b62fc55ba0e02868c541`
- git_commit: `d42fd81`

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
