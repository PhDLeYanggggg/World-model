# M3W-Neural v1 Evidence Matrix

- result_source: `cached_verified` from Stage41 fresh reports, hashes recorded below.
- package_input_hash: `017c27e874ecb9f08b5d8bf71edde0f891318e8eede796e1ad80bb3741946a4c`
- git_commit: `0aca3fe`

| Evidence | Value | Gate interpretation |
| --- | --- | --- |
| Stage41 gates | `41 / 41` | pass if all gates true |
| endpoint geometry pass | `True` | required |
| all improvement vs Stage37 floor | `41.96%` | must be positive |
| t+50 improvement vs Stage37 floor | `40.62%` | must be positive |
| t+100 raw-frame diagnostic | `45.73%` | diagnostic only |
| hard/failure improvement | `43.61%` | must improve |
| easy degradation | `0.00%` | must be <= 2% |
| switch rate | `50.84%` | reported for deployment risk |
| positive external domains | `3` | must be >= 2 for cross-domain evidence |
| Stage5C executed | `False` | must remain false |
| SMC enabled | `False` | must remain false |

## Per-Domain Metrics

| Domain | all | t+50 | t+100 diagnostic | hard/failure | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETH_UCY | 47.20% | 45.44% | 53.86% | 48.75% | 0.00% | 58.38% |
| TrajNet | 51.46% | 50.99% | 60.61% | 53.92% | 0.00% | 53.40% |
| UCY | 12.94% | 10.37% | 0.00% | 13.01% | 0.00% | 25.00% |
