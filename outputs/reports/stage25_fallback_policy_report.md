# Stage 25 Conservative Fallback Policy Search

- best by t+50: `regret_selector`
- best by hard/failure: `regret_selector`
- safest: `all_fallback_strongest`
- selected deployment policy: `regret_selector`

| policy | t50 improvement | hard/failure improvement | easy degradation | harm over fallback |
| --- | ---: | ---: | ---: | ---: |
| regret_selector | 0.013573 | 0.010930 | 0.009655 | -0.281194 |
| failure_assisted_selector | 0.010378 | 0.007118 | 0.009405 | -0.183136 |
| all_fallback_strongest | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
