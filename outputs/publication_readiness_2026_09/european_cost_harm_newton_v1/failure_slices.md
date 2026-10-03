# Frozen Cost-Control Failure Slices

Fresh aggregate diagnosis of hash-verified reports; no refitting or selection. These are
post-hoc exposed-development slices, not independent hypothesis tests.

| Locality | Heads | Cost-original signed MSE | Full utility % | Matched utility % |
|---|---:|---:|---:|---:|
| eu-locality-007 | 6 | +0.058842 | +0.000026 | +0.000018 |
| eu-locality-008 | 6 | -0.001634 | +0.005107 | +0.003455 |
| eu-locality-020 | 6 | -0.026041 | -0.000055 | +0.000000 |
| eu-locality-048 | 6 | -0.008883 | +0.001826 | +0.000348 |
| eu-locality-067 | 6 | +0.256689 | +0.000180 | +0.000252 |
| eu-locality-074 | 6 | +0.006451 | +0.000057 | +0.000000 |
| eu-locality-082 | 6 | -0.019972 | +0.000075 | +0.000000 |
| eu-locality-110 | 6 | +0.653188 | +0.014677 | +0.010237 |
| eu-locality-112 | 6 | +0.091338 | +0.000000 | +0.000000 |
| eu-locality-119 | 6 | +0.046542 | +0.004986 | +0.002591 |
| eu-locality-124 | 6 | +0.188150 | +0.000000 | +0.000000 |
| eu-locality-126 | 6 | +0.005169 | +0.042003 | +0.015514 |

Utility is percent of full known reference error mass, not ADE/FDE improvement.

| Source / seed / producer | Selected | Unknown | Known easy risk % | Completion upper % |
|---|---:|---:|---:|---:|
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 17 | 284 | 10 | 1.8400 | 5.7195 |
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 29 | 289 | 10 | 1.4987 | 5.1526 |
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 43 | 263 | 9 | 1.7398 | 5.1508 |
| single0_seed43_controller2_dimensionless_fit_eu-locality-124 / 43 | 2 | 0 | 2.7486 | 2.7486 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 17 | 2 | 0 | 2.8253 | 2.8253 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 29 | 2 | 0 | 2.8253 | 2.8253 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 43 | 2 | 0 | 2.8253 | 2.8253 |

Three upper failures in locality067 require unknown completion; four failures
in112/124 already exist on known labels. Unknown upper bounds are not observed harm.
All144 head-channel training losses decrease; maximum iterations20. The completed
experiment is therefore not explained by the earlier solver nonconvergence.

No data role, forecast, policy threshold or deployment is changed.
