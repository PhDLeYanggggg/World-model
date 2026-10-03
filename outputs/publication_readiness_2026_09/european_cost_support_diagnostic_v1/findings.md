# Frozen Support and Objective Findings

Post-hoc development diagnosis of frozen predictions, not new training, policy selection,
independent confirmation or a causal ablation. Counts across heads are repeated occurrences.

## Optimization Versus Generalization

| Quantity | Heads / 72 |
|---|---:|
| train_tree_decreased | 72 |
| train_raw_ensemble_decreased | 72 |
| train_projected_ensemble_decreased | 72 |
| validation_tree_increased | 63 |
| validation_raw_ensemble_increased | 55 |
| validation_projected_ensemble_increased | 43 |

| Locality | TRAIN raw ensemble change | TRAIN projected change | Validation raw change | Validation projected change |
|---|---:|---:|---:|---:|
| eu-locality-007 | -0.075885 | -0.075611 | +2877.467712 | +0.058842 |
| eu-locality-008 | -0.081978 | -0.081247 | +0.863796 | -0.001634 |
| eu-locality-020 | -0.146644 | -0.145519 | -0.026210 | -0.026041 |
| eu-locality-048 | -0.120880 | -0.077273 | -0.000579 | -0.008883 |
| eu-locality-067 | -0.242801 | -0.181355 | +75213.817955 | +0.256689 |
| eu-locality-074 | -0.139646 | -0.138217 | +21.414122 | +0.006451 |
| eu-locality-082 | -0.089895 | -0.086309 | -0.021352 | -0.019972 |
| eu-locality-110 | -0.212592 | -0.209418 | +67936.496083 | +0.653188 |
| eu-locality-112 | -0.146232 | -0.111025 | +3.226949 | +0.091338 |
| eu-locality-119 | -0.158288 | -0.146084 | +0.091038 | +0.046542 |
| eu-locality-124 | -0.070652 | -0.070191 | +0.313092 | +0.188150 |
| eu-locality-126 | -0.201369 | -0.168633 | +1.892104 | +0.005169 |

Negative change is better normalized signed-score MSE. These are not ADE/FDE gains.
Do not add overlapping support strata as separate causes. Leaf quality boxes are TRAIN
min/max descriptors, not guarantees, prediction intervals or new action filters.

## Upper-Risk Failure Support

| Source / seed / producer | Selected | Unknown | Upper % | Original actions retained | Selected known EH rows | Mean zero-TRAIN-EH tree fraction | Mean outside-quality tree fraction |
|---|---:|---:|---:|---|---:|---:|---:|
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 17 | 284 | 10 | 5.7195 | False | 64 | 0.000000 | 0.076904 |
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 29 | 289 | 10 | 5.1526 | False | 59 | 0.000000 | 0.095604 |
| single0_seed43_controller1_dimensionless_fit_eu-locality-067 / 43 | 263 | 9 | 5.1508 | False | 55 | 0.000000 | 0.092898 |
| single0_seed43_controller2_dimensionless_fit_eu-locality-124 / 43 | 2 | 0 | 2.7486 | True | 1 | 0.000000 | 1.000000 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 17 | 2 | 0 | 2.8253 | True | 2 | 0.000000 | 0.718750 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 29 | 2 | 0 | 2.8253 | True | 2 | 0.000000 | 0.683594 |
| single1_seed43_controller2_dimensionless_fit_eu-locality-112 / 43 | 2 | 0 | 2.8253 | True | 2 | 0.007812 | 0.695312 |

Upper risk includes completion for unknown outcomes and is not observed harm.
Unavailable future labels are retained as unknown in policy evaluation; label-based
cohorts above are offline diagnosis only. Stage5C/SMC remain off.
