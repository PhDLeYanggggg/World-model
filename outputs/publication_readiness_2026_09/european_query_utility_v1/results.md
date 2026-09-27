# Frozen Query Utility Allocation Results

## Material Passport

Fresh causal allocation and outcome readout; cached_verified estimators, source roles and forecasts. No new training.
Twelve opened development localities, three forecasting seeds and 108 paired groups. Independent roles remain closed.
Joint utility and independent admission have identical switch counts in each current query. Uniform admission is not rate matched; top-k is a risk-unconstrained diagnostic.

## Registered Primary

Joint utility versus independent ADE gain (%): **0.2322 [0.1225, 0.3586]**.
Predicted feasibility is not observed risk control. All safety gates below must be retained.

| Policy | ADE / floor gain % | Hard / floor gain % | Easy / CV gain % | FDE / floor gain % | Intervention % | Selected positive harm % |
|---|---:|---:|---:|---:|---:|---:|
| control_matched_count | 0.2731 [0.1706, 0.3918] | 0.2216 [0.0998, 0.3686] | 5.6187 [4.3986, 6.7469] | 0.3930 [0.2547, 0.5562] | 8.0235 [6.4457, 9.5834] | undefined |
| floor | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 3.8347 [2.9848, 4.5433] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | undefined |
| independent | 0.2594 [0.1632, 0.3687] | 0.1923 [0.0801, 0.3321] | 5.6406 [4.4125, 6.7641] | 0.3719 [0.2449, 0.5184] | 8.0235 [6.4457, 9.5834] | undefined |
| joint_utility | 0.4882 [0.2998, 0.7062] | 0.4996 [0.2634, 0.7590] | 5.3507 [4.1625, 6.4835] | 0.6946 [0.4400, 0.9963] | 8.0235 [6.4457, 9.5834] | undefined |
| query_uniform | 0.0429 [0.0217, 0.0680] | 0.0350 [0.0075, 0.0732] | 4.0103 [3.2055, 4.6942] | 0.0651 [0.0338, 0.0986] | 1.1243 [0.6112, 1.7501] | undefined |
| utility_topk | 1.6807 [1.1117, 2.2587] | 2.4280 [1.7055, 3.1192] | 3.7559 [2.9407, 4.5040] | 2.4573 [1.6717, 3.2602] | 8.0235 [6.4457, 9.5834] | undefined |

## Same-Count Contrasts

| Joint versus control | ADE gain % | Harm reduction pp | Count difference pp |
|---|---:|---:|---:|
| control_matched_count | 0.2185 [0.1176, 0.3310] | undefined | 0.0000 [0.0000, 0.0000] |
| independent | 0.2322 [0.1225, 0.3586] | undefined | 0.0000 [0.0000, 0.0000] |
| utility_topk | -1.2523 [-1.7375, -0.7767] | undefined | 0.0000 [0.0000, 0.0000] |

## Worst Views, Missing Labels and Tail Error

| Policy | Worst easy gain % | Risk violations /216 | Undefined views | P95 / floor P95 | Unknown interventions /view | Complete ADE gain % | Partial ADE gain % |
|---|---:|---:|---:|---:|---:|---:|---:|
| control_matched_count | 0.3716 | 87 | 10 | 0.9985 [0.9974, 0.9994] | 24.0972 [5.2728, 51.3525] | 0.3567 [0.2267, 0.5074] | 0.1709 [0.1080, 0.2459] |
| floor | 0.1214 | 0 | 216 | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| independent | 0.3716 | 82 | 10 | 0.9988 [0.9979, 0.9996] | 23.4954 [5.1706, 49.8024] | 0.3442 [0.2190, 0.4885] | 0.1556 [0.0978, 0.2217] |
| joint_utility | 0.3716 | 99 | 10 | 0.9963 [0.9941, 0.9982] | 28.0787 [5.3191, 59.1446] | 0.6184 [0.3786, 0.8942] | 0.3218 [0.1986, 0.4664] |
| query_uniform | 0.2586 | 59 | 46 | 0.9998 [0.9995, 1.0000] | 1.2917 [0.6619, 1.9909] | 0.0572 [0.0292, 0.0879] | 0.0287 [0.0131, 0.0483] |
| utility_topk | -3.2797 | 186 | 11 | 0.9847 [0.9778, 0.9912] | 55.3611 [10.0814, 120.7906] | 2.0224 [1.3278, 2.7402] | 1.2371 [0.7499, 1.7576] |

## Locality and Seed Evidence

| Locality | Joint versus independent ADE gain % |
|---|---:|
| eu-locality-007 | 0.172465 |
| eu-locality-008 | 0.219265 |
| eu-locality-020 | 0.000653 |
| eu-locality-048 | 0.068031 |
| eu-locality-067 | 0.054382 |
| eu-locality-074 | 0.442116 |
| eu-locality-082 | 0.026365 |
| eu-locality-110 | 0.725591 |
| eu-locality-112 | 0.046469 |
| eu-locality-119 | 0.451319 |
| eu-locality-124 | 0.256330 |
| eu-locality-126 | 0.323079 |

| Forecaster seed | Independent ADE / floor gain % | Joint ADE / floor gain % |
|---|---:|---:|
| 17 | 0.3692 [0.1788, 0.5986] | 0.6550 [0.3398, 1.0551] |
| 29 | 0.1907 [0.1244, 0.2639] | 0.3643 [0.2396, 0.4971] |
| 43 | 0.2183 [0.1378, 0.3098] | 0.4453 [0.2659, 0.6626] |

## Solver and Gates

- queries: 747900
- milp_queries: 181683
- solver_fallback_queries: 22
- changed_queries: 66258
- changed_agents: 236626
- expected_utility_gain: 34307.84503300467
- uniform_queries: 8749

- primary_ADE_advantage: True
- every_view_defined_risk_within_budget: False
- every_view_easy_preserved: True
- no_zero_CV_harm: True
- same_intervention_count: True
- nonzero_each_locality: True
- exploratory_joint_screen_pass: False
- independent_confirmation: False
- calibration_certificate: False
- deployment_changed: False
- stage5c_executed: False
- smc_enabled: False

Expected utility totals use per-head fitting-only normalized scores over repeated query views, not a unique-population benefit estimate.
The four neural components are signed-risk score bases, not separately identified calibrated moments. Aggregate predicted excess constraints can be satisfied while actual selected harm fails.
There is no pairwise interaction penalty in this contrast. This is a joint budget allocation test, not proof of nonadditive interaction or full-visible-scene safety.
Bootstrap uses 3,000 draws over 12 locality means, after averaging dependent producer/fit/seed views. No IID-window, multiple-comparison-adjusted or independent confirmation claim.
Image-local detector silver; obs8/pred12 rawstride12. No metric, seconds, human-gold, physical-safety, true3D or foundation claims. Stage5C and SMC remain disabled.
