# Fixed-Forecast Supplement

Development-only, all three seeds and every frozen cost-head/policy combination. No additional model selection.
Raw50 is the exact native-grid prefix of the original 12-step prediction. Its scale is not refitted.
Primary full-horizon evaluation remains authoritative; this table cannot rescue a failed primary result.

| Seed | Candidate | Arm | Raw50 ADE gain % | Raw50 FDE gain % | Full-path easy degradation % (prefix ADE) |
| --- | --- | --- | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | independent | -0.000454211 | 8.52818e-05 | 6.49374 |
| 17 | seed17_ridge_conservative | independent_count_reference | -0.000454211 | 8.52818e-05 | 6.49374 |
| 17 | seed17_ridge_conservative | joint | -0.000454211 | 8.52818e-05 | 6.49374 |
| 17 | seed17_ridge_conservative | joint_exact_count | -0.000454211 | 8.52818e-05 | 6.49374 |
| 17 | seed17_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | uncontrolled | -3.08249 | -1.19326 | 21430.5 |
| 17 | seed17_ridge_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | independent | 0.000516914 | 0.00136213 | 7.58435 |
| 17 | seed17_ridge_moderate | independent_count_reference | 0.000516914 | 0.00136213 | 7.58435 |
| 17 | seed17_ridge_moderate | joint | 0.000516914 | 0.00136213 | 7.58435 |
| 17 | seed17_ridge_moderate | joint_exact_count | 0.000516914 | 0.00136213 | 7.58435 |
| 17 | seed17_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | uncontrolled | -3.08249 | -1.19326 | 21430.5 |
| 17 | seed17_neural_cost_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | independent | -2.52712e-05 | 3.6551e-05 | 0.0258238 |
| 17 | seed17_neural_cost_conservative | independent_count_reference | -2.52712e-05 | 3.6551e-05 | 0.0258238 |
| 17 | seed17_neural_cost_conservative | joint | -2.52712e-05 | 3.6551e-05 | 0.0258238 |
| 17 | seed17_neural_cost_conservative | joint_exact_count | -2.52712e-05 | 3.6551e-05 | 0.0258238 |
| 17 | seed17_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | uncontrolled | -3.08249 | -1.19326 | 21430.5 |
| 17 | seed17_neural_cost_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | independent | 0.00041607 | 0.000677401 | 0.832283 |
| 17 | seed17_neural_cost_moderate | independent_count_reference | 0.00041607 | 0.000677401 | 0.832283 |
| 17 | seed17_neural_cost_moderate | joint | 0.00041607 | 0.000677401 | 0.832283 |
| 17 | seed17_neural_cost_moderate | joint_exact_count | 0.00041607 | 0.000677401 | 0.832283 |
| 17 | seed17_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | uncontrolled | -3.08249 | -1.19326 | 21430.5 |
| 29 | seed29_ridge_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | independent | -0.514584 | -0.23536 | 3244.71 |
| 29 | seed29_ridge_conservative | independent_count_reference | -0.514584 | -0.23536 | 3244.71 |
| 29 | seed29_ridge_conservative | joint | -0.514584 | -0.23536 | 3244.71 |
| 29 | seed29_ridge_conservative | joint_exact_count | -0.514584 | -0.23536 | 3244.71 |
| 29 | seed29_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | uncontrolled | -2.76153 | -1.2587 | 19365.8 |
| 29 | seed29_ridge_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | independent | -0.564792 | -0.257251 | 3597.63 |
| 29 | seed29_ridge_moderate | independent_count_reference | -0.564792 | -0.257251 | 3597.63 |
| 29 | seed29_ridge_moderate | joint | -0.564792 | -0.257251 | 3597.63 |
| 29 | seed29_ridge_moderate | joint_exact_count | -0.564792 | -0.257251 | 3597.63 |
| 29 | seed29_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | uncontrolled | -2.76153 | -1.2587 | 19365.8 |
| 29 | seed29_neural_cost_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | independent | -0.00090151 | -0.000495387 | 0.0275832 |
| 29 | seed29_neural_cost_conservative | independent_count_reference | -0.00090151 | -0.000495387 | 0.0275832 |
| 29 | seed29_neural_cost_conservative | joint | -0.00090151 | -0.000495387 | 0.0275832 |
| 29 | seed29_neural_cost_conservative | joint_exact_count | -0.00090151 | -0.000495387 | 0.0275832 |
| 29 | seed29_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | uncontrolled | -2.76153 | -1.2587 | 19365.8 |
| 29 | seed29_neural_cost_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | independent | -0.00064654 | 5.7724e-06 | 0.433196 |
| 29 | seed29_neural_cost_moderate | independent_count_reference | -0.00064654 | 5.7724e-06 | 0.433196 |
| 29 | seed29_neural_cost_moderate | joint | -0.00064654 | 5.7724e-06 | 0.433196 |
| 29 | seed29_neural_cost_moderate | joint_exact_count | -0.00064654 | 5.7724e-06 | 0.433196 |
| 29 | seed29_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | uncontrolled | -2.76153 | -1.2587 | 19365.8 |
| 43 | seed43_ridge_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | independent | -0.588036 | -0.09386 | 3683.22 |
| 43 | seed43_ridge_conservative | independent_count_reference | -0.588036 | -0.09386 | 3683.22 |
| 43 | seed43_ridge_conservative | joint | -0.588036 | -0.09386 | 3683.22 |
| 43 | seed43_ridge_conservative | joint_exact_count | -0.588036 | -0.09386 | 3683.22 |
| 43 | seed43_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | uncontrolled | -1.90449 | -0.287103 | 12868.5 |
| 43 | seed43_ridge_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | independent | -0.616414 | -0.0953569 | 3875.22 |
| 43 | seed43_ridge_moderate | independent_count_reference | -0.616414 | -0.0953569 | 3875.22 |
| 43 | seed43_ridge_moderate | joint | -0.616414 | -0.0953569 | 3875.22 |
| 43 | seed43_ridge_moderate | joint_exact_count | -0.616414 | -0.0953569 | 3875.22 |
| 43 | seed43_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | uncontrolled | -1.90449 | -0.287103 | 12868.5 |
| 43 | seed43_neural_cost_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | independent | -0.0656323 | -0.00908372 | 364.797 |
| 43 | seed43_neural_cost_conservative | independent_count_reference | -0.0656323 | -0.00908372 | 364.797 |
| 43 | seed43_neural_cost_conservative | joint | -0.0656323 | -0.00908372 | 364.797 |
| 43 | seed43_neural_cost_conservative | joint_exact_count | -0.0656323 | -0.00908372 | 364.797 |
| 43 | seed43_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | uncontrolled | -1.90449 | -0.287103 | 12868.5 |
| 43 | seed43_neural_cost_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | independent | -0.102661 | -0.0127243 | 584.582 |
| 43 | seed43_neural_cost_moderate | independent_count_reference | -0.102661 | -0.0127243 | 584.582 |
| 43 | seed43_neural_cost_moderate | joint | -0.102661 | -0.0127243 | 584.582 |
| 43 | seed43_neural_cost_moderate | joint_exact_count | -0.102661 | -0.0127243 | 584.582 |
| 43 | seed43_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | uncontrolled | -1.90449 | -0.287103 | 12868.5 |

## Matched Nonzero Intervention Counts

| Seed | Candidate | ADE joint minus independent | Scored agent queries | Matched nonzero scene queries | Solver/unmatched queries |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | 0 | 4792 | 153 | 0 |
| 17 | seed17_ridge_moderate | 0 | 18395 | 614 | 0 |
| 17 | seed17_neural_cost_conservative | 0 | 1827 | 62 | 0 |
| 17 | seed17_neural_cost_moderate | 0 | 9498 | 314 | 0 |
| 29 | seed29_ridge_conservative | 0 | 11503 | 426 | 0 |
| 29 | seed29_ridge_moderate | 0 | 20395 | 710 | 0 |
| 29 | seed29_neural_cost_conservative | 0 | 1961 | 68 | 0 |
| 29 | seed29_neural_cost_moderate | 0 | 8249 | 272 | 0 |
| 43 | seed43_ridge_conservative | 0 | 25125 | 861 | 0 |
| 43 | seed43_ridge_moderate | 0 | 27976 | 959 | 0 |
| 43 | seed43_neural_cost_conservative | 0 | 17734 | 603 | 0 |
| 43 | seed43_neural_cost_moderate | 0 | 23830 | 814 | 0 |

Negative differences favor joint selection on the same frozen candidates. Count matching applies to all past-supported agents before label access; scored-only coverage need not match after incomplete labels are filtered. Zero-count queries are reported in JSON but are not evidence of coordination.
One physical development site does not permit an informative scene-bootstrap CI. Seed SD describes training randomness only. Coordinates and time remain unverified dataset-local/native-frame units. No formal safety or deployment claim.
