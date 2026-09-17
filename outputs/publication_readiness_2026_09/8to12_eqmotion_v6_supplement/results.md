# Fixed-Forecast Supplement

Development-only, all three seeds and every frozen cost-head/policy combination. No additional model selection.
Raw50 is the exact native-grid prefix of the original 12-step prediction. Its scale is not refitted.
Primary full-horizon evaluation remains authoritative; this table cannot rescue a failed primary result.

| Seed | Candidate | Arm | Raw50 ADE gain % | Raw50 FDE gain % | Full-path easy degradation % (prefix ADE) |
| --- | --- | --- | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | independent | -5.16596 | -0.809737 | 36188.3 |
| 17 | seed17_ridge_conservative | independent_count_reference | -5.16596 | -0.809737 | 36188.3 |
| 17 | seed17_ridge_conservative | joint | -5.16596 | -0.809737 | 36188.3 |
| 17 | seed17_ridge_conservative | joint_exact_count | -5.16596 | -0.809737 | 36188.3 |
| 17 | seed17_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | uncontrolled | -50.9613 | -9.21366 | 350882 |
| 17 | seed17_ridge_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | independent | -5.22458 | -0.781078 | 36277.5 |
| 17 | seed17_ridge_moderate | independent_count_reference | -5.22458 | -0.781078 | 36277.5 |
| 17 | seed17_ridge_moderate | joint | -5.22458 | -0.781078 | 36277.5 |
| 17 | seed17_ridge_moderate | joint_exact_count | -5.22458 | -0.781078 | 36277.5 |
| 17 | seed17_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | uncontrolled | -50.9613 | -9.21366 | 350882 |
| 17 | seed17_neural_cost_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | independent | -0.0389457 | -0.00264184 | 275.319 |
| 17 | seed17_neural_cost_conservative | independent_count_reference | -0.0389457 | -0.00264184 | 275.319 |
| 17 | seed17_neural_cost_conservative | joint | -0.0389457 | -0.00264184 | 275.319 |
| 17 | seed17_neural_cost_conservative | joint_exact_count | -0.0389457 | -0.00264184 | 275.319 |
| 17 | seed17_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | uncontrolled | -50.9613 | -9.21366 | 350882 |
| 17 | seed17_neural_cost_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | independent | -0.0815409 | -0.00510725 | 407.535 |
| 17 | seed17_neural_cost_moderate | independent_count_reference | -0.0815409 | -0.00510725 | 407.535 |
| 17 | seed17_neural_cost_moderate | joint | -0.0815409 | -0.00510725 | 407.535 |
| 17 | seed17_neural_cost_moderate | joint_exact_count | -0.0815409 | -0.00510725 | 407.535 |
| 17 | seed17_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | uncontrolled | -50.9613 | -9.21366 | 350882 |
| 29 | seed29_ridge_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | independent | -7.85381 | -3.04091 | 57116.9 |
| 29 | seed29_ridge_conservative | independent_count_reference | -7.85381 | -3.04091 | 57116.9 |
| 29 | seed29_ridge_conservative | joint | -7.85409 | -3.04117 | 57117.8 |
| 29 | seed29_ridge_conservative | joint_exact_count | -7.85409 | -3.04117 | 57117.8 |
| 29 | seed29_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | uncontrolled | -35.8811 | -13.5846 | 245194 |
| 29 | seed29_ridge_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | independent | -8.08231 | -3.07592 | 57617.2 |
| 29 | seed29_ridge_moderate | independent_count_reference | -8.08231 | -3.07592 | 57617.2 |
| 29 | seed29_ridge_moderate | joint | -8.0821 | -3.07569 | 57617.2 |
| 29 | seed29_ridge_moderate | joint_exact_count | -8.0821 | -3.07569 | 57617.2 |
| 29 | seed29_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | uncontrolled | -35.8811 | -13.5846 | 245194 |
| 29 | seed29_neural_cost_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | independent | -0.187641 | -0.0630533 | 1153.2 |
| 29 | seed29_neural_cost_conservative | independent_count_reference | -0.187641 | -0.0630533 | 1153.2 |
| 29 | seed29_neural_cost_conservative | joint | -0.187641 | -0.0630533 | 1153.2 |
| 29 | seed29_neural_cost_conservative | joint_exact_count | -0.187641 | -0.0630533 | 1153.2 |
| 29 | seed29_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | uncontrolled | -35.8811 | -13.5846 | 245194 |
| 29 | seed29_neural_cost_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | independent | -0.384037 | -0.13489 | 2666.36 |
| 29 | seed29_neural_cost_moderate | independent_count_reference | -0.384037 | -0.13489 | 2666.36 |
| 29 | seed29_neural_cost_moderate | joint | -0.384037 | -0.13489 | 2666.36 |
| 29 | seed29_neural_cost_moderate | joint_exact_count | -0.384037 | -0.13489 | 2666.36 |
| 29 | seed29_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | uncontrolled | -35.8811 | -13.5846 | 245194 |
| 43 | seed43_ridge_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | independent | -1.69055 | -1.25613 | 9909.19 |
| 43 | seed43_ridge_conservative | independent_count_reference | -1.69055 | -1.25613 | 9909.19 |
| 43 | seed43_ridge_conservative | joint | -1.69055 | -1.25613 | 9909.19 |
| 43 | seed43_ridge_conservative | joint_exact_count | -1.69055 | -1.25613 | 9909.19 |
| 43 | seed43_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | uncontrolled | -61.2295 | -46.8816 | 426549 |
| 43 | seed43_ridge_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | independent | -1.75907 | -1.30387 | 10403.9 |
| 43 | seed43_ridge_moderate | independent_count_reference | -1.75907 | -1.30387 | 10403.9 |
| 43 | seed43_ridge_moderate | joint | -1.75907 | -1.30387 | 10403.9 |
| 43 | seed43_ridge_moderate | joint_exact_count | -1.75907 | -1.30387 | 10403.9 |
| 43 | seed43_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | uncontrolled | -61.2295 | -46.8816 | 426549 |
| 43 | seed43_neural_cost_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | independent | -0.15759 | -0.0726072 | 950.816 |
| 43 | seed43_neural_cost_conservative | independent_count_reference | -0.15759 | -0.0726072 | 950.816 |
| 43 | seed43_neural_cost_conservative | joint | -0.15759 | -0.0726072 | 950.816 |
| 43 | seed43_neural_cost_conservative | joint_exact_count | -0.15759 | -0.0726072 | 950.816 |
| 43 | seed43_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | uncontrolled | -61.2295 | -46.8816 | 426549 |
| 43 | seed43_neural_cost_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | independent | -0.238565 | -0.113864 | 1430.06 |
| 43 | seed43_neural_cost_moderate | independent_count_reference | -0.238565 | -0.113864 | 1430.06 |
| 43 | seed43_neural_cost_moderate | joint | -0.238565 | -0.113864 | 1430.06 |
| 43 | seed43_neural_cost_moderate | joint_exact_count | -0.238565 | -0.113864 | 1430.06 |
| 43 | seed43_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | uncontrolled | -61.2295 | -46.8816 | 426549 |

## Matched Nonzero Intervention Counts

| Seed | Candidate | ADE joint minus independent | Scored agent queries | Matched nonzero scene queries | Solver/unmatched queries |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | 0 | 25893 | 883 | 0 |
| 17 | seed17_ridge_moderate | 0 | 27686 | 942 | 0 |
| 17 | seed17_neural_cost_conservative | 0 | 3652 | 121 | 0 |
| 17 | seed17_neural_cost_moderate | 0 | 9061 | 301 | 0 |
| 29 | seed29_ridge_conservative | 9.04878e-06 | 28298 | 968 | 0 |
| 29 | seed29_ridge_moderate | -1.244e-05 | 28298 | 968 | 0 |
| 29 | seed29_neural_cost_conservative | 0 | 15022 | 508 | 0 |
| 29 | seed29_neural_cost_moderate | 0 | 20175 | 679 | 0 |
| 43 | seed43_ridge_conservative | 0 | 22242 | 755 | 0 |
| 43 | seed43_ridge_moderate | 0 | 22853 | 774 | 0 |
| 43 | seed43_neural_cost_conservative | 0 | 17649 | 591 | 0 |
| 43 | seed43_neural_cost_moderate | 0 | 21526 | 728 | 0 |

Negative differences favor joint selection on the same frozen candidates. Count matching applies to all past-supported agents before label access; scored-only coverage need not match after incomplete labels are filtered. Zero-count queries are reported in JSON but are not evidence of coordination.
One physical development site does not permit an informative scene-bootstrap CI. Seed SD describes training randomness only. Coordinates and time remain unverified dataset-local/native-frame units. No formal safety or deployment claim.
