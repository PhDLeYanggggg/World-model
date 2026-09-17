# Fixed-Forecast Supplement

Development-only, all three seeds and every frozen cost-head/policy combination. No additional model selection.
Raw50 is the exact native-grid prefix of the original 12-step prediction. Its scale is not refitted.
Primary full-horizon evaluation remains authoritative; this table cannot rescue a failed primary result.

| Seed | Candidate | Arm | Raw50 ADE gain % | Raw50 FDE gain % | Full-path easy degradation % (prefix ADE) |
| --- | --- | --- | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | independent | -0.844471 | -0.415027 | 5260.86 |
| 17 | seed17_ridge_conservative | independent_count_reference | -0.844471 | -0.415027 | 5260.86 |
| 17 | seed17_ridge_conservative | joint | -0.844471 | -0.415027 | 5260.86 |
| 17 | seed17_ridge_conservative | joint_exact_count | -0.844471 | -0.415027 | 5260.86 |
| 17 | seed17_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | uncontrolled | -21.643 | -10.0773 | 143234 |
| 17 | seed17_ridge_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | independent | -0.890579 | -0.426803 | 5546.82 |
| 17 | seed17_ridge_moderate | independent_count_reference | -0.890579 | -0.426803 | 5546.82 |
| 17 | seed17_ridge_moderate | joint | -0.890579 | -0.426803 | 5546.82 |
| 17 | seed17_ridge_moderate | joint_exact_count | -0.890579 | -0.426803 | 5546.82 |
| 17 | seed17_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | uncontrolled | -21.643 | -10.0773 | 143234 |
| 17 | seed17_neural_cost_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | independent | -0.0252929 | -0.0114545 | 133.948 |
| 17 | seed17_neural_cost_conservative | independent_count_reference | -0.0252929 | -0.0114545 | 133.948 |
| 17 | seed17_neural_cost_conservative | joint | -0.0252929 | -0.0114545 | 133.948 |
| 17 | seed17_neural_cost_conservative | joint_exact_count | -0.0252929 | -0.0114545 | 133.948 |
| 17 | seed17_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | uncontrolled | -21.643 | -10.0773 | 143234 |
| 17 | seed17_neural_cost_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | independent | -0.0558312 | -0.0290417 | 208.944 |
| 17 | seed17_neural_cost_moderate | independent_count_reference | -0.0558312 | -0.0290417 | 208.944 |
| 17 | seed17_neural_cost_moderate | joint | -0.0558312 | -0.0290417 | 208.944 |
| 17 | seed17_neural_cost_moderate | joint_exact_count | -0.0558312 | -0.0290417 | 208.944 |
| 17 | seed17_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | uncontrolled | -21.643 | -10.0773 | 143234 |
| 29 | seed29_ridge_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | independent | -0.126773 | -0.0288104 | 971.45 |
| 29 | seed29_ridge_conservative | independent_count_reference | -0.126773 | -0.0288104 | 971.45 |
| 29 | seed29_ridge_conservative | joint | -0.126773 | -0.0288104 | 971.45 |
| 29 | seed29_ridge_conservative | joint_exact_count | -0.126773 | -0.0288104 | 971.45 |
| 29 | seed29_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | uncontrolled | -22.2886 | -12.0036 | 147473 |
| 29 | seed29_ridge_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | independent | -0.173983 | -0.0318445 | 1162.08 |
| 29 | seed29_ridge_moderate | independent_count_reference | -0.173983 | -0.0318445 | 1162.08 |
| 29 | seed29_ridge_moderate | joint | -0.173983 | -0.0318445 | 1162.08 |
| 29 | seed29_ridge_moderate | joint_exact_count | -0.173983 | -0.0318445 | 1162.08 |
| 29 | seed29_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | uncontrolled | -22.2886 | -12.0036 | 147473 |
| 29 | seed29_neural_cost_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | independent | -0.0196754 | -0.00915339 | 28.2629 |
| 29 | seed29_neural_cost_conservative | independent_count_reference | -0.0196754 | -0.00915339 | 28.2629 |
| 29 | seed29_neural_cost_conservative | joint | -0.0196754 | -0.00915339 | 28.2629 |
| 29 | seed29_neural_cost_conservative | joint_exact_count | -0.0196754 | -0.00915339 | 28.2629 |
| 29 | seed29_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | uncontrolled | -22.2886 | -12.0036 | 147473 |
| 29 | seed29_neural_cost_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | independent | -0.0333838 | -0.0121048 | 66.1507 |
| 29 | seed29_neural_cost_moderate | independent_count_reference | -0.0333838 | -0.0121048 | 66.1507 |
| 29 | seed29_neural_cost_moderate | joint | -0.0333838 | -0.0121048 | 66.1507 |
| 29 | seed29_neural_cost_moderate | joint_exact_count | -0.0333838 | -0.0121048 | 66.1507 |
| 29 | seed29_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | uncontrolled | -22.2886 | -12.0036 | 147473 |
| 43 | seed43_ridge_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | independent | -0.462493 | -0.293626 | 2519.12 |
| 43 | seed43_ridge_conservative | independent_count_reference | -0.462493 | -0.293626 | 2519.12 |
| 43 | seed43_ridge_conservative | joint | -0.462493 | -0.293626 | 2519.12 |
| 43 | seed43_ridge_conservative | joint_exact_count | -0.462493 | -0.293626 | 2519.12 |
| 43 | seed43_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | uncontrolled | -15.2377 | -9.49604 | 98125.8 |
| 43 | seed43_ridge_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | independent | -0.482321 | -0.302988 | 2619.1 |
| 43 | seed43_ridge_moderate | independent_count_reference | -0.482321 | -0.302988 | 2619.1 |
| 43 | seed43_ridge_moderate | joint | -0.482321 | -0.302988 | 2619.1 |
| 43 | seed43_ridge_moderate | joint_exact_count | -0.482321 | -0.302988 | 2619.1 |
| 43 | seed43_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | uncontrolled | -15.2377 | -9.49604 | 98125.8 |
| 43 | seed43_neural_cost_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | independent | -0.00587625 | 0.00635806 | 161.497 |
| 43 | seed43_neural_cost_conservative | independent_count_reference | -0.00587625 | 0.00635806 | 161.497 |
| 43 | seed43_neural_cost_conservative | joint | -0.00587625 | 0.00635806 | 161.497 |
| 43 | seed43_neural_cost_conservative | joint_exact_count | -0.00587625 | 0.00635806 | 161.497 |
| 43 | seed43_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | uncontrolled | -15.2377 | -9.49604 | 98125.8 |
| 43 | seed43_neural_cost_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | independent | -0.0233599 | -0.00278061 | 383.253 |
| 43 | seed43_neural_cost_moderate | independent_count_reference | -0.0233599 | -0.00278061 | 383.253 |
| 43 | seed43_neural_cost_moderate | joint | -0.0233599 | -0.00278061 | 383.253 |
| 43 | seed43_neural_cost_moderate | joint_exact_count | -0.0233599 | -0.00278061 | 383.253 |
| 43 | seed43_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | uncontrolled | -15.2377 | -9.49604 | 98125.8 |

## Matched Nonzero Intervention Counts

| Seed | Candidate | ADE joint minus independent | Scored agent queries | Matched nonzero scene queries | Solver/unmatched queries |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | 0 | 26700 | 905 | 0 |
| 17 | seed17_ridge_moderate | 0 | 27285 | 926 | 0 |
| 17 | seed17_neural_cost_conservative | 0 | 17377 | 583 | 0 |
| 17 | seed17_neural_cost_moderate | 0 | 22245 | 751 | 0 |
| 29 | seed29_ridge_conservative | 0 | 27385 | 933 | 0 |
| 29 | seed29_ridge_moderate | 0 | 28172 | 963 | 0 |
| 29 | seed29_neural_cost_conservative | 0 | 13724 | 464 | 0 |
| 29 | seed29_neural_cost_moderate | 0 | 19073 | 647 | 0 |
| 43 | seed43_ridge_conservative | 0 | 15709 | 538 | 0 |
| 43 | seed43_ridge_moderate | 0 | 19015 | 650 | 0 |
| 43 | seed43_neural_cost_conservative | 0 | 18711 | 632 | 0 |
| 43 | seed43_neural_cost_moderate | 0 | 22900 | 773 | 0 |

Negative differences favor joint selection on the same frozen candidates. Count matching applies to all past-supported agents before label access; scored-only coverage need not match after incomplete labels are filtered. Zero-count queries are reported in JSON but are not evidence of coordination.
One physical development site does not permit an informative scene-bootstrap CI. Seed SD describes training randomness only. Coordinates and time remain unverified dataset-local/native-frame units. No formal safety or deployment claim.
