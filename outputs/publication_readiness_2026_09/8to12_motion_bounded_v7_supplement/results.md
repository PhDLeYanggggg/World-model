# Fixed-Forecast Supplement

Development-only, all three seeds and every frozen cost-head/policy combination. No additional model selection.
Raw50 is the exact native-grid prefix of the original 12-step prediction. Its scale is not refitted.
Primary full-horizon evaluation remains authoritative; this table cannot rescue a failed primary result.

| Seed | Candidate | Arm | Raw50 ADE gain % | Raw50 FDE gain % | Full-path easy degradation % (prefix ADE) |
| --- | --- | --- | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | independent | 0.00475654 | 0.00535017 | 4.37754 |
| 17 | seed17_ridge_conservative | independent_count_reference | 0.00475654 | 0.00535017 | 4.37754 |
| 17 | seed17_ridge_conservative | joint | 0.00475654 | 0.00535017 | 4.37754 |
| 17 | seed17_ridge_conservative | joint_exact_count | 0.00475654 | 0.00535017 | 4.37754 |
| 17 | seed17_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_conservative | uncontrolled | 0.0421662 | 0.0604488 | 29.1347 |
| 17 | seed17_ridge_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | independent | 0.0289902 | 0.0360091 | 8.48974 |
| 17 | seed17_ridge_moderate | independent_count_reference | 0.0289902 | 0.0360091 | 8.48974 |
| 17 | seed17_ridge_moderate | joint | 0.0289295 | 0.0359624 | 8.48974 |
| 17 | seed17_ridge_moderate | joint_exact_count | 0.0289336 | 0.0359674 | 8.48974 |
| 17 | seed17_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_ridge_moderate | uncontrolled | 0.0421662 | 0.0604488 | 29.1347 |
| 17 | seed17_neural_cost_conservative | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | independent | 0.000222508 | 0.000236241 | 0.0284424 |
| 17 | seed17_neural_cost_conservative | independent_count_reference | 0.000222508 | 0.000236241 | 0.0284424 |
| 17 | seed17_neural_cost_conservative | joint | 0.000222508 | 0.000236241 | 0.0284424 |
| 17 | seed17_neural_cost_conservative | joint_exact_count | 0.000222508 | 0.000236241 | 0.0284424 |
| 17 | seed17_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_conservative | uncontrolled | 0.0421662 | 0.0604488 | 29.1347 |
| 17 | seed17_neural_cost_moderate | floor | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | independent | 0.00169473 | 0.00189686 | 0.785892 |
| 17 | seed17_neural_cost_moderate | independent_count_reference | 0.00169473 | 0.00189686 | 0.785892 |
| 17 | seed17_neural_cost_moderate | joint | 0.00169473 | 0.00189686 | 0.785892 |
| 17 | seed17_neural_cost_moderate | joint_exact_count | 0.00169473 | 0.00189686 | 0.785892 |
| 17 | seed17_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 17 | seed17_neural_cost_moderate | uncontrolled | 0.0421662 | 0.0604488 | 29.1347 |
| 29 | seed29_ridge_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | independent | 0.00805422 | 0.00874687 | 1.09094 |
| 29 | seed29_ridge_conservative | independent_count_reference | 0.00805422 | 0.00874687 | 1.09094 |
| 29 | seed29_ridge_conservative | joint | 0.00805422 | 0.00874687 | 1.09094 |
| 29 | seed29_ridge_conservative | joint_exact_count | 0.00805422 | 0.00874687 | 1.09094 |
| 29 | seed29_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_conservative | uncontrolled | -0.0469543 | -0.00669968 | 172.847 |
| 29 | seed29_ridge_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | independent | 0.0409398 | 0.0502188 | 10.3891 |
| 29 | seed29_ridge_moderate | independent_count_reference | 0.0409398 | 0.0502188 | 10.3891 |
| 29 | seed29_ridge_moderate | joint | 0.0408897 | 0.0501596 | 10.3891 |
| 29 | seed29_ridge_moderate | joint_exact_count | 0.0409121 | 0.0501839 | 10.3891 |
| 29 | seed29_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_ridge_moderate | uncontrolled | -0.0469543 | -0.00669968 | 172.847 |
| 29 | seed29_neural_cost_conservative | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | independent | 0.000682042 | 0.00133243 | 0.911322 |
| 29 | seed29_neural_cost_conservative | independent_count_reference | 0.000682042 | 0.00133243 | 0.911322 |
| 29 | seed29_neural_cost_conservative | joint | 0.000682042 | 0.00133243 | 0.911322 |
| 29 | seed29_neural_cost_conservative | joint_exact_count | 0.000682042 | 0.00133243 | 0.911322 |
| 29 | seed29_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_conservative | uncontrolled | -0.0469543 | -0.00669968 | 172.847 |
| 29 | seed29_neural_cost_moderate | floor | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | independent | 0.0018924 | 0.00394486 | 2.83526 |
| 29 | seed29_neural_cost_moderate | independent_count_reference | 0.0018924 | 0.00394486 | 2.83526 |
| 29 | seed29_neural_cost_moderate | joint | 0.00191251 | 0.00396491 | 2.83526 |
| 29 | seed29_neural_cost_moderate | joint_exact_count | 0.0018924 | 0.00394486 | 2.83526 |
| 29 | seed29_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 29 | seed29_neural_cost_moderate | uncontrolled | -0.0469543 | -0.00669968 | 172.847 |
| 43 | seed43_ridge_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | independent | 0.0114505 | 0.0119589 | 4.61991 |
| 43 | seed43_ridge_conservative | independent_count_reference | 0.0114505 | 0.0119589 | 4.61991 |
| 43 | seed43_ridge_conservative | joint | 0.0114251 | 0.0119339 | 4.61991 |
| 43 | seed43_ridge_conservative | joint_exact_count | 0.0114505 | 0.0119589 | 4.61991 |
| 43 | seed43_ridge_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_conservative | uncontrolled | 0.00559087 | 0.0232547 | 91.5761 |
| 43 | seed43_ridge_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | independent | 0.0315898 | 0.0412675 | 13.5363 |
| 43 | seed43_ridge_moderate | independent_count_reference | 0.0315898 | 0.0412675 | 13.5363 |
| 43 | seed43_ridge_moderate | joint | 0.0315164 | 0.0412131 | 13.3403 |
| 43 | seed43_ridge_moderate | joint_exact_count | 0.0315093 | 0.0412059 | 13.3403 |
| 43 | seed43_ridge_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_ridge_moderate | uncontrolled | 0.00559087 | 0.0232547 | 91.5761 |
| 43 | seed43_neural_cost_conservative | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | independent | 0.000312674 | 0.000370344 | 0.133925 |
| 43 | seed43_neural_cost_conservative | independent_count_reference | 0.000312674 | 0.000370344 | 0.133925 |
| 43 | seed43_neural_cost_conservative | joint | 0.000312674 | 0.000370344 | 0.133925 |
| 43 | seed43_neural_cost_conservative | joint_exact_count | 0.000312674 | 0.000370344 | 0.133925 |
| 43 | seed43_neural_cost_conservative | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_conservative | uncontrolled | 0.00559087 | 0.0232547 | 91.5761 |
| 43 | seed43_neural_cost_moderate | floor | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | independent | 0.0011077 | 0.00171036 | 1.62834 |
| 43 | seed43_neural_cost_moderate | independent_count_reference | 0.0011077 | 0.00171036 | 1.62834 |
| 43 | seed43_neural_cost_moderate | joint | 0.0011077 | 0.00171036 | 1.62834 |
| 43 | seed43_neural_cost_moderate | joint_exact_count | 0.0011077 | 0.00171036 | 1.62834 |
| 43 | seed43_neural_cost_moderate | scene_uniform | 0 | 0 | -0 |
| 43 | seed43_neural_cost_moderate | uncontrolled | 0.00559087 | 0.0232547 | 91.5761 |

## Matched Nonzero Intervention Counts

| Seed | Candidate | ADE joint minus independent | Scored agent queries | Matched nonzero scene queries | Solver/unmatched queries |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | seed17_ridge_conservative | 0 | 22894 | 775 | 0 |
| 17 | seed17_ridge_moderate | 1.16083e-07 | 28324 | 970 | 0 |
| 17 | seed17_neural_cost_conservative | 0 | 8345 | 289 | 0 |
| 17 | seed17_neural_cost_moderate | 0 | 16390 | 554 | 0 |
| 29 | seed29_ridge_conservative | 0 | 21208 | 718 | 0 |
| 29 | seed29_ridge_moderate | 3.28304e-06 | 28307 | 969 | 0 |
| 29 | seed29_neural_cost_conservative | 0 | 9703 | 320 | 0 |
| 29 | seed29_neural_cost_moderate | 0 | 17354 | 582 | 0 |
| 43 | seed43_ridge_conservative | 0 | 26289 | 902 | 0 |
| 43 | seed43_ridge_moderate | 4.82345e-07 | 28324 | 970 | 0 |
| 43 | seed43_neural_cost_conservative | 0 | 5299 | 176 | 0 |
| 43 | seed43_neural_cost_moderate | 0 | 14816 | 498 | 0 |

Negative differences favor joint selection on the same frozen candidates. Count matching applies to all past-supported agents before label access; scored-only coverage need not match after incomplete labels are filtered. Zero-count queries are reported in JSON but are not evidence of coordination.
One physical development site does not permit an informative scene-bootstrap CI. Seed SD describes training randomness only. Coordinates and time remain unverified dataset-local/native-frame units. No formal safety or deployment claim.
