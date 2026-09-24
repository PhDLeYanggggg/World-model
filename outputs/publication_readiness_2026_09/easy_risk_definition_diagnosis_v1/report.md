# Easy-Risk Definition and Reliability Diagnosis

Fresh post-decision development analysis on frozen outputs, not calibration or threshold selection.
Only complete futures contribute observed costs. Incomplete selections remain separately counted.
Positive harm does not subtract improvements; net harm does. A negative net ratio means improvement.
No policy changes or new external readout. Four exposed SDD sites, obs8/pred12 annotation pixels.

| View / predictor | Strict predicted positive selected ratio % | Observed positive selected ratio % | Observed net selected ratio % | Predicted / observed harm | Benefit / positive harm % |
|---|---:|---:|---:|---:|---:|
| coupa_seed17 / damped_velocity_005 | 15.772 | 11.006 | -10.075 | 1.655 | 191.534 |
| coupa_seed17 / transformer | 12.001 | 7.188 | -38.428 | 1.988 | 634.584 |
| coupa_seed17 / eqmotion | 16.262 | 4.187 | -57.346 | 4.775 | 1469.479 |
| coupa_seed29 / damped_velocity_005 | 14.121 | 9.999 | -11.101 | 1.637 | 211.026 |
| coupa_seed29 / transformer | 13.364 | 7.560 | -41.892 | 2.387 | 654.125 |
| coupa_seed29 / eqmotion | 16.126 | 8.341 | -49.210 | 2.246 | 690.007 |
| coupa_seed43 / damped_velocity_005 | 14.895 | 11.421 | -9.791 | 1.487 | 185.725 |
| coupa_seed43 / transformer | 19.247 | 18.902 | -15.318 | 1.474 | 181.039 |
| coupa_seed43 / eqmotion | 15.773 | 7.388 | -51.768 | 2.564 | 800.683 |
| deathCircle_seed17 / damped_velocity_005 | 109.894 | 44.971 | 27.318 | 3.070 | 39.254 |
| deathCircle_seed17 / transformer | 42.858 | 26.521 | 4.685 | 2.397 | 82.336 |
| deathCircle_seed17 / eqmotion | 59.704 | 53.540 | 14.027 | 1.066 | 73.800 |
| deathCircle_seed29 / damped_velocity_005 | 116.020 | 48.446 | 31.227 | 3.034 | 35.542 |
| deathCircle_seed29 / transformer | 73.009 | 38.357 | 17.057 | 3.362 | 55.530 |
| deathCircle_seed29 / eqmotion | 72.297 | 47.770 | 11.905 | 1.388 | 75.078 |
| deathCircle_seed43 / damped_velocity_005 | 113.667 | 48.135 | 31.063 | 2.806 | 35.467 |
| deathCircle_seed43 / transformer | 65.288 | 34.736 | 10.642 | 2.758 | 69.364 |
| deathCircle_seed43 / eqmotion | 82.462 | 55.799 | 19.321 | 1.474 | 65.374 |
| gates_seed17 / damped_velocity_005 | 65.173 | 59.681 | 37.554 | 1.185 | 37.075 |
| gates_seed17 / transformer | 55.652 | 35.759 | -22.433 | 2.279 | 162.733 |
| gates_seed17 / eqmotion | 39.688 | 7.965 | -45.883 | 7.921 | 676.058 |
| gates_seed29 / damped_velocity_005 | 61.223 | 50.251 | 27.098 | 1.304 | 46.075 |
| gates_seed29 / transformer | 40.950 | 22.211 | -32.451 | 2.241 | 246.107 |
| gates_seed29 / eqmotion | 33.618 | 4.268 | -57.498 | 9.929 | 1447.317 |
| gates_seed43 / damped_velocity_005 | 59.494 | 54.831 | 32.547 | 1.145 | 40.642 |
| gates_seed43 / transformer | 75.153 | 45.496 | -6.846 | 2.208 | 115.047 |
| gates_seed43 / eqmotion | 40.750 | 28.136 | -29.296 | 1.421 | 204.122 |
| hyang_seed17 / damped_velocity_005 | 54.507 | 42.818 | 25.306 | 1.071 | 40.898 |
| hyang_seed17 / transformer | 43.048 | 31.656 | -15.599 | 1.144 | 149.276 |
| hyang_seed17 / eqmotion | 30.254 | 43.319 | -5.946 | 0.591 | 113.726 |
| hyang_seed29 / damped_velocity_005 | 58.078 | 46.069 | 28.868 | 1.059 | 37.337 |
| hyang_seed29 / transformer | 37.418 | 32.924 | -15.541 | 0.963 | 147.204 |
| hyang_seed29 / eqmotion | 30.470 | 43.124 | -6.222 | 0.611 | 114.428 |
| hyang_seed43 / damped_velocity_005 | 58.618 | 47.563 | 30.341 | 1.047 | 36.207 |
| hyang_seed43 / transformer | 68.131 | 52.017 | 5.143 | 1.147 | 90.112 |
| hyang_seed43 / eqmotion | 26.420 | 40.035 | -10.989 | 0.516 | 127.449 |

## Interpretation Boundaries

Scene-balanced training draws are uniform within supported complete rows, not hard-label oversampling.
Complete-label selection and cross-scene shift remain; this cannot identify unconditional risk on missing labels.
Selected-set ratios in this table do not equal population easy degradation. Both denominators are in the CSV.
Ratios pooled across sites would mix pixel scales and are not reported. Zero denominators remain undefined.
Use this diagnostic to register a falsifiable target comparison, not to tune a threshold or promote a policy.
