# Fitting Losses

Four bounded moment targets; draw-weighted fitting MSE, not validation loss or forecast ADE.
No early stopping or outer-outcome selection. All 36 fits use 128 trees.

| View | Action | First MSE (16 trees) | Final MSE (128 trees) | Fit seconds |
|---|---|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | 0.06676208 | 0.06647733 | 25.579 |
| coupa_seed17 | transformer | 0.06244198 | 0.06185101 | 30.196 |
| coupa_seed17 | eqmotion | 0.06289304 | 0.06275170 | 32.562 |
| coupa_seed29 | damped_velocity_005 | 0.06678735 | 0.06643972 | 26.162 |
| coupa_seed29 | transformer | 0.06365008 | 0.06320279 | 30.193 |
| coupa_seed29 | eqmotion | 0.06310798 | 0.06294582 | 33.396 |
| coupa_seed43 | damped_velocity_005 | 0.06591546 | 0.06614140 | 24.828 |
| coupa_seed43 | transformer | 0.06426851 | 0.06408043 | 29.663 |
| coupa_seed43 | eqmotion | 0.06270607 | 0.06268403 | 32.362 |
| deathCircle_seed17 | damped_velocity_005 | 0.05718580 | 0.05678819 | 22.661 |
| deathCircle_seed17 | transformer | 0.05777692 | 0.05746356 | 29.033 |
| deathCircle_seed17 | eqmotion | 0.05790205 | 0.05787427 | 30.893 |
| deathCircle_seed29 | damped_velocity_005 | 0.05703414 | 0.05650164 | 22.663 |
| deathCircle_seed29 | transformer | 0.05711730 | 0.05686593 | 27.183 |
| deathCircle_seed29 | eqmotion | 0.05773479 | 0.05763529 | 29.181 |
| deathCircle_seed43 | damped_velocity_005 | 0.05612803 | 0.05623902 | 21.568 |
| deathCircle_seed43 | transformer | 0.05857471 | 0.05807051 | 27.975 |
| deathCircle_seed43 | eqmotion | 0.05783564 | 0.05768430 | 29.450 |
| gates_seed17 | damped_velocity_005 | 0.05819311 | 0.05800804 | 24.474 |
| gates_seed17 | transformer | 0.05898107 | 0.05882076 | 29.661 |
| gates_seed17 | eqmotion | 0.06064905 | 0.06039315 | 32.775 |
| gates_seed29 | damped_velocity_005 | 0.05808853 | 0.05775138 | 24.410 |
| gates_seed29 | transformer | 0.05821017 | 0.05816729 | 29.290 |
| gates_seed29 | eqmotion | 0.06078284 | 0.06070143 | 33.509 |
| gates_seed43 | damped_velocity_005 | 0.05777149 | 0.05772187 | 24.410 |
| gates_seed43 | transformer | 0.05907893 | 0.05869830 | 29.835 |
| gates_seed43 | eqmotion | 0.06080836 | 0.06058957 | 33.547 |
| hyang_seed17 | damped_velocity_005 | 0.05236396 | 0.05154922 | 9.390 |
| hyang_seed17 | transformer | 0.05545184 | 0.05516371 | 12.748 |
| hyang_seed17 | eqmotion | 0.05691707 | 0.05674488 | 14.390 |
| hyang_seed29 | damped_velocity_005 | 0.05185378 | 0.05134643 | 9.427 |
| hyang_seed29 | transformer | 0.05514969 | 0.05491535 | 12.653 |
| hyang_seed29 | eqmotion | 0.05717191 | 0.05676184 | 14.523 |
| hyang_seed43 | damped_velocity_005 | 0.05102697 | 0.05119244 | 9.351 |
| hyang_seed43 | transformer | 0.05606677 | 0.05581434 | 12.454 |
| hyang_seed43 | eqmotion | 0.05688719 | 0.05674882 | 14.416 |

Summed fitting-loop time: 876.812 seconds, not total wall time. Per-target traces are in analysis.json.
