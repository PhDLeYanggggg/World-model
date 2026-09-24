# Training Loss and Runtime

All 72 fresh six-output forests use the same fixed budget.
Weighted source fitting MSE is not validation loss or independent predictive evidence.

| View | Action | Features | Trees | Fitting mean MSE | Seconds | Unknown draws |
|---|---|---|---:|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | native | 128 | 0.0593674 | 23.418 | 0 |
| coupa_seed17 | damped_velocity_005 | dimensionless | 128 | 0.0626195 | 24.350 | 0 |
| coupa_seed17 | transformer | native | 128 | 0.0610580 | 26.803 | 0 |
| coupa_seed17 | transformer | dimensionless | 128 | 0.0627977 | 26.846 | 0 |
| coupa_seed17 | eqmotion | native | 128 | 0.0621381 | 29.506 | 0 |
| coupa_seed17 | eqmotion | dimensionless | 128 | 0.0642384 | 29.731 | 0 |
| coupa_seed29 | damped_velocity_005 | native | 128 | 0.0594258 | 23.954 | 0 |
| coupa_seed29 | damped_velocity_005 | dimensionless | 128 | 0.0625117 | 24.208 | 0 |
| coupa_seed29 | transformer | native | 128 | 0.0619729 | 26.434 | 0 |
| coupa_seed29 | transformer | dimensionless | 128 | 0.0640538 | 26.537 | 0 |
| coupa_seed29 | eqmotion | native | 128 | 0.0619369 | 30.045 | 0 |
| coupa_seed29 | eqmotion | dimensionless | 128 | 0.0642114 | 29.686 | 0 |
| coupa_seed43 | damped_velocity_005 | native | 128 | 0.0592865 | 23.863 | 0 |
| coupa_seed43 | damped_velocity_005 | dimensionless | 128 | 0.0623106 | 24.408 | 0 |
| coupa_seed43 | transformer | native | 128 | 0.0612320 | 27.133 | 0 |
| coupa_seed43 | transformer | dimensionless | 128 | 0.0631651 | 27.185 | 0 |
| coupa_seed43 | eqmotion | native | 128 | 0.0622343 | 29.677 | 0 |
| coupa_seed43 | eqmotion | dimensionless | 128 | 0.0644433 | 29.631 | 0 |
| deathCircle_seed17 | damped_velocity_005 | native | 128 | 0.0509345 | 21.121 | 0 |
| deathCircle_seed17 | damped_velocity_005 | dimensionless | 128 | 0.0533571 | 22.482 | 0 |
| deathCircle_seed17 | transformer | native | 128 | 0.0558542 | 26.216 | 0 |
| deathCircle_seed17 | transformer | dimensionless | 128 | 0.0573295 | 26.564 | 0 |
| deathCircle_seed17 | eqmotion | native | 128 | 0.0577515 | 28.618 | 0 |
| deathCircle_seed17 | eqmotion | dimensionless | 128 | 0.0597488 | 28.675 | 0 |
| deathCircle_seed29 | damped_velocity_005 | native | 128 | 0.0506061 | 21.491 | 0 |
| deathCircle_seed29 | damped_velocity_005 | dimensionless | 128 | 0.0530651 | 21.814 | 0 |
| deathCircle_seed29 | transformer | native | 128 | 0.0552882 | 26.236 | 0 |
| deathCircle_seed29 | transformer | dimensionless | 128 | 0.0569630 | 26.408 | 0 |
| deathCircle_seed29 | eqmotion | native | 128 | 0.0573555 | 28.351 | 0 |
| deathCircle_seed29 | eqmotion | dimensionless | 128 | 0.0593884 | 28.577 | 0 |
| deathCircle_seed43 | damped_velocity_005 | native | 128 | 0.0504814 | 21.435 | 0 |
| deathCircle_seed43 | damped_velocity_005 | dimensionless | 128 | 0.0529785 | 22.433 | 0 |
| deathCircle_seed43 | transformer | native | 128 | 0.0554495 | 26.426 | 0 |
| deathCircle_seed43 | transformer | dimensionless | 128 | 0.0571849 | 27.086 | 0 |
| deathCircle_seed43 | eqmotion | native | 128 | 0.0581706 | 28.408 | 0 |
| deathCircle_seed43 | eqmotion | dimensionless | 128 | 0.0599546 | 28.302 | 0 |
| gates_seed17 | damped_velocity_005 | native | 128 | 0.0534828 | 23.861 | 0 |
| gates_seed17 | damped_velocity_005 | dimensionless | 128 | 0.0564268 | 25.141 | 0 |
| gates_seed17 | transformer | native | 128 | 0.0579120 | 28.341 | 0 |
| gates_seed17 | transformer | dimensionless | 128 | 0.0595612 | 28.232 | 0 |
| gates_seed17 | eqmotion | native | 128 | 0.0594488 | 31.872 | 0 |
| gates_seed17 | eqmotion | dimensionless | 128 | 0.0615938 | 32.089 | 0 |
| gates_seed29 | damped_velocity_005 | native | 128 | 0.0533826 | 24.144 | 0 |
| gates_seed29 | damped_velocity_005 | dimensionless | 128 | 0.0560947 | 24.691 | 0 |
| gates_seed29 | transformer | native | 128 | 0.0574623 | 28.543 | 0 |
| gates_seed29 | transformer | dimensionless | 128 | 0.0589016 | 28.581 | 0 |
| gates_seed29 | eqmotion | native | 128 | 0.0597169 | 32.056 | 0 |
| gates_seed29 | eqmotion | dimensionless | 128 | 0.0618884 | 32.423 | 0 |
| gates_seed43 | damped_velocity_005 | native | 128 | 0.0532371 | 24.475 | 0 |
| gates_seed43 | damped_velocity_005 | dimensionless | 128 | 0.0560837 | 24.405 | 0 |
| gates_seed43 | transformer | native | 128 | 0.0580092 | 28.570 | 0 |
| gates_seed43 | transformer | dimensionless | 128 | 0.0594846 | 29.013 | 0 |
| gates_seed43 | eqmotion | native | 128 | 0.0597482 | 32.718 | 0 |
| gates_seed43 | eqmotion | dimensionless | 128 | 0.0617975 | 32.547 | 0 |
| hyang_seed17 | damped_velocity_005 | native | 128 | 0.0494883 | 9.263 | 0 |
| hyang_seed17 | damped_velocity_005 | dimensionless | 128 | 0.0519035 | 9.735 | 0 |
| hyang_seed17 | transformer | native | 128 | 0.0551622 | 12.097 | 0 |
| hyang_seed17 | transformer | dimensionless | 128 | 0.0568126 | 12.013 | 0 |
| hyang_seed17 | eqmotion | native | 128 | 0.0561139 | 13.644 | 0 |
| hyang_seed17 | eqmotion | dimensionless | 128 | 0.0577603 | 13.831 | 0 |
| hyang_seed29 | damped_velocity_005 | native | 128 | 0.0493845 | 9.352 | 0 |
| hyang_seed29 | damped_velocity_005 | dimensionless | 128 | 0.0516200 | 9.418 | 0 |
| hyang_seed29 | transformer | native | 128 | 0.0549925 | 12.065 | 0 |
| hyang_seed29 | transformer | dimensionless | 128 | 0.0565256 | 11.905 | 0 |
| hyang_seed29 | eqmotion | native | 128 | 0.0559153 | 13.787 | 0 |
| hyang_seed29 | eqmotion | dimensionless | 128 | 0.0575866 | 13.875 | 0 |
| hyang_seed43 | damped_velocity_005 | native | 128 | 0.0494757 | 9.369 | 0 |
| hyang_seed43 | damped_velocity_005 | dimensionless | 128 | 0.0518490 | 9.748 | 0 |
| hyang_seed43 | transformer | native | 128 | 0.0551904 | 11.725 | 0 |
| hyang_seed43 | transformer | dimensionless | 128 | 0.0568060 | 11.715 | 0 |
| hyang_seed43 | eqmotion | native | 128 | 0.0563125 | 13.822 | 0 |
| hyang_seed43 | eqmotion | dimensionless | 128 | 0.0580052 | 13.640 | 0 |

Total fitting-loop seconds: 1666.7672130800784.
This excludes data preparation, provenance hashing, decision solving, evaluation and replay.
Loss traces and held-source six-target MSE are retained in analysis.json.
