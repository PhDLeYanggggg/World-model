# Fresh Leaf Readout Fits and Support

36 new source-only leaf-frequency fits on 36 cached_verified forest partitions. No new forest or neural forecast training.
Weighted fitting Brier is an in-source fitting diagnostic, not independent calibration.

| View | Action | Effective unique rows | Zero-reference rows | Moving zero-reference rows | Zero-reference weighted draws | Fitting Brier | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | 93261 | 2 | 2 | 6 | 0.0000097 | 6.270 |
| coupa_seed17 | transformer | 105556 | 2209 | 2 | 17304 | 0.0126823 | 7.040 |
| coupa_seed17 | eqmotion | 116450 | 8931 | 2 | 69263 | 0.0283785 | 7.367 |
| coupa_seed29 | damped_velocity_005 | 93131 | 2 | 2 | 7 | 0.0000113 | 6.341 |
| coupa_seed29 | transformer | 105430 | 2200 | 2 | 17230 | 0.0127459 | 7.145 |
| coupa_seed29 | eqmotion | 116325 | 8915 | 2 | 69628 | 0.0284195 | 7.482 |
| coupa_seed43 | damped_velocity_005 | 93247 | 2 | 2 | 8 | 0.0000130 | 6.387 |
| coupa_seed43 | transformer | 105550 | 2210 | 2 | 17172 | 0.0126016 | 7.119 |
| coupa_seed43 | eqmotion | 116465 | 8928 | 2 | 68969 | 0.0282889 | 7.575 |
| deathCircle_seed17 | damped_velocity_005 | 85299 | 7 | 7 | 56 | 0.0000996 | 5.948 |
| deathCircle_seed17 | transformer | 101804 | 2051 | 7 | 16626 | 0.0161326 | 6.844 |
| deathCircle_seed17 | eqmotion | 113875 | 8392 | 7 | 66532 | 0.0347343 | 7.363 |
| deathCircle_seed29 | damped_velocity_005 | 85167 | 7 | 7 | 63 | 0.0001116 | 5.898 |
| deathCircle_seed29 | transformer | 101675 | 2042 | 7 | 16545 | 0.0162464 | 6.741 |
| deathCircle_seed29 | eqmotion | 113746 | 8376 | 7 | 66864 | 0.0350252 | 7.333 |
| deathCircle_seed43 | damped_velocity_005 | 85285 | 7 | 7 | 66 | 0.0001173 | 6.029 |
| deathCircle_seed43 | transformer | 101797 | 2052 | 7 | 16519 | 0.0160167 | 6.776 |
| deathCircle_seed43 | eqmotion | 113888 | 8389 | 7 | 66500 | 0.0348445 | 7.307 |
| gates_seed17 | damped_velocity_005 | 94174 | 7 | 7 | 56 | 0.0000987 | 6.709 |
| gates_seed17 | transformer | 112522 | 2645 | 7 | 20086 | 0.0190123 | 7.496 |
| gates_seed17 | eqmotion | 125985 | 9811 | 7 | 68630 | 0.0395086 | 7.861 |
| gates_seed29 | damped_velocity_005 | 94042 | 7 | 7 | 63 | 0.0001100 | 6.659 |
| gates_seed29 | transformer | 112393 | 2636 | 7 | 19841 | 0.0190054 | 7.586 |
| gates_seed29 | eqmotion | 125856 | 9795 | 7 | 68529 | 0.0392276 | 7.756 |
| gates_seed43 | damped_velocity_005 | 94160 | 7 | 7 | 66 | 0.0001150 | 6.662 |
| gates_seed43 | transformer | 112514 | 2646 | 7 | 19850 | 0.0191559 | 7.358 |
| gates_seed43 | eqmotion | 125996 | 9807 | 7 | 68206 | 0.0392148 | 7.933 |
| hyang_seed17 | damped_velocity_005 | 46433 | 5 | 5 | 50 | 0.0000899 | 3.609 |
| hyang_seed17 | transformer | 58540 | 1999 | 5 | 22076 | 0.0192093 | 4.115 |
| hyang_seed17 | eqmotion | 67486 | 6994 | 5 | 79472 | 0.0397960 | 4.418 |
| hyang_seed29 | damped_velocity_005 | 46433 | 5 | 5 | 56 | 0.0001003 | 3.664 |
| hyang_seed29 | transformer | 58540 | 1999 | 5 | 21855 | 0.0191910 | 4.142 |
| hyang_seed29 | eqmotion | 67486 | 6994 | 5 | 79331 | 0.0397736 | 4.445 |
| hyang_seed43 | damped_velocity_005 | 46433 | 5 | 5 | 58 | 0.0001043 | 3.688 |
| hyang_seed43 | transformer | 58539 | 1999 | 5 | 21836 | 0.0191671 | 3.976 |
| hyang_seed43 | eqmotion | 67484 | 6993 | 5 | 79106 | 0.0396257 | 4.394 |

## Held-Source Event Diagnostic

All held sites remain design-exposed. No probability threshold was selected from these numbers.

| View | Action | Complete labels | Zero-CV rows | Moving zero-CV | Moving zero-CV admitted | Complete-label Brier |
|---|---|---:|---:|---:|---:|---:|
| coupa_seed17 | damped_velocity_005 | 24834 | 2463 | 5 | 5 | 0.0991785 |
| deathCircle_seed17 | damped_velocity_005 | 27394 | 2984 | 0 | 0 | 0.1089288 |
| gates_seed17 | damped_velocity_005 | 15281 | 1565 | 0 | 0 | 0.1024151 |
| hyang_seed17 | damped_velocity_005 | 76409 | 4554 | 2 | 2 | 0.0596003 |
| coupa_seed29 | damped_velocity_005 | 24834 | 2463 | 5 | 4 | 0.0991785 |
| deathCircle_seed29 | damped_velocity_005 | 27394 | 2984 | 0 | 0 | 0.1089288 |
| gates_seed29 | damped_velocity_005 | 15281 | 1565 | 0 | 0 | 0.1024150 |
| hyang_seed29 | damped_velocity_005 | 76409 | 4554 | 2 | 2 | 0.0596003 |
| coupa_seed43 | damped_velocity_005 | 24834 | 2463 | 5 | 5 | 0.0991785 |
| deathCircle_seed43 | damped_velocity_005 | 27394 | 2984 | 0 | 0 | 0.1089288 |
| gates_seed43 | damped_velocity_005 | 15281 | 1565 | 0 | 0 | 0.1024151 |
| hyang_seed43 | damped_velocity_005 | 76409 | 4554 | 2 | 2 | 0.0596004 |
| coupa_seed17 | transformer | 24834 | 2463 | 5 | 0 | 0.0677553 |
| deathCircle_seed17 | transformer | 27394 | 2984 | 0 | 0 | 0.0775149 |
| gates_seed17 | transformer | 15281 | 1565 | 0 | 0 | 0.0625351 |
| hyang_seed17 | transformer | 76409 | 4554 | 2 | 2 | 0.0412517 |
| coupa_seed29 | transformer | 24834 | 2463 | 5 | 5 | 0.0687267 |
| deathCircle_seed29 | transformer | 27394 | 2984 | 0 | 0 | 0.0730409 |
| gates_seed29 | transformer | 15281 | 1565 | 0 | 0 | 0.0561895 |
| hyang_seed29 | transformer | 76409 | 4554 | 2 | 2 | 0.0381609 |
| coupa_seed43 | transformer | 24834 | 2463 | 5 | 0 | 0.0688576 |
| deathCircle_seed43 | transformer | 27394 | 2984 | 0 | 0 | 0.0789309 |
| gates_seed43 | transformer | 15281 | 1565 | 0 | 0 | 0.0495765 |
| hyang_seed43 | transformer | 76409 | 4554 | 2 | 2 | 0.0422729 |
| coupa_seed17 | eqmotion | 24834 | 2463 | 5 | 3 | 0.0806796 |
| deathCircle_seed17 | eqmotion | 27394 | 2984 | 0 | 0 | 0.0478457 |
| gates_seed17 | eqmotion | 15281 | 1565 | 0 | 0 | 0.0366188 |
| hyang_seed17 | eqmotion | 76409 | 4554 | 2 | 2 | 0.0320862 |
| coupa_seed29 | eqmotion | 24834 | 2463 | 5 | 3 | 0.0821800 |
| deathCircle_seed29 | eqmotion | 27394 | 2984 | 0 | 0 | 0.0480345 |
| gates_seed29 | eqmotion | 15281 | 1565 | 0 | 0 | 0.0370569 |
| hyang_seed29 | eqmotion | 76409 | 4554 | 2 | 2 | 0.0320723 |
| coupa_seed43 | eqmotion | 24834 | 2463 | 5 | 5 | 0.0815607 |
| deathCircle_seed43 | eqmotion | 27394 | 2984 | 0 | 0 | 0.0482957 |
| gates_seed43 | eqmotion | 15281 | 1565 | 0 | 0 | 0.0365095 |
| hyang_seed43 | eqmotion | 76409 | 4554 | 2 | 2 | 0.0319407 |
