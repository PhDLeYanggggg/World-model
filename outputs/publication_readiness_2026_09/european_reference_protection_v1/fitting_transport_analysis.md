# Reference Preservation and Harm Transport

Posthoc diagnosis only; no refitting or decision changes.
B uses fresh frozen-model replay; C uses checked frozen predictions. Cost/loss scales remain B-only.
Rows summarize eighteen dependent settings; output order is D_all, H_all, D_easy, H_easy.
A ratio below one means lower normalized squared error, not calibrated selected harm.

| Pair / role | Comparison | Population | Median component MSE ratios | Improved counts /18 |
|---|---|---|---|---|
| full / B | protected_vs_continued | population | 1.04423, 0.99549, 1.02799, 0.98957 | 0, 12, 0, 17 |
| full / B | protected_vs_continued | old_raw_selected | 1.04407, 0.99820, 1.01461, 0.99848 | 0, 14, 0, 13 |
| full / B | protected_vs_uniform | population | 1.00000, 0.91492, 1.00000, 0.94647 | 0, 16, 0, 18 |
| full / B | protected_vs_uniform | old_raw_selected | 1.00000, 0.95106, 1.00000, 0.99063 | 0, 16, 0, 14 |
| full / B | continued_vs_uniform | population | 0.95764, 0.91291, 0.97277, 0.95678 | 18, 16, 18, 17 |
| full / B | continued_vs_uniform | old_raw_selected | 0.95779, 0.95770, 0.98560, 0.99509 | 18, 15, 18, 13 |
| full / C | protected_vs_continued | population | 1.00795, 1.00350, 0.99744, 1.00226 | 6, 8, 12, 3 |
| full / C | protected_vs_continued | old_raw_selected | 0.99690, 1.00156, 1.00053, 1.00206 | 9, 8, 8, 2 |
| full / C | protected_vs_uniform | population | 1.00000, 1.01668, 1.00000, 1.00842 | 0, 5, 0, 4 |
| full / C | protected_vs_uniform | old_raw_selected | 1.00000, 1.01192, 1.00000, 1.00191 | 0, 4, 0, 4 |
| full / C | continued_vs_uniform | population | 0.99212, 1.01407, 1.00257, 1.00430 | 12, 5, 6, 4 |
| full / C | continued_vs_uniform | old_raw_selected | 1.00314, 1.00927, 0.99947, 1.00108 | 9, 3, 10, 7 |
| motion_only / B | protected_vs_continued | population | 1.04891, 0.98733, 1.02871, 0.98588 | 0, 17, 0, 17 |
| motion_only / B | protected_vs_continued | old_raw_selected | 1.04309, 0.99168, 1.02023, 0.99501 | 0, 15, 0, 14 |
| motion_only / B | protected_vs_uniform | population | 1.00000, 0.90403, 1.00000, 0.92587 | 0, 16, 0, 18 |
| motion_only / B | protected_vs_uniform | old_raw_selected | 1.00000, 0.93922, 1.00000, 0.97566 | 0, 17, 0, 16 |
| motion_only / B | continued_vs_uniform | population | 0.95337, 0.91747, 0.97209, 0.94805 | 18, 17, 18, 18 |
| motion_only / B | continued_vs_uniform | old_raw_selected | 0.95869, 0.95872, 0.98018, 0.98823 | 18, 16, 18, 16 |
| motion_only / C | protected_vs_continued | population | 1.01152, 0.99788, 1.00194, 1.00046 | 4, 12, 5, 8 |
| motion_only / C | protected_vs_continued | old_raw_selected | 0.99999, 1.00037, 1.01252, 1.00052 | 9, 8, 2, 7 |
| motion_only / C | protected_vs_uniform | population | 1.00000, 1.01249, 1.00000, 1.00401 | 0, 6, 0, 5 |
| motion_only / C | protected_vs_uniform | old_raw_selected | 1.00000, 1.01908, 1.00000, 1.00390 | 0, 4, 0, 4 |
| motion_only / C | continued_vs_uniform | population | 0.98861, 1.01140, 0.99807, 1.00273 | 14, 4, 13, 5 |
| motion_only / C | continued_vs_uniform | old_raw_selected | 1.00001, 1.01946, 0.98764, 1.00236 | 9, 4, 16, 3 |

## Same-Action Easy-Harm Coverage

Old raw-neural action masks are identical across all models. Values are median predicted/actual easy-harm mass.
| Pair / role | Uniform | Continued | Protected |
|---|---:|---:|---:|
| full / B | 0.61847 | 0.55643 | 0.54905 |
| full / C | 0.58180 | 0.53984 | 0.52150 |
| motion_only / B | 0.57164 | 0.48203 | 0.48452 |
| motion_only / C | 0.44430 | 0.35094 | 0.37630 |

## Boundaries

C is historically opened development, not independent confirmation. Independent calibration/confirmation remain closed.
Net easy preservation does not imply control of positive harm. Own-policy selected sets differ across arms.
Freezing an inaccurate reference estimate does not calibrate it. Population MSE is not selected-group risk.
Image pixels, annotation steps, detector-derived labels; no metric/seconds, physical-safety or foundation claim.
No deployment change. Stage5C and SMC remain off.
