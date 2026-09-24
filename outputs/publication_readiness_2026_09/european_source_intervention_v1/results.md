# Nested Intervention Results

Fresh nested source-cost fitting and source-excluded controls; cached metric reproduction and fresh cost-checkpoint replay pass.
All results remain source-only method development. The predicted positive-harm cap does not certify the actual easy or exact-zero-reference limits.

## Full Indexed Source Cohort

| Seed / cost head | Rule | ADE gain vs strongest (%) | Conditional 95% CI | FDE gain vs strongest (%) | Easy gain vs CV (%) | Worst easy-locality gain (%) | Zero-CV harmed rows | Switch rate |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 17_ridge | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 17_ridge | neural | 4.3385 | [1.5540, 7.3289] | 5.2765 | -13.7319 | -75.9522 | 4 | 1.0000 |
| 17_ridge | pointwise | 4.0131 | [1.4111, 6.6549] | 5.0708 | -13.7580 | -81.4090 | 2 | 0.5430 |
| 17_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 17_neural_underharm4 | neural | 4.3385 | [1.5540, 7.3289] | 5.2765 | -13.7319 | -75.9522 | 4 | 1.0000 |
| 17_neural_underharm4 | pointwise | 3.2851 | [1.3962, 5.3459] | 4.3176 | -13.3839 | -87.2887 | 2 | 0.2543 |
| 29_ridge | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 29_ridge | neural | 3.8309 | [0.8333, 6.8765] | 5.1391 | -13.6479 | -75.6037 | 4 | 1.0000 |
| 29_ridge | pointwise | 3.9054 | [1.2150, 6.6891] | 5.1931 | -13.5547 | -80.3491 | 3 | 0.5518 |
| 29_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 29_neural_underharm4 | neural | 3.8309 | [0.8333, 6.8765] | 5.1391 | -13.6479 | -75.6037 | 4 | 1.0000 |
| 29_neural_underharm4 | pointwise | 3.3657 | [1.3928, 5.5963] | 4.5792 | -12.8069 | -80.7570 | 0 | 0.0716 |
| 43_ridge | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 43_ridge | neural | 4.1711 | [1.6569, 6.7234] | 5.2503 | -14.3936 | -81.0359 | 4 | 1.0000 |
| 43_ridge | pointwise | 3.5638 | [1.2201, 5.8679] | 4.7125 | -14.3825 | -88.4700 | 3 | 0.5095 |
| 43_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | 0.0000 | -15.4759 | -111.6573 | 0 | 0.0000 |
| 43_neural_underharm4 | neural | 4.1711 | [1.6569, 6.7234] | 5.2503 | -14.3936 | -81.0359 | 4 | 1.0000 |
| 43_neural_underharm4 | pointwise | 3.5873 | [1.5211, 5.8927] | 4.6815 | -13.2583 | -86.8980 | 0 | 0.0969 |

## Fixed Joint-Control Population

Exactly 6116 target rows at 1,152 past-selected query frames. This is not the full 318969-row forecast benchmark. All controls below share this population.

| Seed / head | Rule | ADE gain vs floor (%) | Conditional 95% CI | Worst easy gain vs CV (%) | Zero-CV harmed | Switch rate |
|---|---|---:|---|---:|---:|---:|
| 17_ridge | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 17_ridge | neural | 4.9190 | [2.3991, 7.4413] | -42.9424 | 0 | 1.0000 |
| 17_ridge | pointwise | 3.9510 | [1.6066, 6.4947] | -47.4968 | 0 | 0.5672 |
| 17_ridge | independent | 0.3941 | [-0.1852, 0.9620] | -64.7532 | 0 | 0.2752 |
| 17_ridge | scene_uniform | 0.2135 | [0.0241, 0.4349] | -65.3776 | 0 | 0.0240 |
| 17_ridge | joint | 0.3951 | [-0.1879, 0.9624] | -64.7532 | 0 | 0.2752 |
| 17_ridge | unary_exact | 0.3952 | [-0.1878, 0.9626] | -64.7532 | 0 | 0.2752 |
| 17_ridge | joint_exact | 0.3954 | [-0.1876, 0.9628] | -64.7532 | 0 | 0.2752 |
| 17_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 17_neural_underharm4 | neural | 4.9190 | [2.3991, 7.4413] | -42.9424 | 0 | 1.0000 |
| 17_neural_underharm4 | pointwise | 3.0562 | [1.3014, 5.0497] | -45.0454 | 0 | 0.1779 |
| 17_neural_underharm4 | independent | 0.3067 | [0.0222, 0.5989] | -63.3224 | 0 | 0.1187 |
| 17_neural_underharm4 | scene_uniform | 0.0196 | [-0.0006, 0.0458] | -65.4310 | 0 | 0.0139 |
| 17_neural_underharm4 | joint | 0.3070 | [0.0223, 0.5993] | -63.3224 | 0 | 0.1182 |
| 17_neural_underharm4 | unary_exact | 0.3068 | [0.0223, 0.5990] | -63.3224 | 0 | 0.1187 |
| 17_neural_underharm4 | joint_exact | 0.3067 | [0.0222, 0.5989] | -63.3224 | 0 | 0.1187 |
| 29_ridge | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 29_ridge | neural | 4.6430 | [1.8790, 7.3623] | -40.9305 | 0 | 1.0000 |
| 29_ridge | pointwise | 3.8237 | [1.2408, 6.6246] | -46.0218 | 0 | 0.5746 |
| 29_ridge | independent | 0.3506 | [-0.2536, 0.9491] | -64.6799 | 0 | 0.2789 |
| 29_ridge | scene_uniform | 0.2048 | [-0.0301, 0.4809] | -65.2695 | 0 | 0.0204 |
| 29_ridge | joint | 0.3087 | [-0.2845, 0.9042] | -64.6799 | 0 | 0.2783 |
| 29_ridge | unary_exact | 0.3005 | [-0.2923, 0.9004] | -64.6799 | 0 | 0.2789 |
| 29_ridge | joint_exact | 0.3020 | [-0.2923, 0.9004] | -64.6799 | 0 | 0.2789 |
| 29_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 29_neural_underharm4 | neural | 4.6430 | [1.8790, 7.3623] | -40.9305 | 0 | 1.0000 |
| 29_neural_underharm4 | pointwise | 3.0994 | [1.2984, 5.2614] | -42.8496 | 0 | 0.0786 |
| 29_neural_underharm4 | independent | 0.4842 | [0.2255, 0.7811] | -60.1573 | 0 | 0.0388 |
| 29_neural_underharm4 | scene_uniform | 0.0736 | [0.0092, 0.1668] | -65.3309 | 0 | 0.0136 |
| 29_neural_underharm4 | joint | 0.4842 | [0.2255, 0.7811] | -60.1573 | 0 | 0.0388 |
| 29_neural_underharm4 | unary_exact | 0.4842 | [0.2255, 0.7811] | -60.1573 | 0 | 0.0388 |
| 29_neural_underharm4 | joint_exact | 0.4842 | [0.2255, 0.7811] | -60.1573 | 0 | 0.0388 |
| 43_ridge | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 43_ridge | neural | 4.7526 | [2.3658, 7.0474] | -46.5579 | 0 | 1.0000 |
| 43_ridge | pointwise | 3.6693 | [1.3938, 6.0028] | -46.3641 | 0 | 0.5370 |
| 43_ridge | independent | 0.3826 | [-0.1537, 0.9551] | -63.4568 | 0 | 0.2508 |
| 43_ridge | scene_uniform | 0.1240 | [0.0150, 0.2650] | -65.4970 | 0 | 0.0157 |
| 43_ridge | joint | 0.3299 | [-0.2129, 0.9096] | -63.4568 | 0 | 0.2505 |
| 43_ridge | unary_exact | 0.3257 | [-0.2141, 0.9020] | -63.4568 | 0 | 0.2508 |
| 43_ridge | joint_exact | 0.3288 | [-0.2118, 0.9048] | -63.4568 | 0 | 0.2508 |
| 43_neural_underharm4 | floor | 0.0000 | [0.0000, 0.0000] | -65.6840 | 0 | 0.0000 |
| 43_neural_underharm4 | neural | 4.7526 | [2.3658, 7.0474] | -46.5579 | 0 | 1.0000 |
| 43_neural_underharm4 | pointwise | 3.3916 | [1.5303, 5.4771] | -52.8219 | 0 | 0.1161 |
| 43_neural_underharm4 | independent | 0.5260 | [0.1230, 1.0330] | -63.3214 | 0 | 0.0541 |
| 43_neural_underharm4 | scene_uniform | 0.2319 | [-0.0002, 0.6445] | -63.3214 | 0 | 0.0201 |
| 43_neural_underharm4 | joint | 0.5261 | [0.1231, 1.0331] | -63.3214 | 0 | 0.0540 |
| 43_neural_underharm4 | unary_exact | 0.5260 | [0.1230, 1.0330] | -63.3214 | 0 | 0.0541 |
| 43_neural_underharm4 | joint_exact | 0.5260 | [0.1230, 1.0330] | -63.3214 | 0 | 0.0541 |

## Matched Nonzero Intervention Queries

Matching uses the independent decision count before outcomes. Unary geometry removes only nonadditive pair terms. Undefined all-locality ratios are not replaced by zero.

| Seed / head | Matched nonzero queries | Rule | ADE gain vs floor (%) | Conditional CI | Switch rate |
|---|---:|---|---:|---|---:|
| 17_ridge | 615 | independent | 0.7191 | [-0.0985, 1.5798] | 0.3347 |
| 17_ridge | 615 | unary_exact | 0.7207 | [-0.0969, 1.5795] | 0.3347 |
| 17_ridge | 615 | joint_exact | 0.7209 | [-0.0967, 1.5796] | 0.3347 |
| 17_neural_underharm4 | 311 | independent | 1.4038 | [0.4776, 2.3513] | 0.2566 |
| 17_neural_underharm4 | 311 | unary_exact | 1.4040 | [0.4776, 2.3515] | 0.2566 |
| 17_neural_underharm4 | 311 | joint_exact | 1.4038 | [0.4776, 2.3513] | 0.2566 |
| 29_ridge | 606 | independent | 0.8119 | [-0.1069, 1.8257] | 0.3436 |
| 29_ridge | 606 | unary_exact | 0.7023 | [-0.2194, 1.7341] | 0.3436 |
| 29_ridge | 606 | joint_exact | 0.7044 | [-0.2160, 1.7341] | 0.3436 |
| 29_neural_underharm4 | 173 | independent | 2.4014 | [1.2249, 3.7323] | 0.1699 |
| 29_neural_underharm4 | 173 | unary_exact | 2.4014 | [1.2249, 3.7323] | 0.1699 |
| 29_neural_underharm4 | 173 | joint_exact | 2.4014 | [1.2249, 3.7323] | 0.1699 |
| 43_ridge | 593 | independent | 0.7318 | [0.0102, 1.5034] | 0.3094 |
| 43_ridge | 593 | unary_exact | 0.6305 | [-0.1009, 1.4365] | 0.3094 |
| 43_ridge | 593 | joint_exact | 0.6338 | [-0.0959, 1.4398] | 0.3094 |
| 43_neural_underharm4 | 255 | independent | 1.7929 | [0.7459, 3.0196] | 0.1638 |
| 43_neural_underharm4 | 255 | unary_exact | 1.7929 | [0.7459, 3.0196] | 0.1638 |
| 43_neural_underharm4 | 255 | joint_exact | 1.7929 | [0.7459, 3.0196] | 0.1638 |

## Solver and Learning Diagnostics

| Seed / head | Queries with edges | Matched queries | Nonzero matched | Solver failure counts |
|---|---:|---:|---:|---|
| 17_ridge | 629 | 1152 | 615 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |
| 17_neural_underharm4 | 629 | 1152 | 311 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |
| 29_ridge | 629 | 1152 | 606 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |
| 29_neural_underharm4 | 629 | 1152 | 173 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |
| 43_ridge | 629 | 1152 | 593 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |
| 43_neural_underharm4 | 629 | 1152 | 255 | {'independent': 0, 'scene_uniform': 0, 'joint': 0, 'unary_exact': 0, 'joint_exact': 0} |

Per-locality benefit/harm prediction errors, signed-gain correlations and underestimation rates are retained in `analysis.json`. Unknown outcome rows remain in each decision cohort.
The overlap proxy is computed on forecasts in image coordinates, not measured physical collision risk. No calibrated safety, independent confirmation, deployment promotion, Stage5C or SMC claim.
