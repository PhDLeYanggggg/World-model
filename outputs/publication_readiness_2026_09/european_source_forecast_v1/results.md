# Source Forecast Results

Fresh Torch training and source-excluded inference; cached metric reproduction and separate fresh checkpoint replay pass.
Twelve opened training localities, not reserved confirmation. Gains are equal-locality percentages. All coordinates are image pixels; the task is 8 observed / 12 requested raw-stride-12 steps.
Outcome: average prediction signal, failed easy/zero-reference protection, no deployment promotion. The training-selected baseline is not the retrospective best fixed baseline on the entire readout.

## Causal Controls

| Baseline | ADE gain vs CV (%) | Conditional 95% locality bootstrap |
|---|---:|---|
| constant_position | -87.3618 | [-149.9120, -39.7985] |
| constant_velocity_causal_fd | 0.0000 | [0.0000, 0.0000] |
| damped_velocity_090 | 1.0870 | [-12.4299, 10.0049] |
| damped_velocity_097 | 3.9755 | [0.9038, 5.8037] |
| history_ols4_velocity | 3.5357 | [0.4083, 6.7748] |
| history_ols8_velocity | -9.4689 | [-16.0733, -2.9376] |

## Neural Forecaster

| Seed | ADE gain vs fit-selected strongest (%) | 95% CI | FDE gain vs strongest (%) | ADE gain vs CV (%) | Easy gain vs CV (%) | Worst easy-locality gain (%) | Zero-CV harmed rows |
|---|---:|---|---:|---:|---:|---:|---:|
| 17 | 4.3385 | [1.5540, 7.3289] | 5.2765 | 2.4491 | -13.7319 | -75.9522 | 4 |
| 29 | 3.8309 | [0.8333, 6.8765] | 5.1391 | 1.8702 | -13.6479 | -75.6037 | 4 |
| 43 | 4.1711 | [1.6569, 6.7234] | 5.2503 | 2.1535 | -14.3936 | -81.0359 | 4 |

Mean-seed ADE gain vs selected strongest: **4.1135%**, conditional CI [1.3672, 6.9485].
This averages errors from separate seeds, not their predictions. A negative gain is degradation.
On the same equal-locality CV-relative scale, fixed damping 0.97 gives +3.9755%, compared with +2.1576% for mean neural errors. The neural model therefore has not established superiority over every strong fixed control. This observation does not retrospectively change the registered comparison or select a replacement model.
Positive-easy degradation is 13.65--14.39% across seeds, above the unchanged 2% ceiling. All four exact-zero-CV rows are harmed in each neural seed. The corresponding training-selected fallback itself degrades positive-easy errors by 15.48%; falling back to it cannot by itself certify CV-relative safety.

## Per Locality

| Locality | Supported targets | Mean neural ADE (px) | Strongest ADE (px) | Gain (%) | Neural p95 (px) |
|---|---:|---:|---:|---:|---:|
| eu-locality-007 | 5526 | 12.2600 | 13.2776 | 7.6642 | 43.7029 |
| eu-locality-008 | 157961 | 19.1392 | 18.3769 | -4.1483 | 69.7004 |
| eu-locality-020 | 1225 | 9.5568 | 11.0120 | 13.2154 | 29.4637 |
| eu-locality-048 | 8058 | 26.7712 | 26.7207 | -0.1889 | 93.4456 |
| eu-locality-067 | 6903 | 26.8134 | 26.5893 | -0.8426 | 86.5325 |
| eu-locality-074 | 102348 | 18.0470 | 18.6622 | 3.2966 | 59.4041 |
| eu-locality-082 | 1368 | 12.9399 | 12.9678 | 0.2148 | 48.7526 |
| eu-locality-110 | 8550 | 26.3595 | 27.9881 | 5.8189 | 113.4797 |
| eu-locality-112 | 5004 | 22.8516 | 23.2973 | 1.9131 | 79.0609 |
| eu-locality-119 | 5639 | 26.8125 | 30.4754 | 12.0193 | 109.3396 |
| eu-locality-124 | 3550 | 39.8242 | 42.6462 | 6.6174 | 183.0778 |
| eu-locality-126 | 5790 | 17.9341 | 18.6390 | 3.7820 | 76.5865 |

The JSON retains every seed, baseline, hard/positive-easy and complete-future sensitivity, zero-reference harms, unknown label coverage and per-site tails.
Bootstrap units are locality groups, never overlapping windows. Shared fitted models and opened source sites make these intervals conditional exploratory evidence.
No cross-camera pixel pooling is interpreted as a common physical scale. Missing FDE endpoints remain unknown rather than using a last-valid endpoint.
