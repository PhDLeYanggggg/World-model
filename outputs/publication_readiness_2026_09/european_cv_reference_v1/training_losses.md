# CV-Reference Cost Training

Real Torch cost training, not new forecasting training. Three seeds, fixed budgets, no validation or held-out checkpoint selection.

| Head | Steps | First loss | Last logged loss | Training seconds | Unknown rows drawn |
|---|---:|---:|---:|---:|---:|
| complement0_seed17_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement0_seed17_neural_underharm4 | 2000 | 1.880329 | 0.382622 | 1.566 | 0 |
| complement0_seed29_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement0_seed29_neural_underharm4 | 2000 | 1.159645 | 0.873747 | 1.485 | 0 |
| complement0_seed43_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement0_seed43_neural_underharm4 | 2000 | 0.841455 | 0.978714 | 1.632 | 0 |
| complement1_seed17_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement1_seed17_neural_underharm4 | 2000 | 0.282911 | 0.248754 | 2.078 | 0 |
| complement1_seed29_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement1_seed29_neural_underharm4 | 2000 | 0.688263 | 0.414065 | 2.119 | 0 |
| complement1_seed43_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement1_seed43_neural_underharm4 | 2000 | 1.009336 | 0.251122 | 1.509 | 0 |
| complement2_seed17_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement2_seed17_neural_underharm4 | 2000 | 0.198507 | 0.161380 | 1.700 | 0 |
| complement2_seed29_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement2_seed29_neural_underharm4 | 2000 | 0.326085 | 0.202623 | 2.067 | 0 |
| complement2_seed43_ridge | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |
| complement2_seed43_neural_underharm4 | 2000 | 0.162543 | 0.263672 | 2.136 | 0 |

Training losses use a fitting-only cost scale and asymmetric harm penalty; they are not held-site errors or calibrated harm bounds. The 100-update pilot is included in, not added to, 18,000 neural updates.
