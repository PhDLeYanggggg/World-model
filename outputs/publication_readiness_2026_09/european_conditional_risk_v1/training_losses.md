# Event-Moment Training Losses

Real Torch risk-head training; no new neural forecast training. First channel is reference-error mass, not benefit. Event masks change target magnitude, so lower raw loss across event arms does not imply better forecasting.

| Head | Steps | First loss | Last logged loss | Fit-loop seconds | Unique drawn fitting rows |
|---|---:|---:|---:|---:|---:|
| complement0_seed17_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed17_all_neural_underharm4 | 2000 | 2.97761440 | 1.09581399 | 1.4794 | 92036 |
| complement0_seed17_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed17_easy_neural_underharm4 | 2000 | 0.00121669 | 0.00653069 | 1.0872 | 92036 |
| complement0_seed29_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed29_all_neural_underharm4 | 2000 | 1.84682012 | 1.69129992 | 1.5493 | 92376 |
| complement0_seed29_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed29_easy_neural_underharm4 | 2000 | 0.01033666 | 0.00618012 | 1.1876 | 92376 |
| complement0_seed43_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed43_all_neural_underharm4 | 2000 | 1.52175283 | 2.76666975 | 1.3679 | 92170 |
| complement0_seed43_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement0_seed43_easy_neural_underharm4 | 2000 | 0.00352120 | 0.00183476 | 1.2315 | 92170 |
| complement1_seed17_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed17_all_neural_underharm4 | 2000 | 1.42426074 | 0.62885648 | 1.3547 | 128437 |
| complement1_seed17_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed17_easy_neural_underharm4 | 2000 | 0.03778332 | 0.00576846 | 1.2785 | 128437 |
| complement1_seed29_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed29_all_neural_underharm4 | 2000 | 2.44703102 | 0.83444011 | 1.9228 | 128514 |
| complement1_seed29_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed29_easy_neural_underharm4 | 2000 | 0.01192921 | 0.01573924 | 1.7711 | 128514 |
| complement1_seed43_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed43_all_neural_underharm4 | 2000 | 2.45530701 | 0.67914689 | 1.1718 | 128263 |
| complement1_seed43_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement1_seed43_easy_neural_underharm4 | 2000 | 0.00334961 | 0.00610442 | 1.6890 | 128263 |
| complement2_seed17_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed17_all_neural_underharm4 | 2000 | 1.14877784 | 0.57081085 | 1.5638 | 83023 |
| complement2_seed17_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed17_easy_neural_underharm4 | 2000 | 0.00478739 | 0.00131038 | 1.4498 | 83023 |
| complement2_seed29_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed29_all_neural_underharm4 | 2000 | 1.52888870 | 0.76434839 | 1.2826 | 83289 |
| complement2_seed29_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed29_easy_neural_underharm4 | 2000 | 0.00124735 | 0.00184623 | 1.5727 | 83289 |
| complement2_seed43_all_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed43_all_neural_underharm4 | 2000 | 0.95938814 | 0.83382392 | 1.8205 | 83084 |
| complement2_seed43_easy_ridge | 0 (closed-form) | n/a | n/a | n/a | n/a |
| complement2_seed43_easy_neural_underharm4 | 2000 | 0.00133565 | 0.00182074 | 1.5719 | 83084 |

The 100-update pilot is included in 36,000 neural updates. All/easy arms use exactly matching sampled row counts and sampler states for each seed/fold; this is verified from checkpoints, not inferred from matching seeds.
Losses are training minibatch losses, not held-data errors, convergence proof, calibrated risk or a model-selection objective.
