# New Damping-Head Training Losses

Only damping heads are newly fitted. Cached neural-candidate head losses are in the preceding experiment.
Training minibatch objectives differ between utility/all/easy tasks; compare downstream errors, not cross-task raw loss.

| Head | Updates | First loss | Last logged loss | Fit seconds | Unique fitting draws |
|---|---:|---:|---:|---:|---:|
| damping097_complement0_seed17_utility_neural_underharm4 | 2000 | 0.04577683 | 0.02417130 | 1.2820 | 92036 |
| damping097_complement0_seed17_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed17_all_neural_underharm4 | 2000 | 1.25303996 | 0.78061998 | 1.2098 | 92036 |
| damping097_complement0_seed17_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed17_easy_neural_underharm4 | 2000 | 0.00058353 | 0.00089252 | 1.1252 | 92036 |
| damping097_complement0_seed29_utility_neural_underharm4 | 2000 | 0.03949484 | 0.03822715 | 1.2019 | 92376 |
| damping097_complement0_seed29_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed29_all_neural_underharm4 | 2000 | 0.80543083 | 0.99388295 | 1.3492 | 92376 |
| damping097_complement0_seed29_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed29_easy_neural_underharm4 | 2000 | 0.00193556 | 0.00068819 | 1.4351 | 92376 |
| damping097_complement0_seed43_utility_neural_underharm4 | 2000 | 0.04003049 | 0.03169845 | 1.1885 | 92170 |
| damping097_complement0_seed43_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed43_all_neural_underharm4 | 2000 | 0.88499641 | 2.24286723 | 1.4669 | 92170 |
| damping097_complement0_seed43_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement0_seed43_easy_neural_underharm4 | 2000 | 0.00078327 | 0.00058860 | 1.3717 | 92170 |
| damping097_complement1_seed17_utility_neural_underharm4 | 2000 | 0.03729451 | 0.02806181 | 2.0323 | 128437 |
| damping097_complement1_seed17_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed17_all_neural_underharm4 | 2000 | 1.24991024 | 0.50504351 | 1.5702 | 128437 |
| damping097_complement1_seed17_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed17_easy_neural_underharm4 | 2000 | 0.00445422 | 0.00152133 | 2.0102 | 128437 |
| damping097_complement1_seed29_utility_neural_underharm4 | 2000 | 0.04616442 | 0.03043780 | 1.4212 | 128514 |
| damping097_complement1_seed29_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed29_all_neural_underharm4 | 2000 | 2.01636887 | 0.55288899 | 1.4274 | 128514 |
| damping097_complement1_seed29_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed29_easy_neural_underharm4 | 2000 | 0.00232901 | 0.00270868 | 1.1706 | 128514 |
| damping097_complement1_seed43_utility_neural_underharm4 | 2000 | 0.05505189 | 0.02536984 | 1.4257 | 128263 |
| damping097_complement1_seed43_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed43_all_neural_underharm4 | 2000 | 1.58895755 | 0.52175301 | 1.4795 | 128263 |
| damping097_complement1_seed43_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement1_seed43_easy_neural_underharm4 | 2000 | 0.00122582 | 0.00133806 | 1.7936 | 128263 |
| damping097_complement2_seed17_utility_neural_underharm4 | 2000 | 0.03500555 | 0.02418629 | 1.6455 | 83023 |
| damping097_complement2_seed17_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed17_all_neural_underharm4 | 2000 | 1.03370488 | 0.52187300 | 1.5977 | 83023 |
| damping097_complement2_seed17_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed17_easy_neural_underharm4 | 2000 | 0.00152570 | 0.00108992 | 1.4186 | 83023 |
| damping097_complement2_seed29_utility_neural_underharm4 | 2000 | 0.03570398 | 0.03138945 | 1.5653 | 83289 |
| damping097_complement2_seed29_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed29_all_neural_underharm4 | 2000 | 1.45904529 | 0.63246167 | 1.4759 | 83289 |
| damping097_complement2_seed29_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed29_easy_neural_underharm4 | 2000 | 0.00116404 | 0.00102662 | 1.2507 | 83289 |
| damping097_complement2_seed43_utility_neural_underharm4 | 2000 | 0.03993659 | 0.03025493 | 1.4545 | 83084 |
| damping097_complement2_seed43_all_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed43_all_neural_underharm4 | 2000 | 0.90706331 | 0.73305845 | 1.8046 | 83084 |
| damping097_complement2_seed43_easy_ridge | 0 (ridge closed-form) | n/a | n/a | n/a | n/a |
| damping097_complement2_seed43_easy_neural_underharm4 | 2000 | 0.00112564 | 0.00103167 | 1.1479 | 83084 |

54,000 new neural updates; pilot included. Exact source draw sequence, support, weights and CV cost scale match across six neural heads per seed/fold.
Features/targets and their fitted moments differ necessarily with the candidate. This is not new trajectory-model training.
