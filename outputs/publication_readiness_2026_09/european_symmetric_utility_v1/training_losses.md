# Utility Training Losses

Only the utility objective changed. These are minibatch fitting losses, not held-out accuracy.
Raw MSE and underharm4 loss values are not directly comparable objectives.

| Head | Updates | First loss | Last logged loss | Fit seconds | Known unique rows |
|---|---:|---:|---:|---:|---:|
| neural_complement0_seed17 | 2000 | 0.57871103 | 0.23001716 | 1.305 | 92036 |
| neural_complement0_seed29 | 2000 | 0.37169564 | 0.33632177 | 1.275 | 92376 |
| neural_complement0_seed43 | 2000 | 0.35409486 | 0.49486911 | 1.478 | 92170 |
| neural_complement1_seed17 | 2000 | 0.14610437 | 0.15534791 | 2.293 | 128437 |
| neural_complement1_seed29 | 2000 | 0.35726887 | 0.18700774 | 1.725 | 128514 |
| neural_complement1_seed43 | 2000 | 0.34619439 | 0.12556520 | 1.900 | 128263 |
| neural_complement2_seed17 | 2000 | 0.09872885 | 0.09432085 | 1.691 | 83023 |
| neural_complement2_seed29 | 2000 | 0.26306260 | 0.12686454 | 1.432 | 83289 |
| neural_complement2_seed43 | 2000 | 0.11409449 | 0.19947794 | 1.392 | 83084 |
| damping097_complement0_seed17 | 2000 | 0.02898780 | 0.01640303 | 1.149 | 92036 |
| damping097_complement0_seed29 | 2000 | 0.02373221 | 0.01794111 | 1.093 | 92376 |
| damping097_complement0_seed43 | 2000 | 0.02143269 | 0.02438432 | 1.348 | 92170 |
| damping097_complement1_seed17 | 2000 | 0.02484176 | 0.01633625 | 1.661 | 128437 |
| damping097_complement1_seed29 | 2000 | 0.03178028 | 0.01633113 | 1.719 | 128514 |
| damping097_complement1_seed43 | 2000 | 0.03518930 | 0.01562281 | 1.845 | 128263 |
| damping097_complement2_seed17 | 2000 | 0.01993177 | 0.01604749 | 1.596 | 83023 |
| damping097_complement2_seed29 | 2000 | 0.02342943 | 0.02407791 | 1.326 | 83289 |
| damping097_complement2_seed43 | 2000 | 0.02913837 | 0.01734450 | 1.590 | 83084 |

The 100-update pilot is included in 36,000 updates. Every new head has exactly the old head's training draws, known-label support, preprocessing and initialization constants.
All fits are completed before outer-source readout. No early stopping or checkpoint selection from that readout.
