# Fitting-Only Decision Boundary Audit

Fresh recomputation from verified checkpoints. Only B fitting rows are
inspected; no new V or reserved test readout. Full empirical query-draw
weights retain empty proposal queries. Quantities below are normalized
signed-budget costs, not native ADE or calibrated safety probabilities.

| Context | Model | Channel | False-safe outputs | False-reject outputs | Weighted sign regret |
|---|---|---|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | warm | all | 92 | 2422 | 0.00588533 |
| eqmotion_fold1_seed17_critic0 | warm | easy | 51 | 624 | 0.00182307 |
| eqmotion_fold1_seed17_critic0 | in_source_errors | all | 79 | 2372 | 0.00552193 |
| eqmotion_fold1_seed17_critic0 | in_source_errors | easy | 48 | 615 | 0.00165823 |
| eqmotion_fold1_seed17_critic0 | held_source_errors | all | 123 | 2175 | 0.00522639 |
| eqmotion_fold1_seed17_critic0 | held_source_errors | easy | 84 | 536 | 0.00178644 |
| eqmotion_fold0_seed17_critic1 | warm | all | 290 | 1421 | 0.00405004 |
| eqmotion_fold0_seed17_critic1 | warm | easy | 75 | 135 | 0.00062953 |
| eqmotion_fold0_seed17_critic1 | in_source_errors | all | 168 | 1688 | 0.00401806 |
| eqmotion_fold0_seed17_critic1 | in_source_errors | easy | 65 | 131 | 0.00040773 |
| eqmotion_fold0_seed17_critic1 | held_source_errors | all | 327 | 1394 | 0.00508279 |
| eqmotion_fold0_seed17_critic1 | held_source_errors | easy | 91 | 108 | 0.00052220 |
| eqmotion_fold0_seed17_critic2 | warm | all | 2219 | 16614 | 0.02006178 |
| eqmotion_fold0_seed17_critic2 | warm | easy | 876 | 2797 | 0.00837634 |
| eqmotion_fold0_seed17_critic2 | in_source_errors | all | 1397 | 18392 | 0.01739493 |
| eqmotion_fold0_seed17_critic2 | in_source_errors | easy | 931 | 2586 | 0.00672280 |
| eqmotion_fold0_seed17_critic2 | held_source_errors | all | 1780 | 17903 | 0.01810547 |
| eqmotion_fold0_seed17_critic2 | held_source_errors | easy | 992 | 2334 | 0.00481123 |

A false-safe sign has predicted score <=0 and positive realized budget
excess. A false rejection has score >0 and negative realized excess.
Both are retrospective supervision diagnostics, not observable labels
at inference or rowwise safety guarantees. Counts overlap across fitting
contexts. The original missing-reference/support issue is unchanged.

The accompanying JSON reports fixed score bins, error mass and MSE.
This audit can motivate an objective comparison; it cannot show that
changing the objective will generalize or repair missing support.
