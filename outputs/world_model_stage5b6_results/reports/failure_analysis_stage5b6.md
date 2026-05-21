# Stage 5B.6 Failure Analysis

Main failures:

1. Hard benchmark reliability is insufficient: no dataset has >=50 hard episodes, so hard wins remain diagnostic.
2. Real pedestrian/drone long horizon is still missing: no verified t+50/t+100 pedestrian/drone source was added.
3. Official gated residual variants beat the strongest causal baseline on only one dataset target horizon.
4. Graph interaction features did not improve hard-subset performance over no-interaction ablation.
5. Verified t+100 improvement is not achieved by the official gated residual variants.

Best official gated residual by dataset:
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_residual_all_data | eth_ucy | all | 10 | 0.63948 | 0.713643 | 0.103923 | 6 | 0.04003 |
| gated_residual_failure_classifier_aux | tgsim | all | 100 | 6.306686 | 6.062032 | -0.040358 | 4 | 0.149895 |
| gated_residual_all_data | tgsim_i90 | all | 100 | 10.252016 | 10.327657 | 0.007324 | 6 | 0.04 |
| gated_residual_all_data | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 |

Interaction ablation:
| ablation | mean_all_target_improvement | mean_hard_target_improvement | note |
| --- | --- | --- | --- |
| graph attention interaction | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| graph attention + temporal neighbor history | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| nearest-neighbor scalar features only | 0.068819 | 0.054096 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| no interaction | 0.037748 | 0.04006 | quick deterministic ablation; interaction is past-only from kNN world-state table |

Current verdict: `stage5b6_reliability_repaired_but_deterministic_gate_failed`.
