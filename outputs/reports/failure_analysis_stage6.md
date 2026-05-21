# Stage 6 Failure Analysis

Failures that still block Stage 5C:

1. No real pedestrian/drone verified t+50/t+100 source is available.
2. Failure-aware gated residual does not improve BaselineFailureBench by the required 10%.
3. Verified long-horizon improvement still fails the >=5% gate.
4. Interaction features do not produce a reliable gain over no-interaction.
5. Traffic long-horizon results cannot be presented as pedestrian world-model success.

## BaselineFailureBench Best Rows
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scalar_interaction_ablation | eth_ucy | baseline_failure | 10 | 1.493165 | 1.746128 | 0.144871 | 2 | 0.122912 | 0.5 |
| no_interaction_ablation | tgsim | baseline_failure | 100 | 21.038969 | 20.967461 | -0.00341 | 1 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim_i90 | baseline_failure | 100 | 10.088638 | 11.775393 | 0.143244 | 5 | 0.428319 | 0.8 |
| no_interaction_ablation | trajnet | baseline_failure | 10 | 2.351784 | 2.399685 | 0.019962 | 4 | 0.085581 | 0.125 |

## Verified t+100 Best Rows
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_interaction_ablation | tgsim | verified_t100 | 100 | 6.088911 | 6.062032 | -0.004434 | 4 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim_i90 | verified_t100 | 100 | 8.607503 | 10.327657 | 0.166558 | 6 | 0.414962 | 0.8 |

## Interaction Ablation
| model | mean_hard_improvement | mean_failure_improvement |
| --- | --- | --- |
| no_interaction_ablation | 0.109813 | 0.055893 |
| scalar_interaction_ablation | 0.117277 | 0.070062 |
| graph_interaction_ablation | 0.018565 | 0.010558 |

Current verdict: `stage6_failure_bench_built_but_not_stage5c_ready`.
