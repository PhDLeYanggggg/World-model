# Stage 5B.6 Interaction Ablation

| ablation | mean_all_target_improvement | mean_hard_target_improvement | note |
| --- | --- | --- | --- |
| graph attention interaction | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| graph attention + temporal neighbor history | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| nearest-neighbor scalar features only | 0.068819 | 0.054096 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| no interaction | 0.037748 | 0.04006 | quick deterministic ablation; interaction is past-only from kNN world-state table |
