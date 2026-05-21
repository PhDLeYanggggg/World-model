# Stage 5B.6 Hard Benchmark Reliability Audit

Reliability rule: `<30` hard episodes is diagnostic only, `30-49` is weak gate only, and `>=50` is official hard gate eligible.

| dataset_name | all_episode_count | hard_episode_count | extreme_episode_count | hard_count_by_actual_verified_t100 | hard_reliability_label | hard_subset_is_gate_eligible | bootstrap_mean | bootstrap_ci | minimum_detectable_improvement | previous_stage5b5_hard_win_statistically_meaningful |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | 23 | 2 | 3 | 0 | diagnostic_only | False | 0.0 | [None, None] | None | False |
| tgsim | 32 | 3 | 4 | 3 | diagnostic_only | False | 0.0 | [None, None] | None | False |
| tgsim_i90 | 31 | 8 | 4 | 8 | diagnostic_only | False | 0.057029 | [0.020276, 0.118409] | 0.060542 | False |
| trajnet | 32 | 6 | 4 | 0 | diagnostic_only | False | 0.014227 | [None, None] | None | False |

Conclusion: Stage 5B.5 hard wins with one or a few episodes are not statistically reliable gate wins.