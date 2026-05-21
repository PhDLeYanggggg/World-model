# Stage 7 Failure Analysis

Remaining blockers:

1. No verified pedestrian/drone t+50/t+100 source is available locally.
2. Candidate goals are inferred from training endpoints, not true annotated destinations.
3. If GoalBench top3 is saturated by majority baseline, goal prediction is not a strong signal.
4. Interaction auxiliary labels remain weak because most converted episodes are one-primary-agent windows.
5. Do not enter Stage 5C unless failure correction and hardbench gates pass.

## BaselineFailureBench Rows
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention | false_intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_only_residual | eth_ucy | baseline_failure | 10 | 1.319933 | 1.746128 | 0.24408 | 2 | 0.208692 | 0.5 | 0.0 |
| goal_only_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim_i90 | baseline_failure | 100 | 13.5678 | 11.775393 | -0.152216 | 5 | 0.316118 | 0.6 | 0.0 |
| goal_only_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| scene_only_residual | eth_ucy | baseline_failure | 10 | 1.403177 | 1.746128 | 0.196407 | 2 | 0.117042 | 0.25 | 0.0 |
| scene_only_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim_i90 | baseline_failure | 100 | 11.438377 | 11.775393 | 0.02862 | 5 | 0.336895 | 0.6 | 0.0 |
| scene_only_residual | trajnet | baseline_failure | 10 | 2.392291 | 2.399685 | 0.003082 | 4 | 0.05 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | baseline_failure | 10 | 1.691571 | 1.746128 | 0.031244 | 2 | 0.019207 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim | baseline_failure | 100 | 21.036198 | 20.967461 | -0.003278 | 1 | 0.403995 | 0.4 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | baseline_failure | 100 | 11.681197 | 11.775393 | 0.007999 | 5 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_interaction_residual | eth_ucy | baseline_failure | 10 | 1.63435 | 1.746128 | 0.064015 | 2 | 0.100043 | 0.25 | 0.0 |
| goal_interaction_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.008 | 0.0 | 0.0 |
| goal_interaction_residual | tgsim_i90 | baseline_failure | 100 | 12.358656 | 11.775393 | -0.049532 | 5 | 0.397429 | 0.6 | 0.0 |
| goal_interaction_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_scene_interaction_residual | eth_ucy | baseline_failure | 10 | 1.49857 | 1.746128 | 0.141776 | 2 | 0.160772 | 0.5 | 0.0 |
| goal_scene_interaction_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| goal_scene_interaction_residual | tgsim_i90 | baseline_failure | 100 | 13.458937 | 11.775393 | -0.142971 | 5 | 0.393886 | 0.6 | 0.0 |
| goal_scene_interaction_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| topk_goal_mixture_diagnostic | eth_ucy | baseline_failure | 10 | 1.50091 | 1.746128 | 0.140435 | 2 | 0.158391 | 0.5 | 0.0 |
| topk_goal_mixture_diagnostic | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| topk_goal_mixture_diagnostic | tgsim_i90 | baseline_failure | 100 | 13.451885 | 11.775393 | -0.142373 | 5 | 0.392644 | 0.6 | 0.0 |
| topk_goal_mixture_diagnostic | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |

## HardBench Rows
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention | false_intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_only_residual | eth_ucy | hard | 10 | 1.319933 | 1.746128 | 0.24408 | 2 | 0.208692 | 0.5 | 0.0 |
| goal_only_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim_i90 | hard | 100 | 4.29189 | 6.388728 | 0.328209 | 1 | 0.344064 | 0.6 | 0.0 |
| goal_only_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| scene_only_residual | eth_ucy | hard | 10 | 1.403177 | 1.746128 | 0.196407 | 2 | 0.117042 | 0.25 | 0.0 |
| scene_only_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim_i90 | hard | 100 | 4.246666 | 6.388728 | 0.335288 | 1 | 0.398127 | 0.6 | 0.0 |
| scene_only_residual | trajnet | hard | 10 | 2.392291 | 2.399685 | 0.003082 | 4 | 0.05 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | hard | 10 | 1.691571 | 1.746128 | 0.031244 | 2 | 0.019207 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim | hard | 100 | 12.033661 | 11.995052 | -0.003219 | 2 | 0.215999 | 0.2 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | hard | 100 | 6.20554 | 6.388728 | 0.028674 | 1 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_interaction_residual | eth_ucy | hard | 10 | 1.63435 | 1.746128 | 0.064015 | 2 | 0.100043 | 0.25 | 0.0 |
| goal_interaction_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.008 | 0.0 | 0.0 |
| goal_interaction_residual | tgsim_i90 | hard | 100 | 2.480944 | 6.388728 | 0.611669 | 1 | 0.407835 | 0.6 | 0.0 |
| goal_interaction_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_scene_interaction_residual | eth_ucy | hard | 10 | 1.49857 | 1.746128 | 0.141776 | 2 | 0.160772 | 0.5 | 0.0 |
| goal_scene_interaction_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| goal_scene_interaction_residual | tgsim_i90 | hard | 100 | 3.625204 | 6.388728 | 0.432562 | 1 | 0.420808 | 0.6 | 0.0 |
| goal_scene_interaction_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| topk_goal_mixture_diagnostic | eth_ucy | hard | 10 | 1.50091 | 1.746128 | 0.140435 | 2 | 0.158391 | 0.5 | 0.0 |
| topk_goal_mixture_diagnostic | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| topk_goal_mixture_diagnostic | tgsim_i90 | hard | 100 | 3.622159 | 6.388728 | 0.433039 | 1 | 0.421713 | 0.6 | 0.0 |
| topk_goal_mixture_diagnostic | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |

Current verdict: `stage7_scene_goal_grounding_built_but_not_stage5c_ready`.
