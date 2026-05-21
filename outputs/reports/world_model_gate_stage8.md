# Stage 8 Gates

Passed: 4 / 11

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Pedestrian/Drone Long-Horizon Gate | False | eligible pedestrian/drone long-horizon sources=0 | Provide local SDD/OpenTraj/full pedestrian data with verified t+50/t+100. |
| Scene-Gold Gate | True | gold/silver=0; usable_scene_packs=5 | Manually confirm at least one pedestrian/drone scene or two usable scene packs. |
| Multi-Agent Episode Gate | True | episodes_with_ge2_agents=78 | Build more multi-agent windows from real pedestrian/drone scenes. |
| GoalBench-Gold Gate | True | test={'samples': 12, 'top1_accuracy': 0.5, 'top3_accuracy': 0.75, 'goal_NLL': 1.255832, 'goal_ECE': 0.104385, 'goal_entropy': 1.245901, 'majority_top1': 0.333333, 'majority_top3': 0.666667, 'hard_failure_goal_accuracy': 0.5, 'beats_majority': True, 'top3_saturated': False} | Improve gold/silver goals; avoid majority top-k saturation. |
| Failure Predictor Gate | False | stage7_best_AUROC=0.943396; stage8_best_AUROC=0.896021 | Scene/goal/multi-agent features should improve failure prediction. |
| Failure Correction Gate | False | BaselineFailureBench improvements={'tgsim_i90': -0.001179} | Need >=10% over strongest baseline on BaselineFailureBench. |
| HardBench Gate | False | HardBench improvements={'tgsim': 0.0, 'tgsim_i90': 0.000266} | Need >=10% over strongest baseline on HardBench. |
| Easy Preservation Gate | True | easy improvements={'tgsim': 0.001787} | Keep easy cases near baseline. |
| Interaction Gate | False | full_hard={'tgsim': -0.299673, 'tgsim_i90': -0.003275}; no_interaction_hard={'tgsim': -0.298009, 'tgsim_i90': -0.003544}; aux={'graph_improves_over_scalar_auxiliary': False, 'improves_hard_failure_trajectory_performance': False, 'reason': 'Stage 8 interaction auxiliary is diagnostic unless benchmark metrics show trajectory lift.'} | Multi-agent interaction must improve hard/failure trajectory metrics. |
| Verified Long-Horizon Gate | False | verified long-horizon improvements={'tgsim': 0.0, 'tgsim_i90': 0.000266} | Need >=5% on verified t+50/t+100. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C. Scene/goal-conditioned deterministic correction is not strong enough. | Pass Stage 8 deterministic scene/goal gates first. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `71`
verdict: `stage8_scene_goal_multiagent_scaffold_not_stage5c_ready`
