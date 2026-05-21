# Stage 7 Gates

Passed: 5 / 10

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Pedestrian/Drone Long-Horizon Gate | False | 0 pedestrian/drone sources support verified t+50/t+100 | Add SDD/OpenTraj/full pedestrian data with verified long horizon. |
| Scene Pack Gate | True | 4 scene packs; pedestrian scene packs=2 | Add real scene images/homographies/walkable annotations. |
| GoalBench Gate | False | test top3=0.782609, majority_top3=0.826087, NLL=1.375813 | Improve candidate goals; avoid too-few-goal top3 saturation. |
| Failure Predictor Improvement Gate | True | stage6_AUROC=0.899098, best_stage7_AUROC=0.943396, best_stage7_AUPRC=0.81326 | Goal/scene features must improve failure prediction. |
| Failure Correction Gate | True | BaselineFailureBench improvements={'eth_ucy': 0.24408, 'tgsim': 0.0, 'tgsim_i90': 0.02862, 'trajnet': 0.003082} | Need >=10% improvement on BaselineFailureBench. |
| HardBench Gate | True | HardBench improvements={'eth_ucy': 0.24408, 'tgsim': 0.0, 'tgsim_i90': 0.611669, 'trajnet': 0.003082} | Need >=10% improvement on official hard subset. |
| Easy Preservation Gate | True | easy improvements={'eth_ucy': -0.033983, 'trajnet': 0.0} | Do not degrade easy cases. |
| Interaction Auxiliary Gate | False | metrics={'future_nearest_neighbor_distance_MAE': 0.006819, 'future_ttc_min_MAE': 0.009969, 'close_pass_event_AUROC': 0.789547, 'close_pass_event_AUPRC': 0.69836, 'density_increase_event_AUROC': 0.681818, 'density_increase_event_AUPRC': 0.378693, 'crossing_conflict_event_AUROC': 0.5, 'crossing_conflict_event_AUPRC': 0.0, 'stop_go_event_AUROC': 0.5, 'stop_go_event_AUPRC': 0.0, 'local_congestion_event_AUROC': 0.789547, 'local_congestion_event_AUPRC': 0.69836, 'improves_hard_failure_trajectory_performance': False} | Auxiliary interaction tasks must improve hard/failure trajectory metrics. |
| Verified Long-Horizon Gate | False | verified long-horizon improvements={'tgsim': 0.0, 'tgsim_i90': 0.006883} | Need >=5% on at least one verified t+50/t+100 source. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C. Goal-conditioned deterministic world model is not strong enough. | Pass Stage 7 deterministic gates first. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `71`
verdict: `stage7_scene_goal_grounding_built_but_not_stage5c_ready`
