# Auto-Orchestrator Current State

- current_highest_stage: `13`
- expert_audit_score: `84`
- verdict: `stage13_deterministic_repair_loop_executed_not_stage5c_ready`
- model_type: `2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold`
- true_3D: `False`
- large_scale_foundation_model: `False`
- latent_generative_ready: `False`
- smc_ready: `False`
- strongest_causal_baseline: `{'aerialmpt': 'constant_velocity_causal_fd', 'eth_ucy': 'scene_clamped_baseline', 'eth_ucy_ewap': 'constant_position', 'trajnet': 'damped_velocity'}`
- learned_model_beats_strongest_baseline: `否`
- verified_pedestrian_drone_t50_t100: `['eth_ucy_ewap']`
- human_gold_annotations: `0`
- human_silver_annotations: `3`
- goalbench_beats_majority: `False`
- hardbench_baselinefailure_enough: `True`

## Top Failures

- Stage 13 found no evaluable EWAP t+100 rows under the per-agent causal mask.
- HardBench/BaselineFailureBench improvement is below deterministic gate thresholds.
- Interaction features still do not improve trajectory metrics.

## Best Automatic Directions

- Stage 13 deterministic repair: failure-aware bounded residual that preserves strong causal baselines.
- Add/verify more SDD/OpenTraj pedestrian/drone scenes with images/homography where legally available.
- Upgrade rule-confirmed silver annotations into human-confirmed silver/gold and measure GoalBench lift.

## Missing Requested Files

- outputs/reports/report_stage11_final.md
