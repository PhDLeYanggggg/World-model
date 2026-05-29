# Stage42-JR Source Context Fresh Replay

- source: `fresh_stage42_jr_source_context_fresh_replay`
- generated_at_utc: `2026-05-29T04:34:58.411682+00:00`
- git_commit: `0ca07d9`
- input_hash: `48507e74c9c2446619a101aec89152c8c905b0b16cd0527ddf9f1d83bf74ac36`
- gate: `12 / 12`
- verdict: `stage42_jr_source_context_negative_evidence_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JR consolidates fresh AR/AS source-level sequence and graph context replay evidence.
- Negative sequence/graph residual results are not hidden or packaged as a contribution.
- future endpoints / waypoints are labels/evaluation only, not inference input.
- No central velocity, no test endpoint goals, no test threshold tuning.
- No metric/seconds claim, no Stage5C execution, no SMC.

## Summary

- decision: `sequence_and_graph_context_negative_keep_baseline_family_rollout_context_as_dominant_mechanism`
- sequence_report_verdict: `stage42_ar_sequence_context_evidence_partial_or_negative`
- graph_report_verdict: `stage42_as_graph_context_evidence_partial_or_negative`
- baseline_family_all/t50/hard: `0.287773` / `0.315425` / `0.275812`
- best_sequence_all/t50/hard_delta: `-0.024458` / `-0.083057` / `-0.028398`
- best_graph_all/t50/hard_delta: `-0.023009` / `-0.085752` / `-0.026235`
- next_repair_hypothesis: `Switch from residual full-waypoint deltas to gain/harm/intervention or source-slice objectives before claiming independent scene/goal/interaction contribution.`

## Sequence Deltas vs Baseline-Family Context

| variant | all | t50 | t100 raw | hard/failure | easy | positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequence_history` | -0.024458 | -0.083057 | -0.001114 | -0.028398 | 0.014358 | `False` |
| `sequence_goal_neighbor_no_history` | -0.026473 | -0.092136 | -0.001603 | -0.029072 | 0.038166 | `False` |
| `sequence_history_goal_neighbor` | -0.087523 | -0.087529 | -0.002838 | -0.096088 | 0.053745 | `False` |

## Graph Deltas vs Baseline-Family Context

| variant | all | t50 | t100 raw | hard/failure | easy | positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `graph_only` | -0.023503 | -0.085752 | -0.000102 | -0.027002 | 0.006648 | `False` |
| `graph_goal` | -0.023123 | -0.086390 | -0.000246 | -0.026341 | 0.007873 | `False` |
| `graph_history_goal` | -0.023009 | -0.086417 | -0.000507 | -0.026235 | 0.007797 | `False` |

## Failure Taxonomy

- dominant_success_mechanism: `baseline_family_rollout_context`
- sequence_failure: temporal history residuals reduced all/t50/hard improvement versus the protected baseline-family first stage.
- graph_failure: current-frame kNN graph and goal/history graph variants reduced all/t50/hard improvement versus the protected baseline-family first stage.
- likely_cause: The current residual-delta objective rewards small unsafe corrections after a strong protected rollout floor; it does not learn switchability/gain/harm enough to capture independent interaction value.
- what_not_to_claim: `['sequence context is an independent main contribution', 'graph/interaction context is an independent main contribution', 'JEPA/Transformer/scene context is proven by these AR/AS runs']`

## No-Leakage And Claim Boundary

- no_leakage: `{'sequence_no_leakage_pass': True, 'graph_no_leakage_pass': True, 'future_endpoint_input': False, 'future_waypoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'sequence_context_main_claim': False, 'graph_interaction_main_claim': False, 'stage5c_executed': False, 'smc_enabled': False}`
