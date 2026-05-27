# Stage42-ES Interaction / Occupancy Target Selection

- source: `fresh_stage42_interaction_occupancy_target_selection`
- generated_at_utc: `2026-05-27T03:43:38.893031+00:00`
- git_commit: `2657585`
- input_hash: `f516adfd78b84d584737ec846c3666e7b43af714813a7a03dbf5d8d422af5d36`
- gate: `17 / 17`
- verdict: `stage42_es_interaction_occupancy_target_selection_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-ES fresh-reruns the scalar proximity/occupancy loss target and explicit group-consistency repair target under the same source-level raw-frame protocol.
- 本阶段选择下一步 interaction/occupancy target，不下载、不转换、不执行 Stage5C、不启用 SMC。
- future waypoints / endpoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Selected Target

- selected_target_family: `explicit_group_consistency_repair`
- decision: `continue_with_explicit_group_consistency_interaction_target`
- selection_score: `1.018400`
- rationale: Explicit group-consistency is selected only if it improves the protected full-waypoint floor, improves over Stage42-AM on all/hard, preserves easy, and does not worsen near@0.05. Scalar proximity/occupancy weighting is retained only as diagnostic when it is positive but not better than the baseline-family full-waypoint control.

## Target Family Comparison

| family | promotable | all | t50 | t100 raw | hard | easy | delta AM all | delta AM hard | near base/final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scalar_proximity_occupancy_loss` | False | 25.51% | 22.14% | 14.34% | 23.74% | -29.23% | 0.93% | -0.01% | n/a |
| `explicit_group_consistency_repair` | True | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 0.14% | 0.14% | 1.94%/1.38% |

## Interpretation

- Scalar proximity/occupancy weighting is useful as a diagnostic target but is not selected as the next deployable interaction target unless it beats Stage42-AM on all and hard.
- Explicit group-consistency repair is selected only when it is validation/test-once promotable, easy-safe, and not worse on near@0.05.
- This moves ER-2 away from shallow context routing toward a physical group/occupancy-style target.

## Gate

| gate | pass |
| --- | ---: |
| `dh_rerun_completed` | True |
| `di_rerun_completed` | True |
| `target_families_compared` | True |
| `scalar_target_recorded_diagnostic` | True |
| `group_consistency_selected` | True |
| `group_consistency_promotable` | True |
| `group_all_positive` | True |
| `group_t50_positive` | True |
| `group_hard_positive` | True |
| `group_easy_safe` | True |
| `group_beats_stage42_am_all` | True |
| `group_beats_stage42_am_hard` | True |
| `near005_not_worse` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
