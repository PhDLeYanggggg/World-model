# Stage42-CN Bridge / Shape Composer Audit

- source: `fresh_synthesis_from_stage42_cm_j_x_artifacts`
- generated_at_utc: `2026-05-26T18:40:45.326056+00:00`
- git_commit: `a4ff636`
- input_hash: `0d0806bec80193a6564f8e20a9778c6edc6b179375e18f15606afa719f5c7631`
- gate: `15 / 15`
- verdict: `stage42_cn_bridge_shape_composer_audit_pass_blocker_documented`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CN 是 bridge/shape composer 审计，不重新训练，不调 test threshold。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- endpoint-only 或 endpoint-linear bridge 成功不能自动算 learned full-waypoint shape success。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Composer Decision

- selected deployment policy: `keep_endpoint_linear_bridge_floor_with_full_waypoint_auxiliary_reporting`
- deployable bridge/shape composer available now: `False`
- common validation endpoint-vs-full-waypoint comparison available: `False`
- blocked next requirement: Build common validation-aligned endpoint-linear-vs-full-waypoint row cache before any deployment switch.
- reason: Stage42-J supplies validation-only full-waypoint/static gating evidence, but there is not yet a common row-level validation comparison that can safely switch between endpoint-linear bridge and protected full-waypoint sequence. Stage42-CM shows full-waypoint improves t50/t100 raw-frame over the linear bridge but loses all-ADE and hard/failure, so deployment remains the endpoint-linear floor with full-waypoint as auxiliary horizon evidence.

## Candidate Rows

| candidate | source | status | validation rule | rows | all | t50 | t100 diag | hard | easy | switch | note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_linear_bridge_floor` | `fresh_run` | `current_deployable_all_ade_floor` | `preexisting_protected_policy_or_diagnostic` | 55528 | 21.03% | 13.65% | 14.69% | 20.38% | -14.51% | 34.10% | Current M3W-Neural v1 protected endpoint dynamics projected through endpoint-linear waypoint bridge. |
| `protected_full_waypoint_sequence` | `fresh_run` | `protected_full_waypoint_horizon_auxiliary` | `preexisting_protected_policy_or_diagnostic` | 55528 | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% | 29.46% | Actual full-waypoint sequence model; useful on t50/t100 raw-frame but not an all-ADE replacement. |
| `stage42j_static_gated` | `cached_verified_checkpoints_fresh_static_gate_eval` | `validation_selected_full_waypoint_shape_candidate` | `domain_horizon_expert_selected_on_val` | 55528 | 3.62% | 3.69% | 2.67% | 3.97% | 0.00% | 15.08% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42j_static_alpha025` | `cached_verified_checkpoints_fresh_static_gate_eval` | `validation_selected_full_waypoint_shape_candidate` | `domain_horizon_expert_selected_on_val` | 55528 | 3.52% | 3.44% | 3.01% | 3.87% | 0.00% | 15.71% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42j_no_static` | `cached_verified_checkpoints_fresh_static_gate_eval` | `validation_selected_full_waypoint_shape_candidate` | `domain_horizon_expert_selected_on_val` | 55528 | 1.15% | 1.99% | 1.41% | 1.29% | 0.00% | 7.79% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42x_unified_row_level_full_waypoint_cache` | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `row_level_full_waypoint_three_domain_positive_auxiliary` | `combo_sources_selected_on_val` | 166584 | 9.00% | 6.11% | 8.15% | 9.37% | 0.11% | 23.26% | Unified row-level full-waypoint cache is positive but below the current endpoint-linear bridge floor on all/t50/hard. |
| `ungated_full_waypoint_sequence` | `fresh_run` | `diagnostic_unsafe_not_deployable` | `preexisting_protected_policy_or_diagnostic` | 55528 | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | 100.00% | Ungated full-waypoint neural output is unsafe because easy degradation is far above the deployment limit. |

## Interpretation

- Full-waypoint shape heads have real auxiliary horizon value, especially t50/t100 raw-frame.
- They do not yet replace the endpoint-linear bridge all-ADE floor.
- No new composer deployment switch is allowed because endpoint-linear-vs-full-waypoint common validation evidence is missing.
- The honest deployable policy remains M3W-Neural v1 endpoint-linear bridge / Stage37-teacher floor, with full-waypoint evidence reported as auxiliary.
- This is not true 3D, not metric, not seconds-level, not Stage5C, and not SMC.
