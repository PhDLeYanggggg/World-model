# Stage42-JX Current Paper Evidence Refresh

- source: `fresh_stage42_jx_current_paper_evidence_refresh`
- generated_at_utc: `2026-05-29T07:10:14.956184+00:00`
- git_commit: `4cc6aea`
- input_hash: `d039c4121922aaf3db059c1df9ddff0b661ed5467e964d3263f52bfc0e0b5a40`
- gate: `15 / 15`
- verdict: `stage42_jx_current_paper_evidence_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JX 将 JT/JU/JV/JW 的当前证据同步到 paper package；不训练、不调 threshold、不新增下载或转换。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Inputs

| artifact | source | verdict |
| --- | --- | --- |
| JT claim refresh | `fresh_stage42_jt_current_module_claim_refresh` | `stage42_jt_current_module_claim_refresh_pass` |
| JU reviewer replay | `fresh_stage42_ju_current_reviewer_replay_package` | `stage42_ju_current_reviewer_replay_package_pass` |
| JV source slice matrix | `fresh_stage42_jv_source_slice_evidence_matrix_from_cached_verified_row_cache` | `stage42_jv_source_slice_evidence_matrix_pass` |
| JW teacher floor audit | `fresh_stage42_jw_teacher_floor_necessity_slice_audit` | `stage42_jw_teacher_floor_necessity_slice_audit_pass` |
| JS context closure | `fresh_stage42_js_source_context_gain_harm_closure` | `stage42_js_source_context_gain_harm_closure_pass` |

## Current Paper Evidence

- rows/domains/source-files: `47458` / `2` / `3`
- domains: `['TrajNet', 'UCY']`; horizons: `['10', '25', '50', '100']`
- all/t50/t100raw/hard ADE improvement: `29.15%` / `24.70%` / `19.63%` / `28.73%`
- easy degradation: `0.00%`
- weak source files: `[]`

## Teacher/Floor Necessity

- switch/fallback rows: `33355` / `14103`
- fallback exact floor rate: `1.000000`
- hard/failure switch rate vs easy switch rate: `0.729644` / `0.412616`
- guarded t50 relaxation safety: `True` with t50 `28.97%`
- global floor removal allowed: `False`; floor-free neural deployable: `False`

## Supported Claims

- Protected source-level full-waypoint row-cache evidence is positive across the current TrajNet+UCY row cache under safe-switch/floor protection.
- Source/domain/horizon/slice decomposition is paper-usable, but it remains dataset-local/raw-frame evidence.
- The teacher/floor is a necessary deployability mechanism: fallback rows remain exact-floor and global floor-free neural deployment is forbidden.
- Baseline-family rollout context remains the strongest current source-level driver; history/motion-goal signals are bounded auxiliary evidence.

## Blocked Claims

- scene_goal_independent_main_claim
- neighbor_interaction_independent_main_claim
- JEPA_downstream_main_claim
- Transformer_independent_main_claim
- sequence_graph_t50_t100_independent_main_claim
- ungated_full_waypoint_deployment
- floor_free_neural_deployment
- global_teacher_floor_removal
- metric_seconds_or_true3d_claim
- foundation_world_model_claim
- Stage5C_execution_claim
- SMC_readiness_claim
- broad_source_level_generalization_without_terms_or_new_sources

## Source / Time / Metric Blockers

- source_terms: `{'terms_accepted_targets': 0, 'conversion_ready_targets': 0, 'converted_datasets_now': 0, 'evaluated_datasets_now': 0}`
- time_geometry: `{'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'user_action_required': True}`

## Gate

| gate | pass |
| --- | ---: |
| `jt_claim_refresh_passed` | `True` |
| `ju_reviewer_replay_package_passed` | `True` |
| `jv_source_slice_matrix_passed` | `True` |
| `jw_teacher_floor_audit_passed` | `True` |
| `source_slice_evidence_included` | `True` |
| `teacher_floor_necessity_included` | `True` |
| `paper_supported_claims_nonempty` | `True` |
| `independent_context_claims_blocked` | `True` |
| `ungated_and_floor_free_blocked` | `True` |
| `source_terms_and_time_blockers_preserved` | `True` |
| `paper_artifact_updates_listed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_3d_foundation` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Paper Wording Decision

- Center the paper package on protected source-level row-cache/full-waypoint evidence, safe-switch behavior, and teacher-floor necessity. Keep independent scene/goal, neighbor/interaction, JEPA, Transformer, floor-free neural, metric/seconds, true-3D, foundation, Stage5C, and SMC claims blocked.
