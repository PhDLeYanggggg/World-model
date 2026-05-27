# Stage42-HU T100 Runtime Batch Replay Sufficiency Audit

- source: `fresh_audit_from_stage42_hr_hs_ht_artifacts`
- generated_at_utc: `2026-05-27T20:07:15.615054+00:00`
- git_commit: `b2c819e`
- input_hash: `3fef8a81de3f6d3f24b5eae5d1e1fb147a20edf65033d5d20c2a6977f60734f6`
- gate: `17 / 17`
- verdict: `stage42_hu_t100_runtime_batch_replay_sufficiency_pass_with_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HU 审计 Stage42-HR/HS/HT 是否足以支持真实 row-level runtime batch replay。
- HT 已有 callable runtime API 和 smoke replay，但 HR/HS/HT artifact 不包含 per-row candidate/floor/selected_xy arrays。
- 因此 real batch replay 当前标记为 not_run，而不是包装成完成。
- future waypoints / endpoints 只能作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Sufficiency Result

- runtime_api_ready: `True`
- frozen_policy_ready: `True`
- row_level_batch_replay_ready: `False`
- real_batch_replay_status: `not_run`
- blocker: `missing_row_level_candidate_floor_selected_arrays`

## Required Row Cache Fields

- `row_id`
- `split`
- `domain`
- `source_file`
- `scene_id_optional`
- `frame_id`
- `agent_id`
- `horizon`
- `candidate_xy_predicted_rollout`
- `floor_xy_train_horizon_causal_rollout`
- `original_selected_xy_from_stage42_hr_optional`
- `candidate_switch_optional`
- `future_xy_label_eval_only_optional`
- `normalizer_optional`
- `hard_failure_label_optional`
- `easy_label_optional`

## Inherited Frozen Metrics

| metric | value |
| --- | ---: |
| all | 27.72% |
| t50 | 26.99% |
| t100 raw diagnostic | 6.79% |
| hard/failure | 25.93% |
| easy degradation | -32.33% |
| t100 easy degradation | -0.31% |

## Interpretation

- Stage42-HT should be described as runtime API + smoke replay, not real batch replay.
- A real batch replay requires row-level candidate/floor/selected rollout arrays and row identifiers.
- This audit is useful precisely because it prevents overclaiming deployment reproducibility evidence.
