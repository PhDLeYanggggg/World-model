# Stage42-CJ Goal/Scene Gated Expert Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T17:56:15.141694+00:00`
- git_commit: `a79ecec`
- input_hash: `380517090f9b33b3ac33db99e6dd774feb63122697c4ab4a0759b8d5bdc9e9f6`
- gate: `10 / 10`
- verdict: `stage42_cj_goal_scene_gated_expert_pass_diagnostic_no_overclaim`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CJ 是 validation-only goal/scene gated expert audit，不是 metric 或 seconds-level 结果。
- 本阶段专门测试 Stage42-CI 标出的 mixed goal/scene context 是否可被保守 gate 修复为增量贡献。
- 每个 candidate 都重新训练 ridge full-waypoint probe，并在 validation 上重新选 safe policy；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-CI found goal/scene context is mixed and not a main claim.
- Stage42-CJ tests one concrete repair: a validation-only gated goal/scene expert.
- Goal/scene candidates are allowed only if they beat the baseline-family control on validation by a margin and preserve easy cases.
- Test metrics are used only once after validation selection.

## Variant Metrics

| variant | features | val score | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_family_control` | 35 | 2.007275 | 0.287773 | 0.315425 | 0.142825 | 0.275812 | -0.324186 | 0.660626 |
| `goal_scene_only_control` | 17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| `baseline_plus_goal_scene` | 45 | 1.901516 | 0.262502 | 0.227640 | 0.142748 | 0.248648 | -0.301712 | 0.633739 |
| `baseline_plus_motion_goal_context` | 166 | 1.836829 | 0.245788 | 0.220171 | 0.143652 | 0.237494 | -0.256627 | 0.588099 |

## Validation-Only Selection

- selection: `{'source': 'fresh_run', 'selection_rule': 'choose_goal_scene_candidate_only_if_validation_score_beats_baseline_by_margin_and_easy_safe_else_fallback', 'validation_margin': 0.01, 'easy_limit': 0.02, 'baseline_variant': 'baseline_family_control', 'selected_variant': 'baseline_family_control', 'selected_score': 2.007274993106131, 'considered_goal_scene_candidates': [{'variant': 'baseline_plus_goal_scene', 'validation_score': 1.9015158293196792, 'validation_margin_vs_baseline': -0.10575916378645167, 'validation_easy_degradation': -0.3266141397539628, 'passes_validation_gate': False}, {'variant': 'baseline_plus_motion_goal_context', 'validation_score': 1.836828805896121, 'validation_margin_vs_baseline': -0.17044618721000981, 'validation_easy_degradation': -0.2874272837926436, 'passes_validation_gate': False}], 'test_threshold_tuning': False}`
- selected_delta_vs_baseline_family_control: `{'all_improvement': 0.0, 't50_improvement': 0.0, 't100_raw_frame_diagnostic_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0, 'harm_over_fallback': 0.0}`
- goal_scene_rescue_success: `False`

## Delta Vs Baseline-Family Control

| variant | delta all | delta t50 | delta hard/failure | delta easy |
| --- | ---: | ---: | ---: | ---: |
| `goal_scene_only_control` | -0.287773 | -0.315425 | -0.275812 | 0.324186 |
| `baseline_plus_goal_scene` | -0.025271 | -0.087785 | -0.027164 | 0.022474 |
| `baseline_plus_motion_goal_context` | -0.041985 | -0.095255 | -0.038318 | 0.067559 |

## Interpretation

- verdict: `goal_scene_gated_expert_not_validation_selected`
- summary: The validation-only gate did not select a goal/scene candidate over the baseline-family control. This preserves the Stage42-CI boundary: goal/scene remains mixed and not a main claim.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'goal_scene_main_claim_allowed': False, 'stage5c_executed': False, 'smc_enabled': False}`
