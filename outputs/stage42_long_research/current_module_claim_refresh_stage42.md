# Stage42-JT Current Module Claim Refresh

- source: `fresh_stage42_jt_current_module_claim_refresh`
- generated_at_utc: `2026-05-29T05:39:49.906879+00:00`
- git_commit: `5a83e6d`
- input_hash: `1bcded06a5f79c62aac6d554e8a1347e3d0e108aa5b394ae5bbad6c6d1009156`
- gate: `15 / 15`
- verdict: `stage42_jt_current_module_claim_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JT 汇总当前 HEAD 上 fresh replay 的 IV/IW row-cache、AO incremental ablation、JS gain/harm closure。
- Stage42-JT 不重新调 threshold，不使用 test 指标调参，不把 synthesis 当训练结果。
- future waypoints / endpoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Input Status

| input | source | verdict |
| --- | --- | --- |
| `iv` | `fresh_run_current_source_level_row_cache_and_cached_verified_stage42v_ucy` | `stage42_iv_source_level_row_cache_integration_pass` |
| `iw` | `fresh_run_row_cache_mechanism_audit_from_cached_verified_stage42iv_cache` | `stage42_iw_row_cache_mechanism_audit_pass` |
| `ao` | `fresh_run` | `stage42_ao_incremental_component_evidence_partial_or_negative` |
| `js` | `fresh_stage42_js_source_context_gain_harm_closure` | `stage42_js_source_context_gain_harm_closure_pass` |

## Row-Cache Evidence

- rows: `47458`; domains: `{'TrajNet': 37918, 'UCY': 9540}`
- ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`
- easy_degradation: `0.000000`
- t50 bootstrap CI: `[0.24292968809604645, 0.25138823240995406]`

## Mechanism Evidence

- switch_rows: `33355`; fallback_rows: `14103`; switch_rate: `0.702832`
- fallback_exact_floor_rate: `1.000000`
- full_waypoint_rate: `0.675460`; mean_valid_waypoints_per_row: `3.350921`

## Incremental Context Refresh

- AO component evidence verdict: `stage42_ao_context_components_not_independently_supported`
- positive standalone context variants: `['history_only', 'motion_goal_context']`
- positive incremental context variants after baseline-family: `[]`
- JS t50 blocker: `router_under_switches_despite_headroom`
- JS t100 blocker: `weak_predictive_signal_or_baseline_family_dominance`

## Allowed Claims

- protected source-level full-waypoint row-cache is positive on TrajNet+UCY under safe-switch/floor protection
- safe-switch and teacher/floor fallback are directly supported by row-cache mechanism evidence
- baseline-family rollout context remains the strongest current source-level driver
- history-only and motion-goal-context have standalone positive signal under AO, but only as bounded evidence

## Blocked Independent Claims

- incremental_context_after_baseline_family
- scene_goal_independent_main_claim
- neighbor_interaction_independent_main_claim
- sequence_graph_t50_t100_independent_main_claim
- JEPA_downstream_main_claim
- Transformer_independent_main_claim
- ungated_full_waypoint_deployment
- metric_seconds_or_true3d_claim

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input_absent': True, 'future_waypoint_input_absent': True, 'central_velocity_absent': True, 'test_endpoint_goals_absent': True, 'test_threshold_tuning_absent': True, 'future_labels_eval_only': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Verification

- focused pytest: `.venv-pytorch/bin/python -m pytest tests/test_stage42_source_level_incremental_ablation.py tests/test_stage42_current_module_claim_refresh.py -> 7 passed in 0.76s`
- full pytest: `.venv-pytorch/bin/python -m pytest tests -> 1196 passed in 827.27s (0:13:47)`

## Interpretation

- Stage42-JT keeps the strongest current claim as protected row-cache/full-waypoint evidence plus safe-switch/teacher-floor behavior.
- It preserves the negative result that context modules do not yet add incremental value after baseline-family rollout context under the current source-level ridge protocol.
- It allows bounded wording for history standalone signal, but blocks scene/goal, neighbor/interaction, JEPA, Transformer, and sequence/graph t50/t100 independent main claims.
