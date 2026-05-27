# Stage42-DC Context Switchability / Gain-Harm Gate

- source: `fresh_run`
- generated_at_utc: `2026-05-27T01:53:35.163953+00:00`
- git_commit: `3de43c8`
- gate: `15 / 15`
- verdict: `stage42_dc_context_switchability_gate_pass`
- decision: `context_switchability_not_supported`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DC 是 context switchability / gain-harm gate，不是 residual waypoint retrain。
- 本阶段响应 Stage42-DB 的 no-go：换监督目标，训练 context 是否应该切换的 gain predictor。
- future endpoints / waypoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Baseline-Family Control

- protected_metric: `{'rows': 47458, 'all_improvement': 0.2877734037648393, 't10_improvement': 0.47618706265543653, 't25_improvement': 0.31611808582214795, 't50_improvement': 0.31542535139554606, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.2758122379367457, 'easy_degradation': -0.32418582524688455, 'switch_rate': 0.6606262379367019, 'harm_over_fallback': -0.13253112673847436}`

## Context Switchability Candidates

| candidate | lambda | all | t50 | hard/failure | easy | switch | delta all | delta t50 | delta hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_plus_goal_scene` | 0.10 | 0.287471 | 0.315182 | 0.275473 | -0.323767 | 0.038834 | -0.000302 | -0.000243 | -0.000339 |
| `baseline_plus_scalar_neighbor` | 100.00 | 0.287773 | 0.315425 | 0.275812 | -0.324186 | 0.003097 | 0.000000 | 0.000000 | 0.000000 |
| `baseline_plus_knn_graph` | 10.00 | 0.288141 | 0.315351 | 0.276236 | -0.326574 | 0.034262 | 0.000368 | -0.000074 | 0.000424 |
| `baseline_plus_graph_goal` | 0.10 | 0.287691 | 0.315264 | 0.275720 | -0.323822 | 0.035969 | -0.000082 | -0.000162 | -0.000092 |
| `baseline_plus_graph_history_scalar` | 0.10 | 0.288071 | 0.315348 | 0.276155 | -0.326212 | 0.043133 | 0.000298 | -0.000078 | 0.000343 |

## Selected Policy

- selected_candidate: `baseline_plus_knn_graph`
- test_metric: `{'rows': 47458, 'all_improvement': 0.28814146384660755, 't10_improvement': 0.47629831934874767, 't25_improvement': 0.3183830229714819, 't50_improvement': 0.31535142784214254, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.27623622833705697, 'easy_degradation': -0.3265738537676721, 'switch_rate': 0.034261873656706986, 'harm_over_fallback': -0.1327006330816806}`
- delta_vs_baseline_family_control: `{'all_improvement': 0.0003680600817682622, 't50_improvement': -7.392355340352097e-05, 'hard_failure_improvement': 0.00042399040031126933, 'easy_degradation': -0.0023880285207875662, 'switch_rate': -0.626364364279995}`
- context_switchability_supported: `False`

## Interpretation

- Stage42-DC changed the supervision target to gain/harm switchability, but still did not find a safe positive context increment beyond baseline-family control.
- This is fresh training/evaluation of a gain-harm gate; it is not Stage5C, not SMC, not metric/seconds-level, and not true 3D evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'gain_label_train_only_for_model_fit': True, 'validation_only_threshold_selection': True, 'test_threshold_tuning': False, 'central_velocity': False, 'test_endpoint_goals': False, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
