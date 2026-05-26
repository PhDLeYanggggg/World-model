# Stage42-AP Proposed Source-Level Residual Context

- source: `fresh_run`
- generated_at_utc: `2026-05-26T08:35:37.325755+00:00`
- git_commit: `54975d5`
- input_hash: `bca26b572eb2e8ea897faea6ad8839f26ecb00cc2ac9097578857e24230cd69c`
- gate: `8 / 9`
- verdict: `stage42_ap_residual_context_evidence_partial_or_negative`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AP 是 proposed source-level split residual-context retraining，不是 metric 或 seconds-level 结果。
- 第一阶段只用 baseline-family rollout context 训练；第二阶段只让 context features 预测第一阶段剩余误差。
- 所有 residual alpha / lambda / safety policy 均在 validation 上选择；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-AO found that baseline-family rollout context dominates direct ridge evidence.
- Stage42-AP uses a two-stage residual design: first train baseline-family-only, then train history/goal/neighbor context on the remaining full-waypoint residual.
- If context modules have residual world-state information, they should improve over the first-stage baseline-family model without using future inputs.

## Baseline-Family First Stage

- feature_count: `35`
- best_lambda: `10.0`
- protected_metric: `{'rows': 47458, 'all_improvement': 0.2877734037648393, 't10_improvement': 0.47618706265543653, 't25_improvement': 0.31611808582214795, 't50_improvement': 0.31542535139554606, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.2758122379367457, 'easy_degradation': -0.32418582524688455, 'switch_rate': 0.6606262379367019, 'harm_over_fallback': -0.13253112673847436}`

## Residual Variants

| variant | features | alpha | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_history` | 128 | 0.25 | 0.287853 | 0.314424 | 0.144250 | 0.276019 | -0.321191 | 0.000080 | -0.001001 | 0.000206 |
| `residual_goal` | 17 | 0.25 | 0.288079 | 0.314493 | 0.142683 | 0.276448 | -0.323117 | 0.000305 | -0.000932 | 0.000636 |
| `residual_neighbor` | 12 | 0.25 | 0.288074 | 0.315304 | 0.143145 | 0.276108 | -0.324798 | 0.000301 | -0.000121 | 0.000296 |
| `residual_history_goal` | 138 | 0.25 | 0.288408 | 0.314000 | 0.143886 | 0.276683 | -0.320631 | 0.000635 | -0.001425 | 0.000871 |
| `residual_history_neighbor` | 128 | 0.25 | 0.287853 | 0.314424 | 0.144250 | 0.276019 | -0.321191 | 0.000080 | -0.001001 | 0.000206 |
| `residual_goal_neighbor` | 22 | 0.25 | 0.265273 | 0.228697 | 0.142917 | 0.250277 | -0.316825 | -0.022500 | -0.086728 | -0.025535 |
| `residual_history_goal_neighbor` | 138 | 0.25 | 0.288408 | 0.314000 | 0.143886 | 0.276683 | -0.320631 | 0.000635 | -0.001425 | 0.000871 |

## Interpretation

- positive_residual_context_variants: `[]`
- residual_context_verdict: `stage42_ap_residual_context_not_supported`

- Stage42-AP did not find residual context value beyond baseline-family rollout context under this ridge residual protocol.
- This is a boundary result: it constrains paper claims and motivates a stronger neural/graph context model rather than overclaiming ridge context modules.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
