# Stage42-DI Group-Consistency Full-Waypoint Repair

- source: `fresh_stage42_di_group_consistency_full_waypoint_repair`
- generated_at_utc: `2026-05-26T23:55:04.962245+00:00`
- git_commit: `c11f73d`
- gate: `17 / 17`
- verdict: `stage42_di_group_consistency_full_waypoint_repair_pass_promotable`
- decision: `promote_stage42_di_group_consistency_full_waypoint_repair`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DI 针对 Stage42-DE/DH 的 full-waypoint all-agent proximity / group-consistency blocker。
- group-consistency repair 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- validation 选择 repair policy；test 只评一次。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Selected Repair

- candidate: `{'mode': 'repel_unsafe', 'min_sep': 0.08, 'margin': 0.0, 'strength': 0.5}`
- val_score: `1.910356`
- val_metric: `{'rows': 23788, 'all_improvement': 0.44156489895599793, 't10_improvement': 0.6933195833166081, 't25_improvement': 0.3064878897392037, 't50_improvement': 0.47594484424243677, 't100_raw_frame_diagnostic_improvement': 0.3295930766716896, 'hard_failure_improvement': 0.4372599087028438, 'easy_degradation': -0.28703887951299933, 'switch_rate': 0.8334874726752984, 'harm_over_fallback': -0.23895228406192529}`
- val_diagnostics: `{'unsafe_rows': 1200, 'unsafe_rate': 0.05044560282495376, 'base_switch_rate': 0.8334874726752984, 'final_switch_rate': 0.8334874726752984, 'base_near_005': 0.015512022868673281, 'final_near_005': 0.004371952244829325, 'floor_near_005': 0.012737514713300825, 'base_p05_min_distance': 0.07308731743085398, 'final_p05_min_distance': 0.07964797237636109, 'floor_p05_min_distance': 0.0760265211798491}`

## Test Once Metrics vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | near@0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage42-AM rebuilt floor-protected source-level | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 58.81% | 1.94% |
| Stage42-DI group-consistency repair | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 58.81% | 1.38% |

## Delta vs Prior Evidence

- delta_vs_stage42_am all/t50/t100/hard/easy: `0.14%` / `0.35%` / `-0.02%` / `0.14%` / `0.03%`
- delta_vs_stage42_cq all/t50/t100/hard/easy: `22.94%` / `21.30%` / `10.87%` / `21.96%` / `-25.88%`
- delta_vs_stage42_dh all/t50/t100/hard/easy: `-0.79%` / `0.23%` / `0.01%` / `0.15%` / `3.60%`

## Proximity Diagnostics

- base_near_005: `1.94%`
- final_near_005: `1.38%`
- floor_near_005: `2.24%`
- base_p05_min_distance: `0.07437689768396878`
- final_p05_min_distance: `0.07770240407545181`
- floor_p05_min_distance: `0.07354961919220622`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.243986 | 0.247166 | 0.250290 | 47458 |
| `t50` | 0.219272 | 0.223671 | 0.227600 | 11538 |
| `t100_raw_frame_diagnostic` | 0.137466 | 0.143576 | 0.149030 | 7048 |
| `hard_failure` | 0.235347 | 0.238849 | 0.242336 | 35076 |
| `easy_degradation` | -0.359184 | -0.344667 | -0.330070 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 32.24% | 28.62% | 19.03% | 31.43% | -30.27% | 73.61% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-DI changes the repair mechanism from scalar loss weighting to explicit group-consistency / proximity-aware repair over predicted full-waypoint rollouts.
- Repair selection is validation-only; test is evaluated once.
- Promotion requires improving Stage42-AM on all and hard/failure, preserving easy, and not worsening near@0.05 relative to rebuilt Stage42-AM selected rollout.
- If not promotable, keep Stage42-AM/CQ or Stage37/teacher safety floor as deployable floor and treat DI as diagnostic evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'group_features_predicted_rollout_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'validation_only_policy_selection': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
