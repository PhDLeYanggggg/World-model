# Stage42-JF Source-Rotation Easy-Guard Repair

- source: `fresh_stage42_jf_source_rotation_easy_guard_repair`
- generated_at_utc: `2026-05-28T16:00:35.387888+00:00`
- git_commit: `4d7a81d`
- input_hash: `275c53cc9c0d5b2bc3fe99da2b47f323cc7475d6bbad0bcdfe28dfdcf749dc09`
- gate: `9 / 9`
- verdict: `stage42_jf_source_rotation_easy_guard_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JF 是 source-rotation easy-safety repair：它尝试用 validation-only switch budget 降低 held-out ETH_UCY easy harm。
- switch budget 在非 held-out validation rows 上选择；held-out domain test 不参与阈值选择。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `easy_guard_repair_partial_domain_bounded`
- deployable_heldout_domains_after_easy_guard: `['TrajNet', 'UCY']`
- easy_repaired_domains: `[]`
- still_easy_blocked_domains: `['ETH_UCY']`
- next_action: If ETH_UCY remains blocked, treat ETH_UCY as fallback-only until source-specific calibration, safer hard/easy detector, or additional held-out ETH/UCY sources are available.

## Rotation Metrics

| heldout domain | cap | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | base easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 1.00 | 25.23% | 21.07% | 18.64% | 26.08% | 27.83% | 90.11% | 27.83% |
| `TrajNet` | 0.75 | 30.13% | 39.29% | 14.05% | 29.19% | -25.02% | 61.26% | -24.27% |
| `UCY` | 0.75 | 21.86% | 23.73% | 10.86% | 20.19% | -21.09% | 43.99% | -21.09% |

## Bootstrap CI

| heldout domain | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `all` | 25.01% | 25.23% | 25.45% | 150798 |
| `ETH_UCY` | `t50` | 20.42% | 21.08% | 21.67% | 37007 |
| `ETH_UCY` | `t100_raw_frame_diagnostic` | 18.36% | 18.63% | 18.89% | 29328 |
| `ETH_UCY` | `hard_failure` | 25.84% | 26.08% | 26.31% | 123289 |
| `ETH_UCY` | `easy_degradation` | 20.80% | 21.76% | 22.84% | 36557 |
| `TrajNet` | `all` | 29.92% | 30.13% | 30.34% | 120890 |
| `TrajNet` | `t50` | 38.89% | 39.29% | 39.68% | 28692 |
| `TrajNet` | `t100_raw_frame_diagnostic` | 13.82% | 14.04% | 14.27% | 17408 |
| `TrajNet` | `hard_failure` | 28.96% | 29.19% | 29.43% | 86427 |
| `TrajNet` | `easy_degradation` | -34.30% | -33.38% | -32.41% | 39146 |
| `UCY` | `all` | 21.62% | 21.86% | 22.10% | 66303 |
| `UCY` | `t50` | 23.28% | 23.71% | 24.22% | 16263 |
| `UCY` | `t100_raw_frame_diagnostic` | 10.51% | 10.87% | 11.20% | 10008 |
| `UCY` | `hard_failure` | 19.90% | 20.18% | 20.46% | 45917 |
| `UCY` | `easy_degradation` | -27.82% | -26.72% | -25.65% | 20798 |

## Interpretation

- If a domain remains easy-blocked after this validation-only cap, the honest deployment rule is fallback-only for that domain.
- This repair does not make metric/seconds claims and does not execute Stage5C or SMC.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'domain_specific_test_thresholds': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
