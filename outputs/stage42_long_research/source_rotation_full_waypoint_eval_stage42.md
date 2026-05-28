# Stage42-JE Source-Rotation Full-Waypoint Evaluation

- source: `fresh_stage42_je_source_rotation_full_waypoint_eval`
- generated_at_utc: `2026-05-28T15:38:21.630925+00:00`
- git_commit: `625430c`
- input_hash: `d96be6a37fc87b9ec62ef5d79d9ab3542298b8bbc7295c45b58c9d62c9e17e8c`
- gate: `14 / 14`
- verdict: `stage42_je_source_rotation_full_waypoint_eval_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JE 是 leave-one-domain source-rotation full-waypoint evaluation，不是 metric 或 seconds-level 结果。
- 每个 rotation 将一个 external domain 整域留作 test；train/val 只来自其他 domains。
- 策略只在 validation 上选 horizon-level thresholds；test domain 不参与 threshold selection。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Stage Exists

- Stage42-JC established the current strongest source-level row-cache evidence, while Stage42-JD kept metric/seconds claims blocked.
- Stage42-JE asks a stricter question: if an entire external domain is held out, can a domain-invariant full-waypoint probe still help under validation-selected safety?
- This is not used to tune the existing main policy; it is a boundary check for cross-domain world-model claims.

## Summary

- decision: `source_rotation_positive_but_not_global_deployable`
- positive_heldout_domains: `['TrajNet', 'UCY']`
- deployable_heldout_domains: `['TrajNet', 'UCY']`
- next_action: Use this rotation result as the cross-domain boundary: promote only domains that are positive under held-out source rotation; otherwise continue source-specific data/calibration expansion.

## Held-Out Domain Rotations

| heldout domain | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | policy slices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 150798 | 25.23% | 21.07% | 18.64% | 26.08% | 27.83% | 90.11% | 4 |
| `TrajNet` | 120890 | 30.11% | 39.29% | 14.05% | 29.21% | -24.27% | 66.11% | 4 |
| `UCY` | 66303 | 21.86% | 23.73% | 10.86% | 20.19% | -21.09% | 43.99% | 4 |

## Bootstrap CI

| heldout domain | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `all` | 25.00% | 25.22% | 25.44% | 150798 |
| `ETH_UCY` | `t50` | 20.42% | 21.07% | 21.71% | 37007 |
| `ETH_UCY` | `t100_raw_frame_diagnostic` | 18.39% | 18.65% | 18.90% | 29328 |
| `ETH_UCY` | `hard_failure` | 25.85% | 26.08% | 26.32% | 123289 |
| `ETH_UCY` | `easy_degradation` | 20.81% | 21.78% | 22.78% | 36557 |
| `TrajNet` | `all` | 29.91% | 30.11% | 30.33% | 120890 |
| `TrajNet` | `t50` | 38.86% | 39.29% | 39.69% | 28692 |
| `TrajNet` | `t100_raw_frame_diagnostic` | 13.83% | 14.05% | 14.29% | 17408 |
| `TrajNet` | `hard_failure` | 28.98% | 29.22% | 29.45% | 86427 |
| `TrajNet` | `easy_degradation` | -32.96% | -32.05% | -31.12% | 39146 |
| `UCY` | `all` | 21.61% | 21.87% | 22.15% | 66303 |
| `UCY` | `t50` | 23.28% | 23.74% | 24.18% | 16263 |
| `UCY` | `t100_raw_frame_diagnostic` | 10.48% | 10.85% | 11.23% | 10008 |
| `UCY` | `hard_failure` | 19.92% | 20.19% | 20.45% | 45917 |
| `UCY` | `easy_degradation` | -27.83% | -26.73% | -25.68% | 20798 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'domain_specific_test_thresholds': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- A positive held-out rotation supports cross-domain raw-frame transfer only for that held-out domain and only under this domain-invariant policy.
- A negative held-out rotation means the main Stage42 result must remain source/protocol bounded; it must not be written as foundation-scale or metric world-model generalization.
