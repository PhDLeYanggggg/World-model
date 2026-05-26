# Stage42-AV Baseline-Family Mechanism Robustness Audit

- source: `cached_verified_from_stage42_au`
- generated_at_utc: `2026-05-26T10:29:23.787411+00:00`
- git_commit: `d0ffad4`
- input_hash: `82ef757fecce0c25663b5b73a1296caaee490f5079fbe8ca72024d4826afec1e`
- gate: `12 / 12`
- verdict: `stage42_av_baseline_family_robustness_pass_with_limits`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AV 是 Stage42-AU baseline-family mechanism 的 robustness / weak-slice audit。
- 本审计不重新用 test 调 threshold，不执行 Stage5C，不启用 SMC。
- 如果某个 domain 缺 validation support，必须写成 blocker 或 floor-only，不得包装成 positive transfer。
- raw-frame / dataset-local 不能写成 metric 或 seconds-level。

## Global Bootstrap Stability

| variant | all low | t50 low | t100 low | hard low | easy high |
| --- | ---: | ---: | ---: | ---: | ---: |
| `family_baseline_rel_only` | 0.270428 | 0.233072 | 0.136035 | 0.254235 | -0.444036 |
| `baseline_family_all` | 0.284243 | 0.309806 | 0.137082 | 0.271961 | -0.459376 |

## Domain Support

| domain | train | val | test | all | t50 | hard | easy | switch | blocker |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet` | 75287 | 7685 | 37918 | 0.375356 | 0.403711 | 0.362941 | -0.382893 | 0.826837 | `none` |
| `UCY` | 56763 | 0 | 9540 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | `no_validation_rows_for_domain_policy_selection_floor_only` |

## Horizon Support

| horizon | rows | all | horizon metric | hard | easy | switch | weaknesses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10` | 15402 | 0.476187 | 0.476187 | 0.485011 | -0.422168 | 0.675951 | `none` |
| `25` | 13470 | 0.316118 | 0.316118 | 0.278626 | -0.432191 | 0.765553 | `none` |
| `50` | 11538 | 0.315425 | 0.315425 | 0.315425 | -0.211290 | 0.766077 | `none` |
| `100` | 7048 | 0.142825 | 0.142825 | 0.142825 | 0.028497 | 0.253973 | `easy_degradation_over_2pct` |

## Summary

- positive_domains: `['TrajNet']`
- floor_only_or_blocked_domains: `['UCY']`
- weak_horizons: `['100']`
- uniform_domain_claim_allowed: `False`
- uniform_horizon_claim_allowed: `False`
- paper_claim: baseline-family rollout context is statistically stable globally and on TrajNet, but uniform source-level domain/horizon claims remain disallowed because UCY is floor-only under this split and t100 has easy-safety weakness.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`
