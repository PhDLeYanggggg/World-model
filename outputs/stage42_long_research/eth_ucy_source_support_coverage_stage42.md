# Stage42-JL ETH_UCY Source Support Coverage

- source: `fresh_stage42_jl_eth_ucy_source_support_coverage`
- generated_at_utc: `2026-05-28T18:54:29.393234+00:00`
- git_commit: `3d223be`
- input_hash: `561b6685ca9caf6929c705cce42366bf9d11bb9310704d302b1dc26851281ad8`
- gate: `11 / 11`
- verdict: `stage42_jl_eth_ucy_source_support_coverage_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JL follows Stage42-JK: row-level family selection safely refused the blocked ETH_UCY sources.
- JL audits whether those blocked sources have enough source-level geometry/history support to justify any family switch.
- Held-out source support uses past-only/current-row feature distributions for diagnostics; future labels are evaluation-only.
- No central velocity, no test endpoint goals, no test-threshold tuning, and no metric/seconds claim are used.
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `source_support_policy_not_deployable_support_blocker`
- targeted_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repaired_sources: `[]`
- still_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- unsupported_sources: `['UCY/students03/obsmat.txt']`
- in_support_but_no_safe_family_sources: `['ETH/seq_eth/obsmat.txt']`
- next_action: For unsupported or no-safe-family sources, acquire/calibrate source-specific geometry or train a dedicated source family with stronger easy-harm labels before another deployment attempt.

## Held-Out Support Metrics

| source | in support | nearest | distance | threshold | all | t50 | hard/failure | easy degradation | oracle t50 | deployable |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `True` | `UCY/zara01/obsmat.txt` | 4.230 | 4.518 | 0.00% | 0.00% | 0.00% | -0.00% | 53.80% | `False` |
| `UCY/students03/obsmat.txt` | `False` | `UCY/zara02/obsmat.txt` | 11.128 | 3.850 | 0.00% | 0.00% | 0.00% | -0.00% | 39.14% | `False` |

## Support Family Candidates

### `ETH/seq_eth/obsmat.txt`

- decision_reason: `no_nearest_source_easy_safe_family`
- selected_support_family: `None`

| rank | family | safe sources | mean all | mean t50 | mean hard | max easy | score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `constant_position` | 0/2 | -108.89% | -169.09% | -111.02% | 199.50% | -59.6212 |
| 2 | `history_decay_baseline` | 0/2 | -162.38% | -123.33% | -154.58% | 266.04% | -96.4467 |
| 3 | `neighbor_aware_decay_baseline` | 0/2 | -207.77% | -242.78% | -201.51% | 359.51% | -130.1965 |
| 4 | `damped_velocity` | 0/2 | -343.50% | -410.11% | -335.36% | 567.04% | -203.3194 |
| 5 | `constant_velocity_causal_fd` | 0/2 | -1584.03% | -2062.91% | -1607.44% | 2238.72% | -776.6976 |
| 6 | `turn_rate` | 0/2 | -1787.80% | -2301.19% | -1812.98% | 2542.79% | -870.1964 |
| 7 | `prototype_goal_directed_baseline` | 0/2 | -1779.04% | -2167.27% | -1779.72% | 3063.30% | -979.7710 |
| 8 | `constant_acceleration` | 0/2 | -1723.20% | -2149.24% | -1729.60% | 2869.56% | -986.4513 |

### `UCY/students03/obsmat.txt`

- decision_reason: `source_out_of_support`
- selected_support_family: `None`

| rank | family | safe sources | mean all | mean t50 | mean hard | max easy | score |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `constant_position` | 0/2 | -88.68% | -140.21% | -89.67% | 150.36% | -42.8653 |
| 2 | `history_decay_baseline` | 0/2 | -78.03% | -27.26% | -69.40% | 218.89% | -54.7611 |
| 3 | `neighbor_aware_decay_baseline` | 0/2 | -76.92% | -70.65% | -70.18% | 230.02% | -57.8050 |
| 4 | `damped_velocity` | 0/2 | -195.40% | -216.74% | -184.29% | 472.44% | -124.5112 |
| 5 | `constant_velocity_causal_fd` | 0/2 | -1106.74% | -1384.80% | -1128.69% | 1882.24% | -489.9861 |
| 6 | `turn_rate` | 0/2 | -1288.79% | -1595.36% | -1310.66% | 2121.04% | -562.7930 |
| 7 | `constant_acceleration` | 0/2 | -1196.50% | -1436.92% | -1206.15% | 2035.67% | -595.9579 |
| 8 | `prototype_goal_directed_baseline` | 0/2 | -1352.86% | -1546.67% | -1331.81% | 4969.38% | -1096.9102 |

## Bootstrap CI

| source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 1812 |
| `ETH/seq_eth/obsmat.txt` | `oracle_t50` | 52.99% | 53.77% | 54.57% | 5074 |
| `UCY/students03/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 21424 |
| `UCY/students03/obsmat.txt` | `oracle_t50` | 38.64% | 39.14% | 39.66% | 17529 |

## Interpretation

- JL turns the JK failure into a source-support question: a family may have oracle headroom but still be unsafe if no similar source supports it without easy harm.
- If this remains fallback-only or unsupported, the next useful work is source-specific calibration/new source acquisition rather than more global threshold search.
- This is raw-frame / dataset-local 2.5D evidence only; no metric, seconds-level, Stage5C, SMC, or foundation claim is enabled.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True, 'heldout_source_support_uses_labels': False}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
