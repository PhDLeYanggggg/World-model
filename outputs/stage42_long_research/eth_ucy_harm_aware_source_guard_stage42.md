# Stage42-JH ETH_UCY Harm-Aware Source Guard

- source: `fresh_stage42_jh_eth_ucy_harm_aware_source_guard`
- generated_at_utc: `2026-05-28T16:52:08.009072+00:00`
- git_commit: `eac8e39`
- input_hash: `18d97d4d1148eee2d5ef7e224cfd461a7ff6ab95c41474e2367e566046c0d315`
- gate: `9 / 9`
- verdict: `stage42_jh_eth_ucy_harm_aware_source_guard_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JH 是 ETH_UCY harm-aware source guard：它用 train sources 的 switch gain/harm labels 学 predicted-gain guard。
- threshold/cap 只在 non-heldout validation source 上选择；held-out source 不参与调参。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `eth_ucy_harm_aware_guard_partial_support`
- deployable_heldout_sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt', 'UCY/zara02/obsmat.txt']`
- blocked_heldout_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- easy_repaired_sources: `['UCY/zara02/obsmat.txt']`
- next_action: Promote only sources that remain positive and easy-safe under harm-aware source-CV; keep other ETH_UCY sources fallback-only.

## Source-CV Fold Metrics

| heldout source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | base easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | 21598 | 0.58% | -32.47% | 7.74% | 0.63% | -11.82% | 90.99% | -11.79% |
| `ETH/seq_hotel/obsmat.txt` | 16611 | 8.64% | 15.05% | 0.00% | 8.70% | -15.89% | 25.81% | -15.89% |
| `UCY/students03/obsmat.txt` | 70585 | 9.09% | 9.03% | 5.89% | 10.02% | 10.78% | 53.38% | 24.71% |
| `UCY/zara01/obsmat.txt` | 16103 | 12.50% | 17.97% | 0.00% | 11.43% | -24.69% | 64.82% | -24.69% |
| `UCY/zara02/obsmat.txt` | 25901 | 30.39% | 38.99% | 16.24% | 30.27% | -2.52% | 72.78% | 89.33% |

## Bootstrap CI

| heldout source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.07% | 0.57% | 1.05% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | -34.06% | -32.49% | -30.87% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 7.49% | 7.74% | 8.00% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.08% | 0.62% | 1.14% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | -18.01% | -13.48% | -9.12% | 1812 |
| `ETH/seq_hotel/obsmat.txt` | `all` | 8.33% | 8.64% | 8.95% | 16611 |
| `ETH/seq_hotel/obsmat.txt` | `t50` | 14.61% | 15.06% | 15.47% | 3994 |
| `ETH/seq_hotel/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 2560 |
| `ETH/seq_hotel/obsmat.txt` | `hard_failure` | 8.39% | 8.70% | 9.00% | 14776 |
| `ETH/seq_hotel/obsmat.txt` | `easy_degradation` | -20.90% | -18.82% | -16.71% | 5755 |
| `UCY/students03/obsmat.txt` | `all` | 8.86% | 9.09% | 9.31% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 8.78% | 9.04% | 9.29% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 5.54% | 5.88% | 6.17% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 9.80% | 10.02% | 10.27% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 8.88% | 9.74% | 10.57% | 21424 |
| `UCY/zara01/obsmat.txt` | `all` | 12.22% | 12.50% | 12.78% | 16103 |
| `UCY/zara01/obsmat.txt` | `t50` | 17.63% | 17.97% | 18.30% | 3988 |
| `UCY/zara01/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 3251 |
| `UCY/zara01/obsmat.txt` | `hard_failure` | 11.16% | 11.43% | 11.70% | 14187 |
| `UCY/zara01/obsmat.txt` | `easy_degradation` | -35.59% | -32.78% | -29.95% | 1352 |
| `UCY/zara02/obsmat.txt` | `all` | 29.89% | 30.39% | 30.91% | 25901 |
| `UCY/zara02/obsmat.txt` | `t50` | 37.81% | 39.02% | 40.14% | 6422 |
| `UCY/zara02/obsmat.txt` | `t100_raw_frame_diagnostic` | 15.64% | 16.25% | 16.86% | 5433 |
| `UCY/zara02/obsmat.txt` | `hard_failure` | 29.71% | 30.26% | 30.80% | 20861 |
| `UCY/zara02/obsmat.txt` | `easy_degradation` | -6.44% | -2.66% | 0.87% | 6214 |

## Interpretation

- This is an actual harm-aware repair attempt, not an overclaim: if a source remains blocked, default deployment stays fallback-only.
- This remains dataset-local/raw-frame 2.5D evidence and does not enable Stage5C or SMC.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
