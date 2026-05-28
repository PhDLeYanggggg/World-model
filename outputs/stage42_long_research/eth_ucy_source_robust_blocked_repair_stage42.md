# Stage42-JI ETH_UCY Source-Robust Blocked-Source Repair

- source: `fresh_stage42_ji_eth_ucy_source_robust_blocked_repair`
- generated_at_utc: `2026-05-28T17:29:09.576211+00:00`
- git_commit: `68bcc10`
- input_hash: `44a0adb67ee5111fded3f7cb504d9ccc9f1df2a5067d6f027fc75be8a5a42b1b`
- gate: `10 / 10`
- verdict: `stage42_ji_eth_ucy_source_robust_blocked_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JI targets only the Stage42-JH blocked ETH_UCY sources with source-robust support checks.
- Candidate policies are selected on non-heldout train/validation sources only; held-out source is evaluated once.
- future waypoints / endpoints are labels/eval only, never inference inputs.
- No central velocity, no test endpoint goals, and no test-threshold tuning are used.
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `eth_ucy_blocked_sources_still_blocked`
- targeted_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repaired_sources: `[]`
- still_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- easy_improved_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- next_action: Promote only repaired sources; keep still-blocked ETH_UCY sources fallback-only and investigate source-specific geometry/history support.

## Target Source Test Metrics

| heldout source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | JH all | JH t50 | JH easy | deployable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | 21598 | 0.97% | -31.92% | 7.84% | 1.05% | -14.48% | 86.58% | 0.58% | -32.47% | -11.82% | `False` |
| `UCY/students03/obsmat.txt` | 70585 | 5.42% | 3.69% | 7.87% | 6.23% | 7.24% | 25.27% | 9.09% | 9.03% | 10.78% | `False` |

## Selected Candidate Support Summary

### `ETH/seq_eth/obsmat.txt`

- candidate_count: `3024`
- ridge_lambda: `1.0`
- score_lambda: `1.0`
- harm_weight: `5.0`
- threshold: `-0.28843049343373683`
- switch_cap: `0.5`
- support_summary: `{'mean_all_improvement': 0.4168944380689332, 'mean_t50_improvement': 0.47561116318771607, 'mean_hard_failure_improvement': 0.4186073744260826, 'min_all_improvement': 0.14266738483098973, 'max_easy_degradation': -0.04188338932028435, 'all_sources_easy_safe': True, 'all_sources_positive': True}`

### `UCY/students03/obsmat.txt`

- candidate_count: `3024`
- ridge_lambda: `100.0`
- score_lambda: `0.1`
- harm_weight: `5.0`
- threshold: `-1.0123871738466637`
- switch_cap: `0.5`
- support_summary: `{'mean_all_improvement': 0.2725576981190594, 'mean_t50_improvement': 0.1720691958111865, 'mean_hard_failure_improvement': 0.2672295698053363, 'min_all_improvement': 0.23677019000225796, 'max_easy_degradation': -0.2310181855602712, 'all_sources_easy_safe': True, 'all_sources_positive': True}`

## Bootstrap CI

| heldout source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.46% | 0.98% | 1.49% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | -33.68% | -31.86% | -30.39% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 7.60% | 7.84% | 8.10% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.52% | 1.05% | 1.56% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | -21.46% | -16.89% | -12.46% | 1812 |
| `UCY/students03/obsmat.txt` | `all` | 5.23% | 5.42% | 5.62% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 3.48% | 3.69% | 3.88% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 7.51% | 7.86% | 8.19% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 6.03% | 6.23% | 6.42% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 6.05% | 6.78% | 7.53% | 21424 |

## Interpretation

- This is a stricter blocked-source repair attempt: non-heldout source support must be easy-safe before held-out evaluation.
- If a blocked source remains blocked, deployment remains fallback-only for that source.
- This remains dataset-local/raw-frame 2.5D evidence and does not enable metric/seconds claims, Stage5C, or SMC.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
