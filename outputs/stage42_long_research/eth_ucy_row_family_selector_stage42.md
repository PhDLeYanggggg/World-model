# Stage42-JK ETH_UCY Row-Level Family Selector

- source: `fresh_stage42_jk_eth_ucy_row_family_selector`
- generated_at_utc: `2026-05-28T18:34:06.181557+00:00`
- git_commit: `f9337f7`
- input_hash: `0d26a1512e14cf1a7778af2fa2dae17ec6925757c91bc6ed08ff29e5eceecb50`
- gate: `11 / 11`
- verdict: `stage42_jk_eth_ucy_row_family_selector_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JK follows Stage42-JJ: family-oracle t50 headroom exists, but static family policy cannot select rows safely.
- JK trains row-level expected gain/harm predictors over causal family baselines using train sources only, selects thresholds on validation source, and evaluates held-out source once.
- future waypoints / endpoints are labels/eval only, never inference inputs.
- No central velocity, no test endpoint goals, and no test-threshold tuning are used.
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `row_family_selector_not_deployable_on_blocked_sources`
- targeted_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repaired_sources: `[]`
- still_blocked_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- positive_t50_sources: `[]`
- easy_safe_sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- next_action: If row-family selector is positive but unsafe, add a heldout-source-safe harm model; if still negative, acquire/calibrate source-specific geometry support.

## Held-Out Source Metrics

| source | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | oracle t50 | deployable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | 21598 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | 53.80% | `False` |
| `UCY/students03/obsmat.txt` | 70585 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | 39.14% | `False` |

## Selected Candidates

### `ETH/seq_eth/obsmat.txt`

- fallback_only: `True`
- reason: `no_validation_safe_row_family_candidate`
- lambda: `None`
- harm_weight: `None`
- threshold: `None`
- switch_cap: `None`
- support_summary: `None`
- best_rejected_candidate: `{'lambda': 0.01, 'harm_weight': 0.0, 'threshold': -1.4593140003040297, 'switch_cap': 0.02, 'score': -1000000000.0, 'val_metric': {'rows': 16103, 'all_improvement': -0.010587827762140067, 't10_improvement': -0.020300107207082974, 't25_improvement': -0.030570800794409214, 't50_improvement': -0.012798499230454174, 't100_raw_frame_diagnostic_improvement': 0.0016853095172291166, 'hard_failure_improvement': -0.00965411059134147, 'easy_degradation': 0.012795094459367995, 'switch_rate': 0.005278519530522263, 'harm_over_fallback': 0.006182210824537177}, 'support_summary': {'mean_all_improvement': -0.033647700172962924, 'mean_t50_improvement': -0.04087691243516761, 'mean_hard_failure_improvement': -0.027093913486093135, 'min_all_improvement': -0.07584640454186409, 'max_easy_degradation': 0.3553991202847364, 'all_sources_easy_safe': False, 'all_sources_positive': False}}`
- family_switch_counts: `{'constant_position': 0, 'constant_velocity_causal_fd': 0, 'damped_velocity': 0, 'constant_acceleration': 0, 'turn_rate': 0, 'history_decay_baseline': 0, 'prototype_goal_directed_baseline': 0, 'neighbor_aware_decay_baseline': 0}`

### `UCY/students03/obsmat.txt`

- fallback_only: `True`
- reason: `no_validation_safe_row_family_candidate`
- lambda: `None`
- harm_weight: `None`
- threshold: `None`
- switch_cap: `None`
- support_summary: `None`
- best_rejected_candidate: `{'lambda': 0.01, 'harm_weight': 0.0, 'threshold': -1.6627181689057529, 'switch_cap': 0.02, 'score': -1000000000.0, 'val_metric': {'rows': 16611, 'all_improvement': -0.007009086180723045, 't10_improvement': -0.013514155405386585, 't25_improvement': -0.017883924336416612, 't50_improvement': -0.0032726440587831362, 't100_raw_frame_diagnostic_improvement': 0.0, 'hard_failure_improvement': -0.0008145860507302594, 'easy_degradation': 0.10458089237012547, 'switch_rate': 0.004635482511588706, 'harm_over_fallback': 0.0035549169447387867}, 'support_summary': {'mean_all_improvement': -0.010601917225676771, 'mean_t50_improvement': -0.005701435825468504, 'mean_hard_failure_improvement': -0.008168354177559511, 'min_all_improvement': -0.032617023284090996, 'max_easy_degradation': 0.10458089237012547, 'all_sources_easy_safe': False, 'all_sources_positive': False}}`
- family_switch_counts: `{'constant_position': 0, 'constant_velocity_causal_fd': 0, 'damped_velocity': 0, 'constant_acceleration': 0, 'turn_rate': 0, 'history_decay_baseline': 0, 'prototype_goal_directed_baseline': 0, 'neighbor_aware_decay_baseline': 0}`

## Bootstrap CI

| source | slice | low | mid | high | n |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 21598 |
| `ETH/seq_eth/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 5074 |
| `ETH/seq_eth/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 2614 |
| `ETH/seq_eth/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 18865 |
| `ETH/seq_eth/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 1812 |
| `ETH/seq_eth/obsmat.txt` | `oracle_t50` | 52.95% | 53.82% | 54.64% | 5074 |
| `UCY/students03/obsmat.txt` | `all` | 0.00% | 0.00% | 0.00% | 70585 |
| `UCY/students03/obsmat.txt` | `t50` | 0.00% | 0.00% | 0.00% | 17529 |
| `UCY/students03/obsmat.txt` | `t100_raw_frame_diagnostic` | 0.00% | 0.00% | 0.00% | 15470 |
| `UCY/students03/obsmat.txt` | `hard_failure` | 0.00% | 0.00% | 0.00% | 54600 |
| `UCY/students03/obsmat.txt` | `easy_degradation` | 0.00% | 0.00% | 0.00% | 21424 |
| `UCY/students03/obsmat.txt` | `oracle_t50` | 38.66% | 39.14% | 39.66% | 17529 |

## Interpretation

- Row-level family selection directly targets the Stage42-JJ finding: family oracle has t50 headroom, static policy cannot choose rows.
- If the selector remains blocked, the missing piece is likely source-specific geometry/history/harm prediction rather than family availability.
- This remains dataset-local/raw-frame 2.5D evidence; no Stage5C, SMC, metric, or seconds-level claim is enabled.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_gain_harm_targets': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
