# Stage42-AU Source-Level Baseline-Family Mechanism Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T10:21:57.403951+00:00`
- git_commit: `b9fda2f`
- input_hash: `4c3f11ab3bd55d7353171b557660d5c205013d899b5d04de568b73647c849e0d`
- gate: `11 / 11`
- verdict: `stage42_au_baseline_family_mechanism_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AU 是 proposed source-level baseline-family mechanism audit，不是 metric 或 seconds-level 结果。
- 本审计拆解 floor_rel、safe_baseline_rel、family_baseline_rel 和 horizon/domain controls 的贡献。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Audit Exists

Stage42-AN/AO/AP/AQ/AR/AS showed that history, goal, neighbor, sequence, and hand-built graph residual context do not independently beat the baseline-family first stage under the tested source-level protocols.
Stage42-AU therefore tests the mechanism that is actually working: baseline-family rollout context.

## Variant Comparison

| variant | features | protected all | protected t50 | protected hard | protected easy | ungated all | ungated t50 | ungated hard | ungated easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `horizon_domain_control` | 7 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.749323 | -1.075718 | -0.770703 | 1.525765 |
| `floor_rel_only` | 9 | 0.036215 | 0.000000 | 0.041718 | 0.006668 | -0.095932 | -0.381024 | -0.097221 | 0.020304 |
| `safe_baseline_rel_only` | 17 | 0.067707 | -0.099422 | 0.072759 | 0.018080 | 0.019887 | -0.148091 | 0.037576 | -0.088801 |
| `family_baseline_rel_only` | 23 | 0.273815 | 0.237296 | 0.257880 | -0.316080 | 0.462271 | 0.413671 | 0.457559 | -0.299288 |
| `floor_plus_safe` | 19 | 0.120324 | 0.036302 | 0.131648 | -0.011451 | 0.077881 | 0.065211 | 0.106730 | -0.100742 |
| `floor_plus_family` | 25 | 0.266260 | 0.220398 | 0.251421 | -0.309882 | 0.452738 | 0.388565 | 0.447505 | -0.310550 |
| `safe_plus_family` | 33 | 0.263654 | 0.223372 | 0.247525 | -0.317307 | 0.457971 | 0.397785 | 0.454473 | -0.305849 |
| `baseline_family_all` | 35 | 0.287773 | 0.315425 | 0.275812 | -0.324186 | 0.461656 | 0.411874 | 0.458447 | -0.305625 |

## Mechanism Summary

- best_single_family_protected: `family_baseline_rel_only`
- best_single_family_ungated: `family_baseline_rel_only`
- protected_multi_family_increment_supported: `True`
- ungated_multi_family_increment_supported: `False`
- mechanism_verdict: `baseline_family_rollout_context_supported_as_dominant_mechanism`
- interpretation: This audit tests whether current source-level success is just horizon/domain controls, one floor baseline, or a broader baseline-family rollout context. It does not claim true 3D, metric prediction, seconds-level horizons, Stage5C, SMC, or floor-free neural dynamics.

## Pairwise Deltas To Baseline-Family-All

| variant | protected delta all | protected delta t50 | ungated delta all | ungated delta t50 |
| --- | ---: | ---: | ---: | ---: |
| `horizon_domain_control` | 0.287773 | 0.315425 | 1.210979 | 1.487592 |
| `floor_rel_only` | 0.251559 | 0.315425 | 0.557588 | 0.792898 |
| `safe_baseline_rel_only` | 0.220067 | 0.414847 | 0.441770 | 0.559965 |
| `family_baseline_rel_only` | 0.013959 | 0.078129 | -0.000615 | -0.001797 |
| `floor_plus_safe` | 0.167449 | 0.279124 | 0.383775 | 0.346663 |
| `floor_plus_family` | 0.021513 | 0.095027 | 0.008918 | 0.023309 |
| `safe_plus_family` | 0.024119 | 0.092053 | 0.003685 | 0.014089 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`
