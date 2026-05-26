# Stage42-AT Source-Level Safety Floor / Fallback Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T10:06:30.946097+00:00`
- git_commit: `092b12b`
- input_hash: `9c819580049bbca1620d1f058d9c468aca4a0ef81c69eca608e9ad24182da251`
- gate: `11 / 11`
- verdict: `stage42_at_source_level_fallback_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AT 是 proposed source-level fallback / teacher-floor context audit，不是 metric 或 seconds-level 结果。
- 本审计区分 fallback floor removal 与 teacher/floor rollout context removal；二者不能混为一谈。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Key Distinction

- `fallback removal` means the source-level ridge probe predicts every row without falling back to the floor.
- `teacher/floor context removal` means removing floor-related rollout context from the input feature family.
- A pass on fallback removal does not prove floor-free neural dynamics.

## Candidate Comparison

| candidate | features | protected all | protected t50 | protected hard | protected easy | ungated all | ungated t50 | ungated hard | ungated easy | ungated minus protected all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_family_all_context` | 35 | 0.287773 | 0.315425 | 0.275812 | -0.324186 | 0.461656 | 0.411874 | 0.458447 | -0.305625 | 0.173883 |
| `no_floor_rel_context` | 33 | 0.263654 | 0.223372 | 0.247525 | -0.317307 | 0.457971 | 0.397785 | 0.454473 | -0.305849 | 0.194316 |
| `family_only_no_floor_safe_context` | 23 | 0.273815 | 0.237296 | 0.257880 | -0.316080 | 0.462271 | 0.413671 | 0.457559 | -0.299288 | 0.188456 |
| `no_safe_baseline_context` | 25 | 0.266260 | 0.220398 | 0.251421 | -0.309882 | 0.452738 | 0.388565 | 0.447505 | -0.310550 | 0.186478 |

## Context Removal Deltas

| candidate | protected delta all | protected delta t50 | ungated delta all | ungated delta t50 |
| --- | ---: | ---: | ---: | ---: |
| `no_floor_rel_context` | -0.024119 | -0.092053 | -0.003685 | -0.014089 |
| `family_only_no_floor_safe_context` | -0.013959 | -0.078129 | 0.000615 | 0.001797 |
| `no_safe_baseline_context` | -0.021513 | -0.095027 | -0.008918 | -0.023309 |

## Interpretation

- fallback_removal_for_baseline_family_probe: `supported_on_this_source_level_split`
- teacher_floor_context_removal: `not_supported_as_global_replacement`
- interpretation: This audit separates fallback removal from teacher/floor rollout context removal. Baseline-family ridge can be evaluated ungated on this source-level split, but that does not prove floor-free neural dynamics because floor/baseline rollout context remains an input mechanism.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`
