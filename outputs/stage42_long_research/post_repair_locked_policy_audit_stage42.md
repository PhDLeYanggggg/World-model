# Stage42-AK Post-Repair Locked Policy and Source Split Audit

- source: `fresh_synthesis_from_stage42_af_ag_ai_aj_and_source_split`
- generated_at_utc: `2026-05-26T07:02:32.895964+00:00`
- git_commit: `453395b`
- input_hash: `cbc0f7cb892161bc2becfbe8987e9f20cc7702a102267c59428db4fed06ed1a5`
- policy_hash: `06772a241eedacc9b8828bddc7c70569ef7d0abc1951cc83eb1c5251e7979298`
- source_split_hash: `e22c1fc43543da7fea1805460163f8fcd7993e3dcf88a2eb04d40a82269584bd`
- gate: `17 / 17`
- verdict: `stage42_ak_post_repair_locked_policy_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AK 锁定的是 Stage42-AF/AG/AI post-repair policy 规则和 source-level split evidence，不重新训练模型。
- 所有 policy switch/guard 阈值来自 validation 或已有 frozen reports；不使用 test 调阈值。
- Future waypoints / endpoints 只允许作为 supervised labels 或 eval labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizons；t+100 仍只能 diagnostic。
- External coordinates 仍是 dataset-local / unverified weak-metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Locked Policy Rules

| order | rule | decision source | test threshold tuning | fallback |
| ---: | --- | --- | --- | --- |
| 1 | `base_stage42x_row_level_full_waypoint_policy` | cached_verified_stage42x_outputs | `False` | Stage37 / teacher safety floor for unsupported or unsafe slices |
| 2 | `stage42af_validation_margin_guard` | Stage42-R validation margin | `False` | floor_non_harm |
| 3 | `stage42ag_eth_ucy_t50_fde_source_repair` | validation FDE@50 and validation ADE@50 | `False` | Stage37 / teacher safety floor for unsupported or unsafe slices |
| 4 | `stage42ai_trajnet_t100_easy_safety_repair` | validation easy-degradation and validation ADE | `False` | Stage37 / teacher safety floor for unsupported or unsafe slices |

## Source-Level Split Audit

- protocol: `stage42_source_level_split_rebuild`
- source overlap pass: `True`
- split group overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

| split | rows | domains | scenes | sources | t50 | t100 | hard | easy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 266745 | `{'ETH_UCY': 134695, 'TrajNet': 75287, 'UCY': 56763}` | 9 | 13 | 64551 | 45285 | 198685 | 80855 |
| `val` | 23788 | `{'ETH_UCY': 16103, 'TrajNet': 7685}` | 2 | 2 | 5873 | 4411 | 21872 | 4454 |
| `test` | 47458 | `{'TrajNet': 37918, 'UCY': 9540}` | 2 | 3 | 11538 | 7048 | 35076 | 11192 |

## Post-Repair Summary From Stage42-AI

- ADE all CI low: `0.0859783492681093`
- ADE t50 CI low: `0.05851255877278698`
- ADE t100 raw-frame diagnostic CI low: `0.06834922663403784`
- ADE hard/failure CI low: `0.0906618058871814`
- easy degradation CI high: `0.00116827749002908`
- FDE@50 CI low: `0.14823015795452749`

## No-Leakage Audit

- passed: `True`

| check | value |
| --- | --- |
| `future_endpoint_input` | `False` |
| `future_waypoint_input` | `False` |
| `central_velocity` | `False` |
| `test_endpoint_goals` | `False` |
| `test_threshold_tuning` | `False` |
| `source_overlap_pass` | `True` |
| `frozen_eval_uses_old_train_rows` | `False` |
| `af_uses_test_threshold` | `False` |
| `ag_uses_test_threshold` | `False` |
| `ai_uses_test_threshold` | `False` |

## Interpretation

- Stage42-AK locks the post-repair policy after AF/AG/AI and records the source-level split evidence used by Stage42 external validation.
- This is reproducibility and deployment-boundary evidence, not a new model-training result.
- The policy remains protected by Stage37 / teacher floor for unsafe or unsupported slices.
- Claims remain dataset-local raw-frame 2.5D. Metric, seconds-level, true-3D, foundation, Stage5C, SMC, and ungated-neural deployment claims remain rejected.
