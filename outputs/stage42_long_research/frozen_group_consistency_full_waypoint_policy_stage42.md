# Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy

- source: `fresh_policy_freeze_from_stage42_di`
- generated_at_utc: `2026-05-26T22:39:10.692568+00:00`
- git_commit: `fa534ce`
- input_hash: `2e799e6107abf6ab369957fd2395cff26f799f5ee4d6f7ee12225e533d185180`
- policy_hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- policy_artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- gate: `22 / 22`
- verdict: `stage42_dj_frozen_group_consistency_policy_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DJ 冻结 Stage42-DI group-consistency full-waypoint repair policy，形成可复现 policy artifact。
- policy 冻结只记录 validation-selected repair；test 指标来自 Stage42-DI test-once evidence。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Policy

- policy_name: `stage42_dj_frozen_group_consistency_full_waypoint_policy`
- deployment_role: `promoted_protected_source_level_full_waypoint_group_consistency_policy`
- selection_scope: `validation_only`
- test_usage: `test_once_from_stage42_di_after_validation_selection`
- repair_rule: `{'type': 'repel_unsafe', 'min_sep': 0.08, 'margin': 0.0, 'strength': 0.5, 'alpha': None, 'safe_min_sep': None, 'input': 'predicted full-waypoint rollout geometry + source/frame/horizon group key + agent id', 'uses_future_labels': False}`

## Test Metrics Vs Train-Horizon Causal Floor

- all: `24.72%`
- t50: `22.36%`
- t100 raw-frame diagnostic: `14.35%`
- hard/failure: `23.89%`
- easy degradation: `-25.63%`
- switch rate: `58.81%`

## Group Safety

- base near@0.05: `1.94%`
- final near@0.05: `1.38%`
- floor near@0.05: `2.24%`
- base p05 min distance: `0.07437689768396878`
- final p05 min distance: `0.07770240407545181`

## Interpretation

- Stage42-DJ freezes the Stage42-DI group-consistency full-waypoint repair as a reproducible policy artifact.
- This advances deployability and paper reproducibility for the protected full-waypoint branch.
- It remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.
