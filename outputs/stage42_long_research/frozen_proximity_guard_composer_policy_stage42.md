# Stage42-CS Frozen Proximity-Guard Composer Policy

- source: `fresh_policy_freeze_from_stage42_cq_cr`
- generated_at_utc: `2026-05-26T19:37:47.636797+00:00`
- git_commit: `2d338ab`
- input_hash: `988f8c0c5e8141841842c7c01dff26eb1b7a29ff6f307bd2c96a6037e7d5b2bb`
- policy_hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- policy_artifact: `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42_policy.json`
- gate: `25 / 25`
- verdict: `stage42_cs_frozen_proximity_guard_policy_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CS 冻结 Stage42-CQ/CR 选择的 proximity-guard composer policy。
- policy 冻结只使用 validation-selected 阈值和已审计的 predicted-rollout geometry guard。
- test 只用于最终报告，不用于阈值选择。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Policy

- policy_name: `stage42_cs_frozen_proximity_guard_composer_policy`
- deployment_role: `safety_sensitive_deployable_composer_variant`
- selection_scope: `validation_only`
- test_usage: `test_once_after_policy_freeze`
- guard min_sep: `0.2`
- guard margin: `0.005`
- guard input: `predicted endpoint/full-waypoint rollout geometry only`
- accuracy-priority diagnostic policy: `no_proximity_guard`

## Test Metrics Vs Endpoint-Linear ADE

- all: `1.77%`
- t50: `1.07%`
- t100 raw-frame diagnostic: `3.48%`
- hard/failure: `1.93%`
- easy degradation: `0.25%`
- switch rate: `16.96%`

## Joint Safety Vs Endpoint-Linear

- near_collision@0.02 delta: `-0.00%`
- near_collision@0.05 delta: `-0.06%`
- p05 min group distance delta: `-0.01%`
- jagged-rate delta: `0.00%`

## Interpretation

- Stage42-CS freezes the safer Stage42-CQ proximity-aware composer as a reproducible policy artifact.
- The no-proximity-guard composer remains useful as an accuracy-priority diagnostic, but it is not the safety-sensitive deployment policy.
- This artifact advances deployability/reproducibility for the protected full-waypoint composer branch without changing the claim boundary.
- It remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.
