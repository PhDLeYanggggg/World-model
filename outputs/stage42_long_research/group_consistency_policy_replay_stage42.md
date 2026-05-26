# Stage42-DK Group-Consistency Policy Replay

- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`
- generated_at_utc: `2026-05-26T23:09:20.134924+00:00`
- git_commit: `47b5df5`
- input_hash: `f8398a2c8533d4c3a955b3f0dd07ea02dc1c76abf1044cb044ee8f4751f4dd6c`
- policy_hash_recomputed: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `34 / 34`
- verdict: `stage42_dk_group_consistency_policy_replay_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DK 是 Stage42-DJ frozen group-consistency full-waypoint policy 的 artifact replay / reproducibility verifier。
- DK 不重新训练、不重新选择阈值、不使用 test metrics 调参。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- group-consistency replay 只验证 frozen artifact 与 Stage42-DI/DJ 源证据一致。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Replay Checks

- policy hash matches Stage42-DJ: `True`
- policy JSON matches Stage42-DJ embedded policy: `True`
- repair rule matches Stage42-DI selected candidate: `True`
- validation score matches Stage42-DI: `True`
- bootstrap matches Stage42-DI: `True`
- no-leakage flags match Stage42-DI: `True`

## Replayed Metrics Vs Train-Horizon Causal Floor

- all: `24.72%`
- t50: `22.36%`
- t100 raw-frame diagnostic: `14.35%`
- hard/failure: `23.89%`
- easy degradation: `-25.63%`

## Replayed Group Safety

- base near@0.05: `1.94%`
- final near@0.05: `1.38%`
- floor near@0.05: `2.24%`
- base p05 min distance: `0.07437689768396878`
- final p05 min distance: `0.07770240407545181`

## Interpretation

- Stage42-DK proves the frozen Stage42-DJ group-consistency policy artifact replays the Stage42-DI source evidence exactly.
- It does not retrain, does not retune, and does not add a fresh score; it hardens the paper/deployment reproducibility chain.
- The policy remains protected dataset-local/raw-frame 2.5D evidence, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.
