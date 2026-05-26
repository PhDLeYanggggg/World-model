# Stage42-CT Frozen Policy Replay / Reproducibility Verifier

- source: `fresh_replay_from_frozen_policy_artifact`
- generated_at_utc: `2026-05-26T23:08:06.195386+00:00`
- git_commit: `47b5df5`
- input_hash: `6ec42f5dbb0e9758f1cfa429a4b8f76e8405f5bb8eacb0fc0adb9e9dee56e64f`
- policy_hash_recomputed: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- gate: `30 / 30`
- verdict: `stage42_ct_frozen_policy_replay_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CT 是 Stage42-CS frozen policy artifact 的 replay / reproducibility verifier。
- CT 不重新选择阈值，不读取 test endpoint 构建 goals，不执行 Stage5C，不启用 SMC。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- replay 的目的不是新增模型分数，而是证明 frozen policy artifact 与 CQ/CR/CS 证据一致、可复核。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Replay Checks

- policy hash matches CS: `True`
- policy JSON matches CS embedded policy: `True`
- selected policy matches CQ: `True`
- base choices match CQ: `True`
- CR safety recommendation matches artifact: `True`

## Replayed Metrics Vs Endpoint-Linear ADE

- all: `1.77%`
- t50: `1.07%`
- t100 raw-frame diagnostic: `3.48%`
- hard/failure: `1.93%`
- easy degradation: `0.25%`

## Replayed Joint Safety Vs Endpoint-Linear

- near_collision@0.02 delta: `-0.00%`
- near_collision@0.05 delta: `-0.06%`
- p05 min group distance delta: `-0.01%`
- jagged-rate delta: `0.00%`

## Interpretation

- Stage42-CT proves the frozen policy artifact is reproducible from the stored CQ/CR/CS evidence.
- It does not add a new score and does not tune on test; it prevents the deployable policy from being a loose narrative claim.
- The frozen artifact remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.
