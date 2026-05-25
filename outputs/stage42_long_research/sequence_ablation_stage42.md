# Stage42-H Causal Sequence Ablation

- source: `fresh_run`
- generated_at_utc: `2026-05-25T21:07:32.959048+00:00`
- git_commit: `7f7ddac`
- input_hash: `0532940fdbf60ffe62ffc5afd739a6c8d417b65f3ee4ed23558374e3d8a7136d`
- gate: `10 / 10`
- verdict: `stage42_h_sequence_ablation_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- External 数据仍是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- future endpoints / family_fde 只作为 supervised label/eval，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Metrics

| variant | all mean | t50 mean | t100 diag mean | hard mean | easy mean | switch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequence_full_no_safe_switch` | 0.809540 | 0.838216 | 0.934106 | 0.830117 | -0.841261 | 0.856819 |
| `sequence_full_safe_switch` | 0.778471 | 0.783362 | 0.916625 | 0.808073 | -0.768404 | 0.349999 |
| `sequence_no_domain_expert` | 0.736930 | 0.741477 | 0.893436 | 0.768207 | -0.696326 | 0.349999 |
| `sequence_no_goal_scene_tokens` | 0.773151 | 0.787621 | 0.911819 | 0.803702 | -0.757132 | 0.349999 |
| `sequence_no_history_tokens` | 0.330436 | 0.325545 | 0.358510 | 0.337275 | -0.325641 | 0.243332 |
| `sequence_no_neighbor_interaction_tokens` | 0.778549 | 0.782064 | 0.917929 | 0.809417 | -0.749788 | 0.349999 |

## Contribution Deltas

`full_minus_ablation > 0` means the removed component helped the full sequence model on that slice.

| ablation | all delta | t50 delta | hard delta | easy delta ablation-minus-full |
| --- | ---: | ---: | ---: | ---: |
| `sequence_full_no_safe_switch` | -0.031069 | -0.054854 | -0.022043 | -0.072857 |
| `sequence_no_domain_expert` | 0.041542 | 0.041885 | 0.039867 | 0.072077 |
| `sequence_no_goal_scene_tokens` | 0.005321 | -0.004259 | 0.004372 | 0.011272 |
| `sequence_no_history_tokens` | 0.448036 | 0.457817 | 0.470799 | 0.442762 |
| `sequence_no_neighbor_interaction_tokens` | -0.000078 | 0.001298 | -0.001343 | 0.018615 |
