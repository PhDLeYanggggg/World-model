# Stage42-M Policy-Distilled Static Gate Checkpoint

- source: `fresh_run`
- generated_at_utc: `2026-05-26T00:04:55.573611+00:00`
- git_commit: `093ba3f`
- input_hash: `a7d7aeaccc21d0e8a2af2e48b40d47d9f89b60873cffcea5f4fcc59f2141500f`
- gate: `10 / 12`
- verdict: `stage42_m_policy_distilled_static_gate_partial`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-M 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage42-J teacher 来自 validation-selected static expert policy，不使用 test endpoints。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Teacher Alpha Map

```json
{
  "ETH_UCY|10": 0.3333333333333333,
  "ETH_UCY|100": 0.0,
  "ETH_UCY|25": 0.0,
  "ETH_UCY|50": 0.16666666666666666,
  "TrajNet|10": 0.5,
  "TrajNet|100": 0.6666666666666666,
  "TrajNet|25": 0.0,
  "TrajNet|50": 0.3333333333333333
}
```

## Fresh Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | gate | gate t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `policy_distilled_static_gate` | `fresh_run` | 0.016145 | -0.001544 | 0.002127 | 0.017698 | 0.000000 | 0.041539 | 0.072906 | 0.136106 | 0.221998 | 0.180516 |

## Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-L horizon static gate` | `cached_verified` | 0.021866 | 0.002015 | 0.023969 | 0.000000 | 0.053153 |
| `Stage42-J policy static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-M distills Stage42-J's validation-selected domain/horizon static expert choices into a fresh checkpoint gate target.
- The teacher does not use test endpoints; future waypoints remain supervised labels/eval only.
- The goal is to close the gap between Stage42-L's fresh checkpoint and Stage42-J's stronger policy-level gate.
- The result is partial, not a pass: FDE t50 improves over Stage42-L, but ADE t50 remains negative and the model does not beat Stage42-L on ADE all/t50/hard.
- The likely failure mode is that the Stage42-J teacher is only a coarse domain/horizon alpha. It does not provide row-level expected ADE gain, harm risk, or switchability targets.
- Next repair should distill row-level gain/harm from expert predictions instead of only static alpha.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
