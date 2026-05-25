# Stage42-L Horizon-Aware T50 Static-Gate Repair

- source: `fresh_run`
- generated_at_utc: `2026-05-25T23:36:22.341605+00:00`
- git_commit: `f30c8b8`
- input_hash: `dd226a898e33364594917be7b0a2093ebb0e9a2139f29542f744d40afff4e43e`
- gate: `11 / 11`
- verdict: `stage42_l_horizon_static_gate_repair_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-L 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Fresh Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | gate | gate t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `horizon_static_gate_repair` | `fresh_run` | 0.021866 | 0.002015 | 0.000240 | 0.023969 | 0.000000 | 0.039577 | 0.053153 | 0.130853 | 0.232218 | 0.190268 |

## Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-K fresh static-gated` | `cached_verified` | 0.013628 | -0.012228 | 0.014791 | 0.000000 | 0.035841 |
| `Stage42-J policy static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-L is a targeted repair for Stage42-K's negative ADE t50.
- It uses horizon embeddings, lower t50 static dropout, weaker t50 gate penalty, and a t50-weighted validation policy.
- It repairs Stage42-K's t+50 ADE sign (`-0.012228` to `+0.002015`) while improving ADE all/hard and FDE t50 and preserving easy cases.
- It still does not beat the Stage42-J policy-level static gate. Stage42-J remains the strongest static-gated full-waypoint evidence, while Stage42-L is the best fresh checkpoint in this branch so far.
- Future waypoints remain labels only; no future/test leakage, metric claim, Stage5C, or SMC is introduced.
