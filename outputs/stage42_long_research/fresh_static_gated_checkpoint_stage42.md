# Stage42-K Fresh Static-Gated Checkpoint Training

- source: `fresh_run`
- generated_at_utc: `2026-05-25T23:07:16.227222+00:00`
- git_commit: `29c3a0c`
- input_hash: `7700adc38fb5013863dce41506b44d1dcc6fb98a76b1ea8b628229c874662eb4`
- gate: `9 / 9`
- verdict: `stage42_k_fresh_static_gated_checkpoint_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-K 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Fresh Checkpoint Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | static gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_static_gated_checkpoint` | `fresh_run` | 0.013628 | -0.012228 | 0.015858 | 0.014791 | 0.000000 | 0.031189 | 0.035841 | 0.099986 | 0.127814 |

## Cached Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stage42_j_full_static` | `cached_verified_checkpoints_fresh_static_gate_eval` | -0.010558 | -0.032082 | -0.011591 | 0.000000 | 0.020136 |
| `stage42_j_no_static` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.011491 | 0.019854 | 0.012881 | 0.000000 | 0.061141 |
| `stage42_j_static_gated` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-K is the fresh checkpoint version of the Stage42-J static-gate idea.
- Static dropout and a learned low-bias static gate are used to avoid the Stage42-I unconditional-static failure mode.
- Stage42-K improves over the failed Stage42-I full static+sequence head on ADE all/hard and FDE all/t50 while preserving easy cases.
- Stage42-K does not beat Stage42-J: ADE t50 remains negative (`-0.012228`), and Stage42-J remains stronger on ADE all/t50/hard and FDE t50.
- The next repair should make the learned static gate horizon-aware, especially for the t+50 slice.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
