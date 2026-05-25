# Stage42-J Static-Gated Full-Waypoint Repair

- source: `cached_verified_checkpoints_fresh_static_gate_eval`
- generated_at_utc: `2026-05-25T22:42:03.937489+00:00`
- git_commit: `5e382f6`
- input_hash: `d8c21be8f1db30954fb8aef4b86fa22855bf38d6a759a237a455e3aa2a535aee`
- gate: `10 / 10`
- verdict: `stage42_j_static_gated_full_waypoint_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-J 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_static` | `cached_verified_checkpoints_fresh_static_gate_eval` | -0.010558 | -0.032082 | -0.003398 | -0.011591 | 0.000000 | 0.020169 | 0.020136 | 0.081971 |
| `no_static` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.011491 | 0.019854 | 0.014125 | 0.012881 | 0.000000 | 0.025397 | 0.061141 | 0.077901 |
| `static_alpha025` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.035245 | 0.034409 | 0.030068 | 0.038679 | 0.000000 | 0.057215 | 0.117438 | 0.157128 |
| `static_alpha050` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.032435 | 0.013408 | 0.032050 | 0.035554 | 0.000000 | 0.043066 | 0.051379 | 0.126375 |
| `static_alpha075` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.014380 | -0.005110 | 0.018575 | 0.015763 | 0.000000 | 0.034281 | 0.038229 | 0.114495 |
| `static_gated` | `cached_verified_checkpoints_fresh_static_gate_eval` | 0.036222 | 0.036875 | 0.026738 | 0.039705 | 0.000000 | 0.063285 | 0.116638 | 0.150843 |

## Gate Behavior

- selected expert counts by seed: `[{'floor': 3, 'no_static': 0, 'static_alpha025': 3, 'static_alpha050': 2, 'static_alpha075': 0, 'full_static': 0}, {'floor': 4, 'no_static': 0, 'static_alpha025': 0, 'static_alpha050': 3, 'static_alpha075': 0, 'full_static': 1}, {'floor': 3, 'no_static': 0, 'static_alpha025': 3, 'static_alpha050': 2, 'static_alpha075': 0, 'full_static': 0}]`
- Experts and static mix weights are selected on validation by domain/horizon slice; test is evaluated once.
- `no_static` means the Stage42-I no-static-context sequence head; `full_static` means the Stage42-I full static+sequence head.

## Interpretation

- Stage42-J repairs the Stage42-I failure mode by not forcing static/context into every full-waypoint prediction.
- This is a static expert gate over cached-verified Stage42-I checkpoints with fresh validation-gate/test evaluation, not a new checkpoint training run.
- If the static gate falls back mostly to no-static, that is still useful evidence: static/context is currently harmful unless gated.
- Results remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
