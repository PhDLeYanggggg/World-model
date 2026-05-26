# Stage42-N Row-Level Gain/Harm Static-Gate Distillation

- source: `fresh_run`
- generated_at_utc: `2026-05-26T00:50:50.047948+00:00`
- git_commit: `543a217`
- input_hash: `a45cea103a0450936ac9a64648ef51eba5de017a4bbf248e17625e42218c1584`
- gate: `11 / 13`
- verdict: `stage42_n_row_gain_static_gate_partial`
- teacher_seeds: `[53]`
- teacher_cache_dir: `data/stage42_row_gain_teacher_cache`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-N 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 loss/eval label 和 row-level supervised teacher，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- row-level teacher 只在 train/val 上构建，test 不用于训练、阈值或 teacher fitting。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Row-Level Teacher Diagnostics

| split | rows | static positive | mean alpha | mean static gain | mean floor gain | harm positive | switchable | t50 static positive | t50 switchable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | 219667 | 0.447468 | 0.248674 | 0.084433 | 0.059291 | 0.497148 | 0.445943 | 0.414881 | 0.461955 |
| `val` | 53256 | 0.516430 | 0.265336 | 0.066947 | 0.065030 | 0.424309 | 0.522589 | 0.499504 | 0.572017 |

## Fresh Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | gate | gate t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `row_gain_static_gate` | `fresh_run` | 0.025024 | -0.027816 | 0.020788 | 0.026923 | 0.000000 | 0.053537 | 0.055456 | 0.142132 | 0.283838 | 0.257578 |

## Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-L horizon static gate` | `cached_verified` | 0.021866 | 0.002015 | 0.023969 | 0.000000 | 0.053153 |
| `Stage42-M coarse alpha distillation` | `cached_verified` | 0.016145 | -0.001544 | 0.017698 | 0.000000 | 0.072906 |
| `Stage42-J policy static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-N is a direct follow-up to Stage42-M's failure: it replaces coarse domain/horizon alpha with row-level static gain, floor gain, harm, and switchability supervision.
- This run is a single-teacher-seed row-level pilot for speed and recoverability; it is not a full teacher ensemble.
- Row-level teacher labels are built only for train/val from cached-verified Stage42-I checkpoints plus train/val full-waypoint labels; test labels are not used for teacher fitting or threshold tuning.
- Future waypoints remain supervised labels only and never inference inputs.
- If Stage42-N still fails, the evidence points away from alpha-style distillation and toward row-level expert prediction or a separate gain/harm selector head.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
