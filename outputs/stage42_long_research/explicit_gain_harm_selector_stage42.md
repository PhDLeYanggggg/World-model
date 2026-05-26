# Stage42-O Explicit Row-Level Gain/Harm Selector Head

- source: `fresh_run`
- generated_at_utc: `2026-05-26T01:28:43.323463+00:00`
- git_commit: `211cfc2`
- input_hash: `e6139e2503d7610e3c34d788cd9223d5f7dea6cca7679eab97c8d1c0d0ae9507`
- gate: `13 / 14`
- verdict: `stage42_o_explicit_gain_harm_selector_partial`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-O 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。
- future waypoints / future endpoints 只作为 train/val selector labels 和 eval label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage42-O 策略选择只在 validation 上完成，test 只最终评估一次。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Fresh Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `explicit_gain_harm_selector` | `fresh_run` | 0.052646 | -0.000776 | 0.060206 | 0.053527 | 0.015491 | 0.084362 | 0.057614 | 0.162825 |
| `same Stage42-N checkpoint baseline policy` | `fresh_run` | 0.025024 | -0.027816 | 0.020788 | 0.026923 | 0.000000 | 0.053537 | 0.055456 | 0.142132 |

## Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-N row gain static gate` | `cached_verified` | 0.025024 | -0.027816 | 0.026923 | 0.000000 | 0.055456 |
| `Stage42-J policy static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-O tests the Stage42-N diagnosis by adding an explicit row-level selector head for switch probability, gain, harm, and uncertainty.
- The deployment policy uses predicted switch/gain/harm/uncertainty only; it does not use test easy/hard labels as inference guards.
- Stage42-N checkpoints are cached-verified base predictors; the gain/harm selector and validation policy are fresh-run.
- Future waypoints remain train/val labels and final eval labels only, never inference inputs.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
