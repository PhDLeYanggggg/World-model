# Stage42-P T50-Specific Gain/Harm Selector Repair

- source: `fresh_run`
- generated_at_utc: `2026-05-26T01:57:12.436279+00:00`
- git_commit: `596e53c`
- input_hash: `74b6f6eab95a4c189e5d4d2ef29b1d607c23d715a47b918da07e6521b714f0fb`
- gate: `14 / 14`
- verdict: `stage42_p_t50_gain_harm_selector_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-P 是 t+50-specific gain/harm selector repair，不是 metric 或 seconds-level 结果。
- future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- feature normalization 只使用 train split statistics。
- policy thresholds 只在 validation 上选择，test 只最终评估一次。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Fresh Metrics

| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `t50_gain_harm_selector` | `fresh_run` | 0.051537 | 0.006596 | 0.059254 | 0.053256 | 0.008580 | 0.080118 | 0.057431 | 0.139443 |

## Comparison

| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-O explicit gain/harm` | `cached_verified` | 0.052646 | -0.000776 | 0.053527 | 0.015491 | 0.057614 |
| `Stage42-J policy static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.116638 |

## Interpretation

- Stage42-P is a targeted follow-up to Stage42-O's strict train-normalized partial result.
- It increases t+50 row weight in train/val teacher supervision and uses a t+50-weighted validation policy search.
- The policy still uses only predicted switch/gain/harm/uncertainty; it does not use test easy/hard labels as inference guards.
- Future waypoints remain train/val labels and final eval labels only, never inference inputs.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
