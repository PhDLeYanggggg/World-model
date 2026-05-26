# Stage42-R Row Prediction Cache + Combo Eval

- source: `fresh_run_from_row_prediction_cache`
- generated_at_utc: `2026-05-26T03:16:06.802204+00:00`
- git_commit: `e0a3cb6`
- input_hash: `f02e6ccb5bd47a8aba749d3d1f20d178448bb7d3b4e2c7f52236c9a0480c4307`
- gate: `15 / 15`
- verdict: `stage42_r_row_cached_combo_pass`
- cache_dir: `data/stage42_row_prediction_cache` (not committed)

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-R 构建 row-level prediction cache 并从 cache 做 validation-only combo eval，不是 metric 或 seconds-level 结果。
- future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- Stage42-P feature normalization 只使用 train split statistics。
- combo source policy 只在 validation 上选择，test 只最终评估一次。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Metrics

| candidate | source | ADE all | ADE t50 | ADE t50 CI low | ADE t100 diag | ADE hard | ADE easy degr | FDE t50 | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Stage42-R cached combo` | `fresh_run_from_row_prediction_cache` | 0.052387 | 0.037934 | 0.027740 | 0.041846 | 0.054792 | 0.001102 | 0.100059 | 0.154571 |
| `Stage42-J from cache` | `fresh_run_from_row_prediction_cache` | 0.036222 | 0.036875 | 0.016268 | 0.026738 | 0.039705 | 0.000000 | 0.116638 | 0.150843 |
| `Stage42-P from cache` | `fresh_run_from_row_prediction_cache` | 0.051537 | 0.006596 | -0.017931 | 0.059254 | 0.053256 | 0.008580 | 0.057431 | 0.139443 |

## Bootstrap Over Seed-Mean Row Improvements

| slice | rows | mean | ci_low | ci_high |
| --- | ---: | ---: | ---: | ---: |
| `all` | 55528 | 0.025074 | 0.023733 | 0.026436 |
| `t50` | 13689 | 0.018563 | 0.016897 | 0.020246 |
| `t100_raw_frame_diagnostic` | 9905 | 0.044213 | 0.038031 | 0.050660 |
| `hard_failure` | 41741 | 0.031826 | 0.029996 | 0.033543 |
| `easy` | 16739 | 0.000027 | -0.000159 | 0.000264 |

## Source Choices

- validation-selected source counts across seed/domain/horizon slices: `{'stage42p_t50_gain_harm': 14, 'floor': 4, 'stage42j_static_expert': 6}`
- Candidate sources are `floor`, `Stage42-J static expert`, and `Stage42-P t50 gain/harm`.
- Source selection is by validation domain/horizon slice only; test labels are not used for threshold or source selection.

## Interpretation

- Stage42-R turns the Stage42-Q preflight into a cache-backed row-level evaluation path.
- Cache files are local derived arrays and are intentionally not committed.
- If the cached combo fails t+50 CI or source-diversity gates, that is honest evidence that simple slice-level source selection is still not enough.
- Future waypoints remain train/val labels and final eval labels only, never inference inputs.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
