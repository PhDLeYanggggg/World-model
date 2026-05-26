# Stage42-Q T50 Static Expert + Gain/Harm Combo Preflight

- source: `cached_verified_report_level_preflight`
- generated_at_utc: `2026-05-26T02:45:52.340387+00:00`
- git_commit: `2cc68d5`
- input_hash: `bb1a7aceba8c957606cef96f0d76c1848ea589e5a4585ba5c1a40b19c0bc5851`
- gate: `7 / 7`
- verdict: `stage42_q_preflight_partial_row_cache_required`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-Q 是 validation-only static expert + t50 gain/harm selector 组合评估，不是 metric 或 seconds-level 结果。
- future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- feature normalization 只使用 train split statistics。
- combo source policy 只在 validation 上选择，test 只最终评估一次。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Cached Candidate Metrics

| candidate | source | ADE all | ADE t50 | ADE t50 CI low | ADE hard | easy degr | FDE t50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Stage42-J static-gated` | `cached_verified` | 0.036222 | 0.036875 | 0.016268 | 0.039705 | 0.000000 | 0.116638 |
| `Stage42-P t50 gain/harm` | `cached_verified` | 0.051537 | 0.006596 | -0.017931 | 0.053256 | 0.008580 | 0.057431 |
| `Stage42-O explicit gain/harm` | `cached_verified` | 0.052646 | -0.000776 | -0.017531 | 0.053527 | 0.015491 | 0.057614 |

## Complementarity

- `{'p_beats_j_all': True, 'p_beats_j_hard': True, 'j_beats_p_t50': True, 'p_t50_seed_ci_low_negative': True, 'both_preserve_easy': True}`

## Diagnostic Envelope

- `{'source': 'diagnostic_only_not_deployable', 'ade_all_best_available': 0.0526457864037421, 'ade_t50_best_available': 0.036875348395170704, 'ade_hard_best_available': 0.0535270529782426, 'fde_t50_best_available': 0.11663789673246368, 'easy_degradation_worst_available': 0.015491233410829327, 'warning': 'This envelope is not a deployable policy because it chooses metrics after reading cached reports; it only motivates row-level caching and validation-only combo evaluation.'}`

## Interpretation

- Stage42-Q preflight confirms that Stage42-J and Stage42-P are complementary: P is stronger on all/hard, while J is stronger and more stable on t+50.
- This is not a deployable combo result and must not be used as a final policy claim.
- The direct row-level recomputation path was attempted but is too heavy without a row prediction cache; the next aligned engineering step is an NPZ row-cache for floor/J/P selected ADE-FDE/switch arrays.
- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
