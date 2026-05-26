# Stage42-S Frozen Row Combo Policy + Stress Audit

- source: `fresh_run_from_stage42r_row_cache`
- generated_at_utc: `2026-05-26T03:29:50.836926+00:00`
- git_commit: `7b9ef39`
- stage42r_verdict: `stage42_r_row_cached_combo_pass`
- policy_hash: `33450e033e14b10293b8a10796d934d7689e39358ab5eaa338d684a36b015d3f`
- cache_hash: `f338f5c57b735b013ca210e30e9a6bbcfeebb646d4e0bc2e7f9e799006ac4ed6`
- feature_schema_hash: `58df31ca42fc6539799c6f66b9e4823b3ab6f5188cf3d535c8df4fb5114f551f`
- gate: `13 / 13`
- verdict: `stage42_s_frozen_row_combo_policy_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-S 冻结 Stage42-R row-cache combo policy，并做 stress audit；不是 metric 或 seconds-level 结果。
- future waypoints / endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。
- combo source 只由 validation domain/horizon slice 选择，test 只最终评估一次。
- row prediction cache 是本地 derived cache，不提交 GitHub。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Frozen Policy Summary

- policy artifact: `outputs/stage42_long_research/frozen_row_combo_policy_stage42_policy.json`
- candidate sources: `['floor', 'stage42j_static_expert', 'stage42p_t50_gain_harm']`
- positive domains: `['ETH_UCY', 'TrajNet']`
- t50 positive domains: `['ETH_UCY', 'TrajNet']`

## Core Metrics

| metric | mean | ci_low | ci_high |
| --- | ---: | ---: | ---: |
| ADE all | 0.052387 | 0.027941 | 0.076833 |
| ADE t50 | 0.037934 | 0.027740 | 0.048128 |
| ADE t100 raw-frame diagnostic | 0.041846 | 0.003158 | 0.080533 |
| ADE hard/failure | 0.054792 | 0.030217 | 0.079366 |
| ADE easy degradation | 0.001102 | -0.001058 | 0.003262 |
| FDE t50 | 0.100059 | 0.056062 | 0.144056 |
| switch rate | 0.154571 | 0.138275 | 0.170866 |

## Per-Domain Stress

| domain | rows | ADE all | ADE t50 | ADE hard | easy degr | FDE t50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 25901 | 0.042817 | 0.017093 | 0.044320 | 0.001614 | 0.059435 |
| `TrajNet` | 20087 | 0.102635 | 0.097465 | 0.108920 | 0.004552 | 0.235413 |
| `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

- Stage42-S freezes the Stage42-R row-cache combo policy as a lightweight policy artifact.
- The policy is still protected and validation-selected; test labels are not used for source selection.
- UCY remains fallback-only in the Stage42-R combo stress table, so this is stronger branch evidence but not a foundation-scale generalization claim.
- All results remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.
