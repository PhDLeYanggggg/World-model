# Stage42-HP Group-Consistency Source-Level Breakdown

- source: `fresh_run_group_consistency_source_breakdown`
- generated_at_utc: `2026-05-27T18:54:57.543271+00:00`
- git_commit: `0eb5bbb`
- input_hash: `6d4938837fa4a5ccede706bcb5bce122a6655d71a6c2d747a7efe7a8fbd3241e`
- gate: `23 / 23`
- verdict: `stage42_hp_group_consistency_breakdown_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HP 是 frozen group-consistency full-waypoint policy 的 fresh source-level breakdown，不是新阈值搜索。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Overall

| metric | value |
| --- | ---: |
| `rows` | `47458` |
| `ade_all_improvement` | `24.72%` |
| `ade_t50_improvement` | `22.36%` |
| `ade_t100_raw_frame_diagnostic_improvement` | `14.35%` |
| `ade_hard_failure_improvement` | `23.89%` |
| `ade_easy_degradation` | `-25.63%` |
| `fde_all_improvement` | `22.29%` |
| `fde_t50_improvement` | `22.57%` |
| `fde_t100_raw_frame_diagnostic_improvement` | `12.85%` |
| `switch_rate` | `58.81%` |
| `harm_over_floor_ade` | `-11.38%` |

## By Domain

| domain | rows | ADE all | ADE t50 | ADE t100 raw | hard/failure | easy | FDE t50 | switch | near delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `domain:TrajNet` | 37918 | 32.24% | 28.62% | 19.03% | 31.43% | -30.27% | 29.06% | 73.61% | -0.69% |
| `domain:UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | 0.00% | 0.00% |

## By Horizon

| horizon | rows | ADE all | FDE all | switch | near delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `horizon:10` | 15402 | 45.86% | 45.75% | 62.91% | -0.12% |
| `horizon:100` | 7048 | 14.35% | 12.85% | 25.64% | -0.30% |
| `horizon:25` | 13470 | 23.62% | 22.28% | 66.47% | -0.58% |
| `horizon:50` | 11538 | 22.36% | 22.57% | 64.66% | -1.26% |

## Weak Slice Ledger

| slice | rows | t50 rows | ADE all | ADE t50 | easy | near delta | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `domain:UCY` | 9540 | 2340 | 0.00% | 0.00% | -0.00% | 0.00% | `non_positive_all; non_positive_t50` |
| `source:UCY::TrajNet/Train/crowds/crowds_zara03.txt` | 9540 | 2340 | 0.00% | 0.00% | -0.00% | 0.00% | `non_positive_all; non_positive_t50` |
| `scene:UCY::UCY_crowds` | 9540 | 2340 | 0.00% | 0.00% | -0.00% | 0.00% | `non_positive_all; non_positive_t50` |
| `fallback_only` | 19548 | 4077 | 0.00% | 0.00% | -0.00% | -0.05% | `non_positive_all; non_positive_t50` |
| `horizon:100` | 7048 | 0 | 14.35% | 0.00% | 2.74% | -0.30% | `easy_degradation_over_2pct` |
| `t100_raw_frame_diagnostic` | 7048 | 0 | 14.35% | 0.00% | 2.74% | -0.30% | `easy_degradation_over_2pct` |

## Interpretation

- Stage42-HP adds per-domain/per-source/per-scene/per-horizon evidence for the frozen group-consistency full-waypoint policy.
- This is useful for paper-level evidence because it exposes weak slices instead of hiding them behind aggregate metrics.
- It does not execute Stage5C, does not enable SMC, and does not make metric/seconds-level claims.
