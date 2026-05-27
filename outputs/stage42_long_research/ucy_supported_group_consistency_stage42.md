# Stage42-DZ UCY-Supported Group-Consistency Full-Waypoint Repair

- source: `fresh_ucy_internal_validation_group_consistency_repair`
- generated_at_utc: `2026-05-27T01:14:09.849854+00:00`
- gate: `15 / 15`
- verdict: `stage42_dz_ucy_supported_group_consistency_pass_dual_domain`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DZ 在 Stage42-AW 的 UCY train-only internal-val split 上 fresh-runs Stage42-DI group-consistency repair。
- 目标是检查 explicit physical/group-consistency 是否能在 UCY validation support 修复后获得双域支持，而不是只在 TrajNet 上有效。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Split Repair

- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- ucy_val_rows_after: `9540`
- test_rows_unchanged: `True`

## Metrics Vs Train-Horizon Causal Floor

| policy | all | t50 | t100 raw | hard/failure | easy | switch | near@0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AW/AM rebuilt baseline-family selected | 0.327535 | 0.265934 | 0.211471 | 0.317495 | -0.321204 | 0.718951 | 0.020797 |
| DZ group-consistency repaired | 0.328904 | 0.269864 | 0.211165 | 0.318864 | -0.320940 | 0.718951 | 0.013148 |

## By Domain

| domain | rows | all | t50 | t100 raw | hard/failure | easy | switch | positive_safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet` | 37918 | 0.320715 | 0.281804 | 0.190091 | 0.312868 | -0.305529 | 0.740809 | `True` |
| `UCY` | 9540 | 0.355808 | 0.227206 | 0.275645 | 0.337848 | -0.406038 | 0.632075 | `True` |

## Interpretation

- This is a fresh repair run, not cached reuse of AW or DI reports.
- Stage42-AW fixes the UCY validation-support blocker by carving internal validation from original UCY train sources only.
- Stage42-DZ shows whether explicit group/physical consistency remains safe and positive when UCY has validation support.
- This still remains dataset-local/raw-frame 2.5D evidence and does not permit metric/seconds-level, true-3D, Stage5C, or SMC claims.

## Gate

| gate | pass |
| --- | --- |
| `ucy_internal_val_created` | `True` |
| `test_sources_unchanged` | `True` |
| `source_overlap_pass` | `True` |
| `group_repair_candidates_run` | `True` |
| `validation_selected_without_test` | `True` |
| `global_positive_safe` | `True` |
| `ucy_positive_safe` | `True` |
| `trajnet_positive_safe` | `True` |
| `group_consistency_not_worse_near005` | `True` |
| `dual_domain_group_consistency_supported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `ungated_full_waypoint_blocked` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
