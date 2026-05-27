# Stage42-EA Dual-Domain Group-Consistency Statistical Evidence

- source: `fresh_stage42_ea_dual_domain_group_consistency_statistics`
- generated_at_utc: `2026-05-27T01:20:24.248175+00:00`
- gate: `12 / 12`
- verdict: `stage42_ea_dual_domain_group_consistency_statistics_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EA fresh-runs the Stage42-DZ UCY-supported group-consistency repair and adds 2000-bootstrap dual-domain statistical evidence。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Bootstrap Confidence Intervals

| slice | all | t50 | t100 raw diag | hard/failure | easy degradation | positive safe |
| --- | --- | --- | --- | --- | --- | --- |
| `global` | [0.325616, 0.328949, 0.332309] n=47458 | [0.265328, 0.269842, 0.274426] n=11538 | [0.204499, 0.211320, 0.217585] n=7048 | [0.315115, 0.318928, 0.322634] n=35076 | [-0.329610, -0.320970, -0.312813] n=11192 | `True` |
| `TrajNet` | [0.317175, 0.320758, 0.324137] n=37918 | [0.277244, 0.281780, 0.286073] n=9198 | [0.183073, 0.190119, 0.197128] n=5608 | [0.308982, 0.312919, 0.316642] n=27687 | [-0.313873, -0.305579, -0.297362] n=9213 | `True` |
| `UCY` | [0.346983, 0.355902, 0.364877] n=9540 | [0.213784, 0.227544, 0.241802] n=2340 | [0.258242, 0.275150, 0.293326] n=1440 | [0.328373, 0.337853, 0.347613] n=7389 | [-0.431496, -0.406609, -0.379076] n=1979 | `True` |

## Near-Collision Bootstrap

| slice | base near@0.05 | final near@0.05 | delta final-base |
| --- | --- | --- | --- |
| `global` | [0.019554, 0.020797, 0.022062] n=47458 | [0.012095, 0.013148, 0.014160] n=47458 | [-0.008555, -0.007628, -0.006722] n=47458 |
| `TrajNet` | [0.018698, 0.020122, 0.021626] n=37918 | [0.011893, 0.013054, 0.014241] n=37918 | [-0.008096, -0.007094, -0.006092] n=37918 |
| `UCY` | [0.020335, 0.023375, 0.026415] n=9540 | [0.011216, 0.013522, 0.015831] n=9540 | [-0.011845, -0.009853, -0.007859] n=9540 |

## Summary

- bootstrap_n: `2000`
- ci_positive_safe_domains: `2`
- global_ci_positive_safe: `True`
- ucy_ci_positive_safe: `True`
- trajnet_ci_positive_safe: `True`
- near005_delta_high: `-0.0067217329006700665`
- ucy_val_rows_after: `9540`

## Interpretation

- This is fresh statistical evidence rebuilt from row-level selected/floor ADE arrays, not a reuse of aggregate DZ metrics.
- The claim is dual-domain raw-frame/dataset-local 2.5D support for protected group-consistency full-waypoint dynamics.
- It does not allow metric/seconds-level, true-3D, foundation, Stage5C, or SMC claims.

## Gate

| gate | pass |
| --- | --- |
| `fresh_repair_rebuilt` | `True` |
| `bootstrap_n_2000` | `True` |
| `global_ci_positive_safe` | `True` |
| `ucy_ci_positive_safe` | `True` |
| `trajnet_ci_positive_safe` | `True` |
| `two_domain_ci_supported` | `True` |
| `near_collision_not_worse` | `True` |
| `test_sources_unchanged` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
