# Stage42-EV Constraint-Aware Composer

- source: `fresh_stage42_constraint_aware_composer`
- generated_at_utc: `2026-05-27T04:30:04.905714+00:00`
- git_commit: `06a5618`
- input_hash: `60cd68bef12185531e12fda04f3f7e591d491250b282fb8a9113aca0ce1702a8`
- gate: `12 / 14`
- verdict: `stage42_ev_constraint_aware_composer_positive_not_promoted`
- decision: `constraint_aware_composer_positive_but_keep_stage42_di`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EV 在 Stage42-EU positive-not-promoted 后，测试 validation-only constraint-aware composer 是否能在 AM / DI / EU 候选之间按 domain+horizon+group-risk 安全切换。
- composer 只使用 validation 选择候选和阈值；test 只评一次。
- composer 的 risk bucket 来自当前/过去可得信息、source/frame/horizon group key、predicted rollout geometry，不使用 future waypoint 作为 inference input。
- 不下载、不转换、不执行 Stage5C、不启用 SMC。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Selected Composer

- mode: `domain_horizon`
- test_score: `0.935902`
- candidate_usage: `{'floor': 0, 'stage42_am': 0, 'stage42_di': 28720, 'stage42_eu': 18738}`

## Test Once Metrics

| all | t50 | t100 raw | hard/failure | easy | switch | near base/final |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24.71% | 22.35% | 14.35% | 23.88% | -25.10% | 58.80% | 1.94%/1.37% |

## Mode Comparison

| mode | score | all | t50 | t100 raw | hard | easy | usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | 0.879152 | 22.81% | 22.35% | 12.68% | 21.97% | -23.91% | `{'floor': 0, 'stage42_am': 0, 'stage42_di': 0, 'stage42_eu': 47458}` |
| `domain_horizon` | 0.935902 | 24.71% | 22.35% | 14.35% | 23.88% | -25.10% | `{'floor': 0, 'stage42_am': 0, 'stage42_di': 28720, 'stage42_eu': 18738}` |
| `domain_horizon_risk` | 0.932990 | 24.48% | 22.35% | 14.35% | 23.88% | -23.87% | `{'floor': 0, 'stage42_am': 0, 'stage42_di': 18771, 'stage42_eu': 28687}` |

## Delta vs Prior

| prior | all | t50 | t100 raw | hard | easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-AM` | 0.13% | 0.34% | -0.02% | 0.13% | 0.57% |
| `Stage42-DI` | -0.00% | -0.01% | 0.00% | -0.00% | 0.53% |
| `Stage42-EU` | 1.90% | 0.00% | 1.67% | 1.91% | -1.18% |

## Interpretation

- Stage42-EV asks whether Stage42-EU has any validation-supported slice where it should replace Stage42-DI.
- If the selected composer does not beat Stage42-DI on all and hard, the deployable policy remains Stage42-DI / CQ floor.
- This is still source-level raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.

## Gate

| gate | pass |
| --- | ---: |
| `candidate_families_rebuilt` | True |
| `validation_composer_modes_evaluated` | True |
| `selected_mode_recorded` | True |
| `test_all_positive_vs_floor` | True |
| `test_t50_positive_vs_floor` | True |
| `test_hard_positive_vs_floor` | True |
| `easy_degradation_under_2pct` | True |
| `near005_not_worse_than_base` | True |
| `beats_stage42_di_all` | False |
| `beats_stage42_di_hard` | False |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
