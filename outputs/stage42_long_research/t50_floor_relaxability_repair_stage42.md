# Stage42-BY Protected T50 Floor-Relaxability Repair

- source: `fresh_stage42_by_t50_floor_relaxability_repair`
- generated_at_utc: `2026-05-26T16:05:29.395021+00:00`
- git_commit: `6f54b66`
- input_hash: `2ae008cc66939eeae91e0a1e5e171d0c94fab7d9ff726d8abe0c916ec2f957ec`
- gate: `15 / 15`
- verdict: `stage42_by_t50_floor_relaxability_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BY 是 t50 floor-relaxability repair audit，不训练新模型，不执行 Stage5C，不启用 SMC。
- 本修复使用 Stage42-AW train-only UCY internal validation support；test source 不参与 policy/threshold 选择。
- 本修复不允许去掉 teacher/floor rollout context，也不是 ungated neural deployment。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。

## Summary

- verdict_short: `protected_t50_relaxability_repaired_but_not_floor_free`
- selected_variant: `family_baseline_rel_only`
- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- repaired_t50_slices: `['TrajNet|50', 'UCY|50']`
- still_blocked_t50_slices: `[]`
- global_t50_improvement: `28.97%`
- global_easy_degradation: `-37.05%`
- teacher_floor_context_required: `True`
- floor_free_neural_deployable: `False`

## Target T50 Decisions

| slice | before BX status | after rows | after all | after t50 | after hard | after easy | switch | repaired |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet|50` | `blocked_by_validation_safety` | 9198 | 30.21% | 30.21% | 30.21% | -22.95% | 95.26% | True |
| `UCY|50` | `blocked_no_validation_support` | 2340 | 24.53% | 24.53% | 24.53% | -12.64% | 65.00% | True |

## Interpretation

- Stage42-BX showed t50 fallback relaxation was blocked under the original slice audit.
- Stage42-BY repairs t50 only under the Stage42-AW protected validation policy with train-only UCY internal validation.
- This is not a floor-free result: teacher/floor rollout context and protected selection remain required.
- This does not change t100, metric, seconds-level, Stage5C, or SMC claims.
