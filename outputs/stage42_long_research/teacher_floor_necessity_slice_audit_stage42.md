# Stage42-JW Teacher Floor Necessity Slice Audit

- source: `fresh_stage42_jw_teacher_floor_necessity_slice_audit`
- generated_at_utc: `2026-05-29T06:47:51.835300+00:00`
- git_commit: `98b691b`
- input_hash: `dabd28d8e0d61cf3abd4395f21c74b5acd2aabef44398ca00db6886b5f22f1c9`
- gate: `14 / 14`
- verdict: `stage42_jw_teacher_floor_necessity_slice_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JW 审计 Stage37/teacher floor 与 safe-switch 的 slice-level 必要性，不训练、不调 threshold。
- JW 只使用 cached_verified IW/JV/GT 证据做 fresh synthesis，不把 floor-free 或 ungated neural 包装成成功。
- future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Floor / Switch Summary

- rows: `47458`
- switch_rows: `33355`; fallback_rows: `14103`
- switch_rate: `0.702832`; fallback_rate: `0.297168`
- fallback_exact_floor_rate: `1.000000`
- mean_gain_all_rows: `0.134267`; mean_gain_switched_rows: `0.191037`
- harm_rate_all_rows: `0.066164`; harm_rate_switched_rows: `0.076600`
- hard_failure_switch_rate: `0.729644`; easy_switch_rate: `0.412616`
- easy_mean_harm: `0.003129`

## Slice Evidence

| slice | rows | ADE improvement | easy degradation | switch rate |
| --- | ---: | ---: | ---: | ---: |
| `switched` | 33355 | 0.372722 | 0.000000 | 1.000000 |
| `fallback` | 14103 | 0.000000 | 0.000000 | 0.000000 |
| `easy` | 11192 | 0.256627 | 0.000000 | 0.412616 |
| `hard_or_failure` | 35076 | 0.287273 | 0.000000 | 0.729644 |

## Partial Floor Relaxation

- target_slices: `['TrajNet|50', 'UCY|50']`
- target_union_rows: `11538`
- target_union_t50_improvement: `0.289698`
- target_union_hard_failure_improvement: `0.289698`
- target_union_easy_degradation: `-0.214067`
- target_union_near_collision_005_delta: `-0.007379`
- target_union_jagged_rate_delta: `0.000000`
- target_union_safety_pass: `True`
- global_floor_removal_allowed: `False`
- floor_free_neural_deployable: `False`

## Gate

| gate | pass |
| --- | ---: |
| `iw_mechanism_passed` | `True` |
| `jv_slice_matrix_passed` | `True` |
| `gt_floor_relaxation_stress_passed` | `True` |
| `floor_is_actively_used` | `True` |
| `fallback_exact_floor` | `True` |
| `switch_slice_positive` | `True` |
| `fallback_slice_nonharmful` | `True` |
| `hard_switch_rate_ge_easy` | `True` |
| `partial_t50_relaxation_supported` | `True` |
| `global_floor_free_forbidden` | `True` |
| `deployment_decision_protected` | `True` |
| `no_metric_seconds_or_3d_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- The teacher/floor is not merely cosmetic: a nontrivial fallback share remains exact-floor, hard/failure rows switch more often than easy rows, and global floor-free neural deployment remains forbidden. A restricted t50 relaxation has safety evidence but does not remove the global floor.
- This supports teacher/floor as a safety mechanism and bounded contribution, not as a claim that floor-free neural dynamics is deployable.
