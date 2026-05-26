# Stage42-BX Slice-Level Floor Relaxability Audit

- source: `fresh_stage42_bx_floor_relaxability_audit`
- generated_at_utc: `2026-05-26T15:55:30.733771+00:00`
- git_commit: `4bf358e`
- input_hash: `83af4bf659326c2c2aa90d28b3771d4f43dc2f0d9e9fe03532422c81a5fcf806`
- gate: `14 / 14`
- verdict: `stage42_bx_floor_relaxability_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BX 是 slice-level floor relaxability audit，不训练新模型，不执行 Stage5C，不启用 SMC。
- 本审计只判断哪些 source/horizon slice 可安全放松 fallback；不允许去掉 teacher/floor rollout context。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。

## Summary

- verdict_short: `fallback_relaxation_is_slice_limited_teacher_context_still_required`
- relaxable slices: `['TrajNet|25']`
- blocked_no_validation_support: `['UCY|10', 'UCY|100', 'UCY|25', 'UCY|50']`
- blocked_by_validation_safety: `['TrajNet|10', 'TrajNet|100', 'TrajNet|50']`
- t50_relaxable_slices: `[]`
- t50_blocked_slices: `['TrajNet|50', 'UCY|50']`
- t100_relaxable_slices: `[]`
- teacher_floor_context_required: `True`
- floor_free_neural_deployable: `False`

## Slice Decisions

| slice | val rows | test rows | val all | val easy | test all | test easy | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 2465 | 12342 | 60.42% | 40.45% | 70.00% | -59.27% | `blocked_by_validation_safety` |
| `TrajNet|100` | 1160 | 5608 | 63.45% | 112.99% | 43.12% | 46.19% | `blocked_by_validation_safety` |
| `TrajNet|25` | 2175 | 10770 | 17.67% | -24.33% | 42.63% | -51.85% | `relaxable_under_validation_rule` |
| `TrajNet|50` | 1885 | 9198 | 44.97% | 20.22% | 46.06% | -19.21% | `blocked_by_validation_safety` |
| `UCY|10` | 0 | 3060 | 0.00% | -0.00% | 50.82% | -43.64% | `blocked_no_validation_support` |
| `UCY|100` | 0 | 1440 | 0.00% | -0.00% | 53.29% | 72.92% | `blocked_no_validation_support` |
| `UCY|25` | 0 | 2700 | 0.00% | -0.00% | -17.56% | -30.46% | `blocked_no_validation_support` |
| `UCY|50` | 0 | 2340 | 0.00% | -0.00% | 23.78% | 28.91% | `blocked_no_validation_support` |

## Interpretation

- Fallback relaxation is not globally deployable; it is allowed only for validation-supported source/horizon slices.
- UCY slices in this audit lack validation support and therefore remain fallback-required here even when test metrics look positive.
- TrajNet t100 remains blocked by validation easy harm; t100 remains raw-frame diagnostic and not a deployable long-horizon claim.
- This audit does not authorize removing teacher/floor rollout context and does not authorize ungated neural dynamics.

## Claim Boundary

- Not true 3D, not foundation, not global metric, not seconds-level.
- Stage5C remains unexecuted and SMC remains disabled.
