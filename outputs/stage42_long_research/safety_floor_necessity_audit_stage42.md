# Stage42-BW Safety Floor Necessity Audit

- source: `fresh_stage42_bw_safety_floor_necessity_audit`
- generated_at_utc: `2026-05-26T15:47:39.714956+00:00`
- git_commit: `25b17b2`
- input_hash: `ea91a32da16d25551889c51b447e0c0bf894df1c341b95958576fb3fc365bb90`
- gate: `15 / 15`
- verdict: `stage42_bw_safety_floor_necessity_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BW 是 safety-floor necessity audit，不训练新模型，不执行 Stage5C，不启用 SMC。
- 本审计区分三件事：fallback floor、teacher/floor rollout context、以及无保护 neural dynamics。
- future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- source-specific calibration 不等于 global metric claim。

## Main Conclusion

- verdict_short: `teacher_floor_required_but_baseline_family_probe_can_relax_fallback`
- current_deployable_family: `current_composite_tail_policy`
- current all / t50 / hard: `21.03%` / `13.65%` / `20.38%`
- current easy degradation: `0.00%`
- floor_free_neural_deployable: `False`

## Safety Floor Evidence

| policy | all | t50 | t100 raw diag | hard | easy degradation | collision delta | deployable interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current composite-tail protected | 21.03% | 13.65% | 14.69% | 20.38% | 0.00% | -0.39% | deployable under safety floor |
| ungated endpoint | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | n/a | unsafe: easy harm |
| ungated full-waypoint | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | n/a | unsafe: easy harm |
| teacher raw policy | 35.15% | 23.67% | 35.80% | 35.09% | 0.00% | 1.87% | not deployed due physical/proximity warning |

## Fallback vs Teacher/Floor Context

- fallback_removal_for_baseline_family_probe: `supported_on_this_source_level_split`
- teacher_floor_context_removal: `not_supported_as_global_replacement`
- no_floor_rel_context_protected_delta_t50: `-9.21%`
- no_safe_baseline_context_protected_delta_t50: `-9.50%`
- interpretation: Fallback relaxation in one baseline-family probe is not equivalent to removing teacher/floor rollout context or deploying ungated neural dynamics.

## Mechanism Evidence

- dominant_supported_mechanism: `baseline_family_rollout_context_supported_as_dominant_mechanism`
- best_single_family_protected: `family_baseline_rel_only`
- protected_multi_family_increment_supported: `True`
- small tabular neural context verdict: `stage42_aq_neural_context_not_supported`
- positive_neural_context_variants: `[]`

## Row-Level / Full-Waypoint Evidence

- frozen_row_combo ADE all CI low: `2.79%`
- frozen_row_combo ADE t50 CI low: `2.77%`
- frozen_row_combo easy CI high: `0.33%`
- unified_row_cache ADE all CI low: `8.24%`
- unified_row_cache ADE t50 CI low: `5.37%`
- unified_row_cache easy CI high: `0.33%`

## Claim Boundary

- This is evidence for safety-floor necessity and baseline-family rollout context, not true 3D, not foundation, not global metric, not seconds-level prediction.
- Stage5C remains unexecuted and SMC remains disabled.
