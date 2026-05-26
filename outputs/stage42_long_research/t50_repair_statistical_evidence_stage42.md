# Stage42-BZ Protected T50 Repair Statistical Evidence

- source: `fresh_stage42_bz_t50_repair_statistical_evidence`
- generated_at_utc: `2026-05-26T16:13:17.940946+00:00`
- git_commit: `2719eb7`
- input_hash: `8c091e67aad92d45c67ef0e6362ffdbac77b3c943d4fd77d39ebaf8b0cb4fc49`
- gate: `13 / 13`
- verdict: `stage42_bz_t50_repair_statistical_evidence_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BZ 给 Stage42-BY 的 protected t50 repair 补 bootstrap statistical evidence。
- Stage42-BZ 不重新选择 threshold，不使用 test metric 调 policy，不训练新模型。
- 本审计使用 Stage42-AW train-only internal validation policy；test rows 只用于最终报告和 bootstrap。
- teacher/floor rollout context 仍然是部署安全地板；本结果不是 floor-free neural deployment。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C 未执行，SMC 未启用。

## Summary

- verdict_short: `protected_t50_repair_has_positive_bootstrap_evidence_but_not_floor_free`
- selected_variant: `family_baseline_rel_only`
- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- robust_t50_slices: `['TrajNet|50', 'UCY|50']`
- weak_t50_slices: `[]`
- target_union_t50_ci_low: `28.52%`
- target_union_easy_ci_high: `-25.16%`
- bootstrap_n: `3000`
- floor_free_neural_deployable: `False`

## Slice Bootstrap Evidence

| slice | rows | t50 | t50 CI low | t50 CI high | hard CI low | easy CI high | switch | robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet|50` | 9198 | 30.21% | 29.80% | 30.67% | 29.74% | -27.61% | 95.26% | True |
| `UCY|50` | 2340 | 24.53% | 23.02% | 26.08% | 23.03% | -8.16% | 65.00% | True |

## Target Union Evidence

- rows: `11538`
- t50 improvement: `28.97%`
- t50 CI: `[28.52%, 29.45%]`
- hard/failure CI low: `28.51%`
- easy degradation CI high: `-25.16%`

## Interpretation

- Stage42-BZ upgrades Stage42-BY from point-estimate protected t50 repair to bootstrap-backed statistical evidence.
- The policy was selected by train plus internal validation only; test rows are final reporting/bootstrap evidence only.
- This remains protected policy evidence, not floor-free neural world dynamics.
- No true-3D, foundation, global metric, seconds-level, Stage5C, or SMC claim is allowed.
