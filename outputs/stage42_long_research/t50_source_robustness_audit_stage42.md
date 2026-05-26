# Stage42-CB Protected T50 Source Robustness Audit

- source: `fresh_stage42_cb_t50_source_robustness_audit`
- generated_at_utc: `2026-05-26T16:26:59.473768+00:00`
- git_commit: `d28b880`
- input_hash: `0e0447c133fada9b1ee16a1f11686f5fa5b49ca90b5ffcc9186ef3847cdbc6a8`
- gate: `11 / 11`
- verdict: `stage42_cb_t50_source_robustness_pass_with_source_diversity_limit`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CB 是 BY/BZ protected t50 repair 的 source-level robustness / concentration audit。
- Stage42-CB 不重新选择 policy，不调 threshold，不训练新模型。
- test rows 只用于最终 source-level reporting/bootstrap，不用于 policy selection。
- 如果 evidence 集中在少数 source，必须报告 source concentration limitation，不能包装成 broad source-level generalization。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C 未执行，SMC 未启用。

## Summary

- verdict_short: `aggregate_t50_bootstrap_strong_but_source_diversity_limited`
- robust_major_source_slices: `['TrajNet|50', 'UCY|50']`
- concentration_limited_slices: `['TrajNet|50', 'UCY|50']`
- broad_source_generalization_claim_allowed: `False`
- source_level_claim: `major-source robust within available rows; not broad source-level generalization`

## Slice Source Summary

| slice | rows | sources | largest source frac | robust sources | underpowered | broad source claim |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `TrajNet|50` | 9198 | 2 | 99.08% | `crowds/students002.txt, crowds/students003.txt` | `none` | False |
| `UCY|50` | 2340 | 1 | 100.00% | `crowds/crowds_zara03.txt` | `none` | False |

## Per-Source Evidence

| slice | source | rows | t50 | t50 CI low | t50 CI high | easy CI high | switch | robust |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet|50` | `crowds/students002.txt` | 85 | 71.35% | 68.64% | 73.96% | 0.00% | 100.00% | True |
| `TrajNet|50` | `crowds/students003.txt` | 9113 | 29.20% | 28.80% | 29.59% | -27.49% | 95.22% | True |
| `UCY|50` | `crowds/crowds_zara03.txt` | 2340 | 24.53% | 23.05% | 26.00% | -8.57% | 65.00% | True |

## Interpretation

- Stage42-CB supports the major available t50 sources but exposes source concentration.
- The aggregate BY/BZ evidence remains strong, but broad source-level generalization is not yet allowed.
- More independent legal t50-capable sources are still needed for a stronger paper claim.
- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.
