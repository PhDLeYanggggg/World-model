# Stage42-HD Floor-Free Proximity-Guard Repair

- source: `fresh_stage42_hd_floor_free_proximity_guard_repair`
- generated_at_utc: `2026-05-27T16:34:47.975405+00:00`
- git_commit: `96914d9`
- input_hash: `ec5d420142a378a0ac07179f924f0ce0d25eb5aeaba3e93d4882e73cdf1a532d`
- gate: `13 / 13`
- verdict: `stage42_hd_floor_free_proximity_guard_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HD 不是重新训练大模型；它是 floor-free gate 的 validation-only proximity-guard repair。
- proximity guard 只用预测 rollout、group key、normalizer 和 causal floor rollout；future endpoint/waypoint 只作为监督或评估标签。
- guard threshold 只在 validation 选择，test 只最终评估一次。
- 本轮不下载、不转换、不执行 Stage5C，不启用 SMC。
- 即使 floor-free gate 修复成功，也不是 global floor removal；它仍依赖 causal baseline floor fallback 作安全约束。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。

## Direct Decision

- deployment_decision: `teacherless_floor_free_gate_can_be_proximity_repaired_but_only_with_causal_floor_safety_fallback`
- pre_guard_deployable_count: `0`
- post_guard_deployable_count: `4`
- best_post_guard_family: `harm_predictor_gate`
- best_post_guard_selected_min_sep: `0.05`
- teacher_gate_used: `False`
- causal_floor_fallback_used: `True`
- global_floor_removal_allowed: `False`

## Best Post-Guard Metrics

- all: `20.74%`
- t50: `13.82%`
- t100 raw diagnostic: `13.68%`
- hard/failure: `19.99%`
- easy degradation: `0.00%`
- collision delta @0.05: `-0.47%`
- switch rate: `32.11%`
- guarded-off rate: `18.44%`

## Repair Matrix

| family | min sep | deployable | all | t50 | t100 raw | hard | easy | collision d005 | switch | guarded off | collision repair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `harm_predictor_gate` | 0.05 | True | 20.74% | 13.82% | 13.68% | 19.99% | 0.00% | -0.47% | 32.11% | 18.44% | 2.44% |
| `uncertainty_gate` | 0.05 | True | 20.71% | 13.80% | 13.62% | 19.95% | 0.00% | -0.47% | 32.27% | 18.47% | 2.43% |
| `internal_self_gate` | 0.05 | True | 20.69% | 13.70% | 13.62% | 19.93% | 0.00% | -0.47% | 31.86% | 18.26% | 2.42% |
| `conformal_risk_gate` | 0.05 | True | 20.23% | 12.96% | 13.46% | 19.61% | 0.00% | -0.36% | 28.72% | 16.37% | 2.22% |

## Interpretation

- Stage42-HC showed floor-free gates had high raw gain but failed strict proximity safety.
- Stage42-HD tests whether a validation-selected proximity guard can repair those floor-free gates without using the teacher gate.
- The repaired policy is not a global floor removal: it still falls back to the strongest causal floor when predicted proximity becomes unsafe.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.
