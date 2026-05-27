# Stage42-HE Floor-Free Proximity-Guard Robustness Audit

- source: `fresh_stage42_he_floor_free_proximity_guard_robustness`
- generated_at_utc: `2026-05-27T16:54:52.240091+00:00`
- git_commit: `5b4bc33`
- input_hash: `9310ffb0abb734e3917c59d9b22ecd044743fe811fa903ace93cf107cb971c45`
- gate: `21 / 21`
- verdict: `stage42_he_floor_free_proximity_guard_robustness_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-HE 是 Stage42-HD teacherless proximity-guard repaired gate 的 robustness audit，不重新训练大模型。
- Stage42-HE 使用 HD 已冻结的 validation-selected min_sep，不用 test 调 threshold。
- teacher gate 不参与该 repaired gate；但 causal floor fallback 仍必须存在。
- future endpoint / future waypoint 只作为监督或评估标签，不允许作为 inference input。
- 不使用 central velocity，不使用 test endpoints 建 goals，不执行 Stage5C，不启用 SMC。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。

## Direct Decision

- deployment_decision: `teacherless_gate_has_robust_evidence_but_requires_causal_floor_fallback`
- policy_family: `harm_predictor_gate`
- min_sep: `0.05`
- teacher_gate_used: `False`
- causal_floor_fallback_used: `True`
- global_floor_removal_allowed: `False`
- teacherless_gate_paper_evidence_supported: `True`

## Aggregate Metrics

- rows: `55528`
- all: `20.74%`
- t50: `13.82%`
- t100 raw diagnostic: `13.68%`
- hard/failure: `19.99%`
- easy degradation: `0.00%`
- collision delta @0.05: `-0.47%`
- switch rate: `32.11%`
- raw switch rate: `50.55%`
- guarded-off rate: `18.44%`

## Bootstrap CI

| slice | low | mid | high | n | bootstrap_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 20.38% | 20.74% | 21.11% | 55528 | 2000 |
| `t10` | 47.60% | 48.19% | 48.73% | 16726 | 2000 |
| `t25` | 10.43% | 11.02% | 11.63% | 15208 | 2000 |
| `t50` | 13.22% | 13.82% | 14.44% | 13689 | 2000 |
| `t100_raw_frame_diagnostic` | 12.94% | 13.67% | 14.39% | 9905 | 2000 |
| `hard_failure` | 19.57% | 20.00% | 20.35% | 41741 | 2000 |
| `easy_degradation` | -19.47% | -17.85% | -16.17% | 16739 | 2000 |

## Domain Robustness

| domain | robust | rows | all | all CI low | t50 | t50 CI low | t100 raw | hard | hard CI low | easy | easy CI high | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | True | 25901 | 18.10% | 17.59% | 12.95% | 12.11% | 11.53% | 17.65% | 17.13% | -13.35% | -13.07% | 33.73% |
| `TrajNet` | True | 20087 | 23.16% | 22.47% | 17.05% | 15.92% | 13.78% | 22.01% | 21.27% | -16.55% | -16.87% | 28.85% |
| `UCY` | True | 9540 | 24.02% | 23.06% | 11.08% | 9.79% | 19.68% | 23.36% | 22.35% | -16.01% | -15.74% | 34.58% |

## Horizon Robustness

| horizon | rows | positive CI | improvement | CI low | CI high | raw-frame diagnostic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 16726 | True | 48.19% | 47.62% | 48.75% | False |
| 25 | 15208 | True | 11.03% | 10.44% | 11.64% | False |
| 50 | 13689 | True | 13.82% | 13.19% | 14.41% | False |
| 100 | 9905 | True | 13.68% | 12.94% | 14.41% | True |

## Weak Domain-Horizon Slices

- weak_domain_horizon_slices: `none`

## Interpretation

- Stage42-HE does not choose a new threshold; it audits the Stage42-HD frozen validation-selected repaired gate.
- The teacher gate is removed for this repaired switch gate, but causal floor fallback is still the safety floor.
- This supports a teacherless switch-gate evidence claim only under floor fallback; it does not support global floor removal or ungated neural deployment.
- t+100 remains raw-frame diagnostic; no metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.
