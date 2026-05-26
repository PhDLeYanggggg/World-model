# Stage42-CR Proximity Guard Ablation / Pareto Audit

- source: `fresh_synthesis_from_stage42_co_cp_cq_artifacts`
- generated_at_utc: `2026-05-26T19:28:18.239312+00:00`
- git_commit: `8420811`
- input_hash: `415138ebcc0430ffd8989254c112980b313199e025a7f44dbfeaa301be87eeab`
- gate: `19 / 19`
- verdict: `stage42_cr_proximity_guard_ablation_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CR 是 CO/CP/CQ 的 proximity guard ablation / Pareto 审计。
- CR 不新增 test tuning；它比较已经 validation-selected 的 CO composer 与 CQ proximity guard。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Ablation Rows

| variant | role | all | t50 | t100 raw | hard | easy | near@0.05 vs endpoint | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `endpoint_linear_reference` | no_full_waypoint_shape_reference | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `no_proximity_guard` | accuracy_priority_diagnostic | 3.02% | 1.50% | 6.12% | 3.28% | 0.25% | 0.34% | 21.35% |
| `proximity_guard` | safety_sensitive_deployable | 1.77% | 1.07% | 3.48% | 1.93% | 0.25% | -0.06% | 16.96% |

## Guard Contribution

- all-ADE cost versus no guard: `-1.24%`
- t50-ADE cost versus no guard: `-0.44%`
- t100 raw diagnostic cost versus no guard: `-2.64%`
- hard/failure cost versus no guard: `-1.35%`
- near-collision@0.05 repair versus no guard: `-0.40%`

## Recommendation

- accuracy-priority diagnostic policy: `no_proximity_guard`
- safety-sensitive deployment policy: `proximity_guard`
- Do not present no-guard CO/CP as the safety-sensitive deployment policy because its near-collision@0.05 is worse than endpoint-linear.
- Do not present CQ as strictly more accurate than CO/CP; CQ is the safer Pareto point.

## Interpretation

- The proximity guard has a real causal contribution: it repairs the near-collision caveat while keeping all/t50/t100 raw-frame/hard-failure gains positive.
- The tradeoff is explicit: CQ gives up some CO/CP ADE gain to satisfy joint-proximity safety.
- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.
