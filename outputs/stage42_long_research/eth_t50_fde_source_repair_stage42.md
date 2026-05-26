# Stage42-AG ETH_UCY T50/FDE Validation-Only Source Repair

- source: `fresh_run_from_stage42x_stage42r_stage42af_validation_fde_repair`
- generated_at_utc: `2026-05-26T06:28:10.440315+00:00`
- git_commit: `8469e33`
- input_hash: `e9045c7039270e05733144ca0e783c8c617702716615228cfbe4ce2e032c25d1`
- gate: `13 / 13`
- verdict: `stage42_ag_eth_t50_fde_source_repair_pass`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AG 是 ETH_UCY t50/FDE@50 validation-only source repair，不重新训练大模型。
- Source repair 使用 validation FDE@50 threshold，不用 test 调阈值。
- Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。
- t+50 / t+100 是 raw-frame horizons，不能说成 seconds-level。
- External coordinates remain dataset-local / unverified weak metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Source Repair Rule

- rule: `eth_ucy_t50_fde_guarded_static_source`
- target: `ETH_UCY|50`
- validation_fde_t50_threshold: `0.05`
- uses_test_metrics_for_threshold: `False`

## Summary

- ADE all: `0.09165604913216159`
- ADE t50: `0.06495706381315831`
- ADE t50 CI low: `0.05851255877278698`
- ADE t100 raw-frame diagnostic: `0.08153326024168321`
- ADE hard/failure: `0.09571602094795917`
- easy degradation CI high: `0.00334819268959823`
- FDE@50: `0.17785631342714833`

## ETH_UCY T50 Repair Effect

- ETH_UCY t50 ADE before: `0.017092525274558956`
- ETH_UCY t50 ADE after: `0.024635928198443218`
- ETH_UCY t50 ADE CI low before: `-0.013218100958604987`
- ETH_UCY t50 ADE CI low after: `0.002820688160982139`
- ETH_UCY FDE@50 before: `0.059435135650619385`
- ETH_UCY FDE@50 after: `0.10547615864239786`
- ETH_UCY FDE@50 CI low before: `-0.04199023614248535`
- ETH_UCY FDE@50 CI low after: `0.021040393452369632`
- ETH_UCY t50 limitation repaired: `True`

## Per-Seed Source Choices

| pair | choice | j val ADE@50 | j val FDE@50 | margin guards |
| ---: | --- | ---: | ---: | --- |
| 0 | `stage42j_static_expert` | 0.018378 | 0.141488 | `ETH_UCY|25, TrajNet|25` |
| 1 | `floor` | 0.000000 | 0.000000 | `ETH_UCY|25, TrajNet|25` |
| 2 | `stage42j_static_expert` | 0.009886 | 0.149292 | `ETH_UCY|25, TrajNet|25` |

## Conclusion

Stage42-AG repairs the ETH_UCY t50/FDE@50 lower-bound weakness by using a validation-only FDE@50 source guard. It promotes the static expert source on ETH_UCY|50 only where validation FDE@50 support is strong and otherwise falls back to the safety floor. This improves the weak slice without test threshold tuning and preserves the raw-frame/dataset-local claim boundary.
