# Stage42-AI TrajNet|100 Validation-Only Easy-Safety Repair

- source: `fresh_run_from_stage42ag_trajnet_t100_validation_easy_safety`
- generated_at_utc: `2026-05-26T06:43:13.049315+00:00`
- git_commit: `74f4e07`
- gate: `13 / 13`
- verdict: `stage42_ai_trajnet_t100_safety_repair_pass`

## Current Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-AI 是 TrajNet|100 validation-only easy-safety repair，不重新训练大模型。
- t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。
- Source repair 使用 validation easy-degradation，不用 test 调阈值。
- Future waypoints/endpoints 只作为 labels/eval，不作为 inference input。
- External coordinates remain dataset-local / unverified weak metric diagnostic。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Source Repair Rule

- rule: `trajnet_t100_validation_easy_safety_guard`
- target_slice: `TrajNet|100`
- validation_easy_nonharm_threshold: `0.0`
- uses_test_metrics_for_threshold: `False`

## Summary

- ADE all CI low: `0.0859783492681093`
- ADE t50 CI low: `0.05851255877278698`
- ADE t100 raw-frame diagnostic CI low: `0.06834922663403784`
- ADE hard/failure CI low: `0.0906618058871814`
- easy degradation CI high: `0.00116827749002908`

## TrajNet|100 Repair Effect

- TrajNet|100 ADE before: `0.09875907078956987`
- TrajNet|100 ADE after: `0.11720354460558946`
- TrajNet|100 ADE CI low after: `0.048714385704072966`
- TrajNet|100 easy CI high before: `0.08498424090178214`
- TrajNet|100 easy CI high after: `0.0`
- TrajNet|100 safety repaired: `True`

## Per-Seed Source Choices

| pair | TrajNet|100 choice | j val all | j val easy | p val all | p val easy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `stage42j_static_expert` | 0.091949 | 0.000000 | 0.113807 | 0.004137 |
| 1 | `stage42j_static_expert` | 0.103089 | 0.000000 | 0.159361 | 0.033877 |
| 2 | `stage42p_t50_gain_harm` | 0.078879 | 0.000000 | 0.134993 | 0.000000 |

## Conclusion

Stage42-AI repairs the TrajNet|100 easy-safety limitation using a validation-only source safety guard. The t100 result remains raw-frame diagnostic and must not be described as seconds-level long-horizon prediction, but the safety boundary is stronger than Stage42-AH: the repaired TrajNet|100 slice keeps positive ADE/hard lower bounds while reducing easy degradation to non-harm.
