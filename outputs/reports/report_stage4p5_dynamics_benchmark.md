# Stage 4.5 Dynamics Benchmark

## Summary

Stage 4.5 修复了 dt/velocity/coordinate 的动力学审计路径，并把 native / causal / central velocity 分开。正式 benchmark 使用 causal_fd velocity。TGSIM 仍然是 verified real t+100 benchmark，但它更像 traffic/generic trajectory dynamics，而不是纯 human crowd dynamics。

## Episode Summary

| dataset_name | total_scenes | total_agents | total_tracks | total_frames | mean_track_length | coordinate_unit | whether_metric_coordinates | whether_scene_geometry_available | velocity_source | samples_t10 | samples_t25 | samples_t50 | samples_t100 | whether_t100_verified | build_horizon | cannot_evaluate_t100 | train_episodes | val_episodes | test_episodes | mean_agents_per_episode | split_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TGSIM Foggy Bottom | 1 | 119 | 119 | 24941 | 420.168 | meter | True | False | causal_fd | 7401 | 2390 | 1128 | 482 | True | 100 | None | 7 | 2 | 3 | 2.583 | scene split when possible; single-scene datasets use chronological non-overlapping windows |

## Velocity Audit

| dataset_name | default_velocity_source | official_benchmark_velocity_source | central_fd_usage | dt_min | dt_median | dt_max | dt_unique_rounded | native_vs_causal_velocity_MAE | native_vs_causal_velocity_corr | native_vs_central_velocity_MAE | native_vs_central_velocity_corr | causal_speed_mean | causal_speed_p95 | causal_speed_max | causal_accel_mean | causal_accel_p95 | causal_accel_max | missing_frame_gaps | abnormal_jumps_gt_10m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TGSIM Foggy Bottom | causal_fd | causal_fd | diagnostic_only | 0.1 | 0.1 | 1350.0 | [0.1, 0.2] | 1e-05 | 1.0 | 0.0125 | 0.99988 | 0.72342 | 4.6996 | 15.96784 | 0.25425 | 1.29867 | 6.16462 | 10 | 1 |

## Agent Type Audit

| agent_type | rows | tracks | mean_speed | p95_speed | mean_acceleration |
| --- | --- | --- | --- | --- | --- |
| 3.0 | 47517 | 110 | 0.71688 | 4.68802 | 0.25203 |
| 7.0 | 2454 | 7 | 0.83125 | 4.91992 | 0.30015 |
| 5.0 | 29 | 2 | 2.31883 | 2.67848 | 0.01913 |

## Metrics

| model | branch_count | ADE@1 | FDE@1 | minADE@N@1 | minFDE@N@1 | ADE@10 | FDE@10 | minADE@N@10 | minFDE@N@10 | ADE@25 | FDE@25 | minADE@N@25 | minFDE@N@25 | ADE@50 | FDE@50 | minADE@N@50 | minFDE@N@50 | ADE@100 | FDE@100 | minADE@N@100 | minFDE@N@100 | coverage_FDE_lt_1m | coverage_FDE_lt_2m | coverage_FDE_lt_5m | coverage_FDE_lt_10m | physical_validity_rate | boundary_violation_rate | collision_violation_rate | speed_violation_rate | acceleration_violation_rate | status | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_velocity_native_velocity | 1 | 0.00053 | 0.00053 | 0.00053 | 0.00053 | 0.0056 | 0.01143 | 0.0056 | 0.01143 | 0.01501 | 0.03036 | 0.01501 | 0.03036 | 0.03267 | 0.06399 | 0.03267 | 0.06399 | 0.06245 | 0.12288 | 0.06245 | 0.12288 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| constant_velocity_causal_fd | 1 | 0.00053 | 0.00053 | 0.00053 | 0.00053 | 0.0056 | 0.01143 | 0.0056 | 0.01143 | 0.01501 | 0.03036 | 0.01501 | 0.03036 | 0.03267 | 0.06399 | 0.03267 | 0.06399 | 0.06245 | 0.12288 | 0.06245 | 0.12288 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| constant_velocity_central_fd_diagnostic | 1 | 0.00027 | 0.00027 | 0.00027 | 0.00027 | 0.00448 | 0.00957 | 0.00448 | 0.00957 | 0.01298 | 0.02693 | 0.01298 | 0.02693 | 0.02921 | 0.05825 | 0.02921 | 0.05825 | 0.05527 | 0.10646 | 0.05527 | 0.10646 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| constant_acceleration_causal | 1 | 0.00039 | 0.00039 | 0.00039 | 0.00039 | 0.00752 | 0.02084 | 0.00752 | 0.02084 | 0.0508 | 0.14989 | 0.0508 | 0.14989 | 0.21353 | 0.64402 | 0.21353 | 0.64402 | 0.86724 | 2.57365 | 0.86724 | 2.57365 | 0.0 | 0.33333 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.13531 | 0.0 |  |  |
| damped_velocity | 1 | 0.00053 | 0.00053 | 0.00053 | 0.00053 | 0.00538 | 0.0107 | 0.00538 | 0.0107 | 0.0135 | 0.02598 | 0.0135 | 0.02598 | 0.02698 | 0.04785 | 0.02698 | 0.04785 | 0.04265 | 0.07066 | 0.04265 | 0.07066 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| constant_turn_rate_velocity | 1 | 0.00057 | 0.00057 | 0.00057 | 0.00057 | 0.00675 | 0.01397 | 0.00675 | 0.01397 | 0.01571 | 0.02698 | 0.01571 | 0.02698 | 0.02573 | 0.0379 | 0.02573 | 0.0379 | 0.03398 | 0.0482 | 0.03398 | 0.0482 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| identity_hand_physics | 1 | 0.00053 | 0.00053 | 0.00053 | 0.00053 | 0.0056 | 0.01143 | 0.0056 | 0.01143 | 0.01501 | 0.03036 | 0.01501 | 0.03036 | 0.03267 | 0.06399 | 0.03267 | 0.06399 | 0.06245 | 0.12288 | 0.06245 | 0.12288 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| tuned_hand_physics | 1 | 0.00053 | 0.00053 | 0.00053 | 0.00053 | 0.0056 | 0.01143 | 0.0056 | 0.01143 | 0.01501 | 0.03036 | 0.01501 | 0.03036 | 0.03267 | 0.06399 | 0.03267 | 0.06399 | 0.06245 | 0.12288 | 0.06245 | 0.12288 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.0 | 0.0 |  |  |
| residual_over_constant_velocity | 1 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 0.00322 | 0.00726 | 0.00322 | 0.00726 | 0.01481 | 0.0393 | 0.01481 | 0.0393 | 0.04903 | 0.12442 | 0.04903 | 0.12442 | 0.24119 | 1.15487 | 0.24119 | 1.15487 | 0.66667 | 0.66667 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.09571 | 0.0264 |  |  |
| residual_over_constant_acceleration | 1 | 0.00032 | 0.00032 | 0.00032 | 0.00032 | 0.00321 | 0.00725 | 0.00321 | 0.00725 | 0.0148 | 0.0393 | 0.0148 | 0.0393 | 0.04901 | 0.12423 | 0.04901 | 0.12423 | 0.23962 | 1.14498 | 0.23962 | 1.14498 | 0.66667 | 0.66667 | 1.0 | 1.0 | 1.0 | None | 0.0 | 0.09571 | 0.0264 |  |  |
| residual_over_tuned_hand_physics | 1 | 0.00037 | 0.00037 | 0.00037 | 0.00037 | 0.00743 | 0.02066 | 0.00743 | 0.02066 | 0.10527 | 0.48015 | 0.10527 | 0.48015 | 0.76196 | 1.87091 | 0.76196 | 1.87091 | 3.0218 | 8.92705 | 3.0218 | 8.92705 | 0.0 | 0.0 | 0.0 | 0.66667 | 1.0 | None | 0.0 | 0.73597 | 0.63696 |  |  |
| residual_over_constant_velocity_with_multistep_loss | 1 | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 3.38581 | 10.23662 | 3.38581 | 10.23662 | 26.81826 | 80.91337 | 26.81826 | 80.91337 | 115.40368 | 347.50721 | 115.40368 | 347.50721 | 479.45455 | 1442.48877 | 479.45455 | 1442.48877 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | None | 0.0 | 0.967 | 0.9769 |  |  |
| best_model_SMC |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | premature | deterministic learned model is not competitive with strongest causal baseline |

## Gates

See `outputs/reports/world_model_gate_stage4p5.md`.

## Direct Conclusions

项目是否跑通：
是

是否修复 dt / velocity / coordinate 问题：
部分

正式 benchmark 是否使用 causal velocity：
是

是否存在 velocity leakage 风险：
不确定；native `_kf` velocity 可能经过平滑，所以不作为正式 causal score。

TGSIM 主要 agent 类型：
3.0

当前数据更像 human crowd 还是 traffic trajectory：
traffic / generic trajectory benchmark

constant velocity 为什么这么强：
TGSIM quick endpoint 的测试片段非常平滑，很多 agent 近似静止或短期匀速；使用 causal dt 后，惯性模型已经解释大部分 t+100 位移。

hand physics 为什么失败：
Stage 4 在没有真实 goal、exit、obstacle、walkable boundary 的情况下施加 human crowd social-force / goal attraction，把平滑 traffic trajectory 推离真实路径。

learned residual 为什么失败或改善：
Stage 4.5 改为 residual over inertial baselines，但 quick 版仍是线性 residual，且真实场景缺 route/intent/geometry 标注；如果它未超过 strongest causal baseline，就说明 residual 仍在过拟合局部噪声而不是学习稳定路线意图。

最强 causal baseline：
constant_turn_rate_velocity + FDE@100=0.0482

最强 learned model：
residual_over_constant_acceleration + FDE@100=1.14498

learned model 是否超过最强 causal baseline：
否

超过幅度：
-22.755

是否值得进入 Stage 5 latent generative：
否

如果不值得：
先修 type-specific traffic dynamics、真实 route/goal labels、多步 rollout training、以及 intent-aware proposal；不要扩大生成模型。

如果值得：
当前不值得。

当前 verdict：
prototype_with_repaired_baselines_but_failed_learned_dynamics_gate

expert audit score：
64
