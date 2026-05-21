# Stage 4 Real Long-Trajectory Benchmark

## Required Current-State Admission

1. 当前模型是 pseudo-3D physics-informed learned residual state-space world model。
2. 它不是真 3D。
3. 它不是 exceptional world model。
4. 当前 expert audit score 是 58/100。
5. 当前 verdict 是 prototype_with_major_failures。
6. Synthetic t+100 已经可验证。
7. AerialMPT bauma3 t+100 仍然只能 qualitative free-run，不能报 ADE@100/FDE@100。
8. learned residual 没有显著超过 hand physics。
9. SMC 没有明显提升 coverage。
10. 真实长轨迹 t+100 benchmark 在本报告中按实际数据接入结果判定。

## Real Data Summary

| dataset_name | total_scenes | total_agents | total_tracks | total_frames | mean_track_length | coordinate_unit | whether_metric_coordinates | whether_scene_geometry_available | samples_t10 | samples_t25 | samples_t50 | samples_t100 | whether_t100_verified | build_horizon | cannot_evaluate_t100 | train_episodes | val_episodes | test_episodes | mean_agents_per_episode | split_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TGSIM Foggy Bottom | 1 | 119 | 119 | 24941 | 420.168 | meter | True | False | 7401 | 2390 | 1128 | 482 | True | 100 | None | 7 | 2 | 3 | 2.583 | scene split when possible; single-scene datasets use chronological non-overlapping windows |

## Training

| trained | samples | feature_dim | target | residual_std |
| --- | --- | --- | --- | --- |
| True | 1600 | 29 | real residual acceleration = true_next_acceleration - hand_physics_acceleration | [0.5373420119285583, 0.6271871328353882] |

## Metrics

| model | branch_count | ADE@1 | FDE@1 | minFDE@N@1 | ADE@10 | FDE@10 | minFDE@N@10 | ADE@25 | FDE@25 | minFDE@N@25 | ADE@50 | FDE@50 | minFDE@N@50 | ADE@100 | FDE@100 | minFDE@N@100 | coverage_FDE_lt_1m | coverage_FDE_lt_2m | coverage_FDE_lt_5m | coverage_FDE_lt_10m | collision_violation_rate | boundary_violation_rate | physical_validity_rate | cluster_diversity_score | NLL_endpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_velocity_baseline | 1 | 0.00924 | 0.00924 | 0.00924 | 0.05374 | 0.09945 | 0.09945 | 0.12907 | 0.25029 | 0.25029 | 0.25642 | 0.50351 | 0.50351 | 0.50846 | 1.00923 | 1.00923 | 0.66667 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 2.9113 |
| hand_physics_baseline | 1 | 1.00923 | 1.00923 | 1.00923 | 5.13678 | 9.28339 | 9.28338 | 11.69082 | 19.14802 | 19.14802 | 15.65688 | 20.05286 | 20.05286 | 17.63117 | 20.05663 | 20.05663 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 92.19593 |
| deterministic_neural_residual | 1 | 0.75068 | 0.75068 | 0.75068 | 4.53455 | 8.46647 | 8.46647 | 11.31938 | 22.73926 | 22.73926 | 20.92622 | 34.83946 | 34.83946 | 28.23922 | 35.6023 | 35.6023 | 0.0 | 0.0 | 0.0 | 0.0 | 0.65347 | 0.0 | 0.34653 | 0.0 | 298.78826 |
| stochastic_neural_residual | 8 | 0.66945 | 0.66945 | 0.75415 | 3.95858 | 7.74054 | 8.00054 | 10.675 | 21.57206 | 20.4506 | 19.36188 | 31.15392 | 33.40725 | 25.6339 | 33.00006 | 31.93581 | 0.0 | 0.0 | 0.0 | 0.0 | 0.64233 | 0.0 | 0.35767 | 0.33349 | 249.39333 |
| physics_plus_neural_residual | 1 | 0.75068 | 0.75068 | 0.75068 | 4.53455 | 8.46647 | 8.46647 | 11.31938 | 22.73926 | 22.73926 | 20.92622 | 34.83946 | 34.83946 | 28.23922 | 35.6023 | 35.6023 | 0.0 | 0.0 | 0.0 | 0.0 | 0.65347 | 0.0 | 0.34653 | 0.0 | 298.78826 |
| hand_physics_SMC | 16 | 0.98603 | 0.98603 | 0.94417 | 5.01781 | 9.06681 | 8.75996 | 11.52486 | 19.84641 | 19.35959 | 15.64539 | 19.70777 | 19.40675 | 17.70542 | 19.78324 | 19.43796 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.04724 | 86.80411 |
| physics_plus_neural_residual_SMC | 16 | 0.67466 | 0.67466 | 0.63742 | 4.00797 | 7.63058 | 7.75202 | 10.58723 | 21.41813 | 20.75811 | 19.32502 | 31.53205 | 31.58904 | 26.10303 | 34.05537 | 31.96071 | 0.0 | 0.0 | 0.0 | 0.0 | 0.64666 | 0.0 | 0.35334 | 0.3339 | 249.57177 |

## Gates

See `outputs/reports/world_model_gate_stage4.md`.

## Direct Conclusions

项目是否跑通：
是

接入真实长轨迹数据：
是

真实数据名称：
TGSIM Foggy Bottom

真实数据是否支持 t+100 verified evaluation：
是

当前模型是否仍是 pseudo-3D：
是

是否是真 3D：
否

learned residual 是否超过 hand physics：
否

超过幅度：
hand ADE@100=17.63117, best learned ADE@100=25.6339; hand FDE@100=20.05663, best learned FDE@100=33.00006

SMC 是否提升 coverage：
否

coverage_FDE_lt_5m：
0.0

best_of_N_FDE@100：
minFDE@16=31.96071

physical validity 是否可接受：
否

terminal clusters 是否有语义差异：
弱

expert audit score：
58

是否超过 70：
否

当前 verdict：
prototype_with_major_failures

是否值得进入 latent generative model Stage 5：
否

如果值得，原因：
当前不值得；真实 t+100、learned dynamics、coverage gate 尚未同时通过。

如果不值得，先修什么：
先修真实数据上的 learned dynamics：使用多步 rollout loss、按行人/车辆类型分层训练、引入真实 scene geometry/goal labels，并让 SMC proposal 表达 latent intent 而不是只加局部噪声。
