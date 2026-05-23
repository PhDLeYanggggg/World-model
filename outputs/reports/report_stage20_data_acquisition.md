# Stage 20 Data Acquisition Report

## Honest Status

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。
- BPSG-MA World Model v1 已交付，但部署策略仍是 strongest causal baseline fallback + diagnostics。
- Stage 18 SAM-JEPA-2.5D 是 representation pretraining，不是 latent generative rollout。
- Stage 18 JEPA non-collapse，但没有改善 selector / failure predictor / correction / official t+50。
- Stage 5C latent generative 仍不 ready。
- SMC 仍不 ready。
- 当前最大瓶颈仍是 raw scene image/video + trajectory + long-horizon pedestrian/drone data。

## Direct Answers

1. 上网找到了多少候选数据源？`33`
2. 哪些是官方来源？`31` 个记录 official_url；详见 registry。
3. 哪些可以合法自动下载？默认 dry-run 下未自动下载；只有 project-generated simulation/local tooling 可安全自动生成或索引。
4. 哪些需要用户手动下载或申请？`2` 个高优先源优先需要用户操作，前三名见结论。
5. 哪些已经在本地找到？本地路径验证 found_paths=`6`。
6. 哪些成功转换？light raw-index/conversion sources=`5`；不等同 full benchmark。
7. 哪些能 t+50？本地候选 t+50 sources=`5`；Stage20 actual verified 新增=`0`，仍需 Stage21 episode conversion。
8. 哪些能 t+100？本地候选 t+100 sources=`5`；Stage20 actual verified 新增=`0`，estimated 不等于 actual verified。
9. 哪些有 raw video / scene image？详见 registry `has_raw_video` / `has_scene_images`。
10. 哪些有 trajectories？详见 registry `has_trajectories`。
11. 哪些有 homography / metric？详见 registry `has_homography` / `has_metric_coordinates`。
12. 哪些能作为 official benchmark？仅真实 top-down trajectory 且本地转换/审计通过者；当前 Stage20 新增 official topdown count=`0`。
13. 哪些只能做 JEPA pretraining？Ego4D/Ego-Exo4D/EPIC/HoloAssist/Assembly101/HOI4D 等 egocentric/human video。
14. 哪些只能做 simulation / diagnostic？simulation_* 和 traffic/driving 类。
15. 哪些数据最应该用户下一步提供？`['full ETH/UCY / EWAP', 'AerialMPT longer sequences']`。
16. 下一步是否可以扩大 Stage 18/19 JEPA？可以扩大 registry/本地验证后的 JEPA 数据，但不能包装成 downstream success。
17. 下一步是否可以重跑 deterministic head？只有 Stage21 完成 full conversion/no-leakage 后才值得重跑。
18. 是否可以进入 Stage 5C？`否`。
19. 是否可以启用 SMC？`否`。

## Final Conclusion

项目是否跑通：是
web search 是否完成：是
数据 registry 是否更新：是
合法下载计划是否完成：是
成功自动下载数据源数量：0
成功验证本地路径数量：6
成功转换数据源数量：5
新增 official top-down benchmark 数据源：0
新增 JEPA pretraining 数据源：3
新增 t+50 数据源：0
新增 t+100 数据源：0
是否需要用户提供数据：是
最需要用户提供的数据前三名：['full ETH/UCY / EWAP', 'AerialMPT longer sequences']
是否允许 Stage 5C：否
是否允许 SMC：否
当前 verdict：stage20_web_dataset_acquisition_package_built_stage5c_blocked
expert audit score：92
