# Stage 8 Final Report

Stage 8 upgrades the scaffold toward a scene/goal-grounded pedestrian world model, but it does not make the project a true 3D or large-scale foundation world model.

## Current Status

- The system is still a multi-source 2.5D/pseudo-3D trajectory world-state benchmark scaffold.
- Latent generative modeling remains disabled.
- SMC remains disabled.
- Traffic t+100 results are not pedestrian world-model evidence.
- Inferred goals are not true goals.

## Direct Answers

1. pedestrian/drone t+50/t+100 replenished: no
2. gold/silver scene annotations: 0
3. true multi-agent episodes built: yes (78 episodes with >=2 agents)
4. GoalBench-Gold vs majority: top1=0.5, majority_top1=0.333333; top3=0.75, majority_top3=0.666667
5. Stage 8 failure predictor best AUROC: 0.896021
6. BaselineFailureBench best improvement: -0.001179
7. HardBench best improvement: 0.000266
8. interaction encoder trajectory evidence: {'graph_improves_over_scalar_auxiliary': False, 'improves_hard_failure_trajectory_performance': False, 'reason': 'Stage 8 interaction auxiliary is diagnostic unless benchmark metrics show trajectory lift.'}
9. verified long-horizon best improvement: 0.000266
10. Stage 5C ready: False
11. SMC ready: False

## Final Conclusion

项目是否跑通：是
pedestrian/drone long-horizon 是否补上：否
scene-gold annotation 是否建立：部分
multi-agent episodes 是否建立：是
GoalBench-Gold 是否有效：是
failure predictor 是否改善：否
goal-conditioned world model 是否改善 failure/hard cases：部分
interaction encoder 是否有效：否
verified long-horizon 是否改善：否
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage8_scene_goal_multiagent_scaffold_not_stage5c_ready
expert audit score：71

如果不能进入 Stage 5C，下一步先修什么：

1. 提供或接入本地 Stanford Drone Dataset / OpenTraj 原始数据，并确认 license 与路径。
2. 对至少一个 pedestrian/drone scene 做 gold/silver exit/goal/walkable annotation，而不是只用 endpoint inferred goals。
3. 把 world model 从 primary-agent residual 升级为 per-agent multi-agent residual，并在 HardBench/BaselineFailureBench 上稳定超过 strongest causal baseline。
