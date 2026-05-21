# Stage 12 Final Report

Stage 12 is a pedestrian/drone data acquisition, human/silver annotation, long-horizon audit, and deterministic re-benchmark preparation stage. It does not enable latent generative modeling or SMC.

## Direct Answers

1. 是否接入真实 pedestrian/drone 数据：是 (['eth_ucy_ewap', 'aerialmpt', 'full_trajnet_original_quick'])
2. 是否补上 verified t+50/t+100：是 (['eth_ucy_ewap'])
3. 是否建立 human-confirmed gold/silver annotations：是 (human_confirmed=3)
4. 是否仍主要依赖 rule-confirmed silver：是 (silver_rule_confirmed=33)
5. 是否建立 usable scene packs：是 (scene_packs_with_goals=43)
6. 是否扩展 multi-agent episodes：是 (episodes_ge2=660)
7. 是否扩展 hard/failure episodes：是 (records=649)
8. GoalBench v4 official records 是否足够：是 (official=5574)
9. 是否可以进入 Stage 13 training：是
10. 是否仍禁止 Stage 5C latent generative：是
11. 是否仍禁止 SMC：是

## Deterministic Re-benchmark

Stage 12 gates allowed deterministic re-benchmarking, so a per-agent deterministic residual model was retrained/evaluated on Stage 12 episodes.

| dataset | target horizon | best learned variant | FDE | strongest baseline FDE | improvement |
| --- | --- | --- | --- | --- | --- |
| aerialmpt | t+5 | per_agent_no_scene | 2.746454 | 2.735528 | -0.003994 |
| eth_ucy | t+10 | per_agent_scene_only | 5.157833 | 5.17126 | 0.002596 |
| eth_ucy_ewap | t+100 | per_agent_no_scene | 5.460224 | 5.460224 | 0.0 |
| trajnet | t+10 | per_agent_goal_only | 20.748859 | 20.771859 | 0.001107 |

结论：数据门槛显著改善，尤其是 `eth_ucy_ewap` 提供了 verified pedestrian t+50/t+100；但 deterministic learned residual 仍没有稳定超过 strongest causal baseline 5%。因此 Stage 13 可以继续做 deterministic model repair/training，Stage 5C latent generative 和 SMC 仍不能启用。

## Final Conclusion

项目是否跑通：是
pedestrian/drone 数据是否接入：是
verified pedestrian/drone t+50/t+100 是否补上：是
human-confirmed annotation 是否建立：是
scene packs 是否可用于 official training：是
multi-agent episodes 是否足够：是
hard/failure episodes 是否足够：是
GoalBench v4 是否足够：是
是否可以进入 Stage 13：是
是否可以进入 Stage 5C latent generative：否
是否可以启用 SMC：否
当前 verdict：stage12_ready_for_stage13_training_with_long_horizon_source
expert audit score：83
deterministic learned model 是否超过 strongest causal baseline：否

如果不能进入 Stage 13，下一步先修什么：

1. 提供 Stanford Drone Dataset / OpenTraj 本地路径，补更多 scene images 和 verified pedestrian/drone long-horizon samples。
2. 将更多 silver_rule_confirmed scene annotations 升级为 silver_human_confirmed 或 gold_human。
3. 加强 deterministic per-agent residual 模型，但只在 Stage 12 gates 允许后进行。
