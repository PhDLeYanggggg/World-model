# Stage 9 Final Report

Stage 9 trains deterministic per-agent multi-agent scene-grounded residual models. It does not enable latent generative modeling or SMC.

## Direct Answers

1. 是否训练了 per-agent multi-agent world model：是
2. 是否预测所有 agents，而不是只预测 primary agent：是
3. full model 是否超过 strongest causal baseline：否 (mean all-test improvement=-0.001592)
4. hard/failure subset 是否超过 baseline：否 (best hard/failure improvement=0.000537)
5. easy subset 是否保持：否/不可充分评估
6. interaction 是否真正提升轨迹预测：否 (gain=-0.000414)
7. scene/goal 是否真正提升轨迹预测：否 (gain=-0.000005)
8. per-agent model 是否比 primary/simple model 更好：否 (ge5 gain=-0.003381)
9. 是否仍缺 pedestrian/drone t+50/t+100：是
10. 是否可以进入 Stage 5C latent generative：否
11. 是否可以启用 SMC：否
12. 当前是否仍只是 trajectory forecasting scaffold：是，但已经是 per-agent all-agent 形式。
13. 当前是否更接近 world model：部分，更接近 scene/goal/multi-agent state-space scaffold，但 deterministic gates 未过。

## Final Conclusion

项目是否跑通：是
per-agent multi-agent model 是否训练：是
是否预测所有 agents：是
per-agent model 是否超过 strongest causal baseline：否
hard/failure subset 是否改善：部分
easy subset 是否保持：否/不可充分评估
interaction 是否有效：否
scene/goal 是否有效：否
multi-agent 是否优于 primary-agent：否
pedestrian/drone t+50/t+100 是否仍缺：是
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage9_per_agent_training_done_not_stage5c_ready
expert audit score：75

如果不能进入 Stage 5C，下一步先修什么：

1. 接入 verified pedestrian/drone t+50/t+100 数据，优先 SDD/OpenTraj，并保留 homography/scale 状态。
2. 将 silver scene annotations 升级为人工确认 gold，并补 walkable/exit/goal/obstacle 标注。
3. 改进 per-agent residual：加入更稳的 failure-aware gating、按 agent 类型/场景分层训练，并解决 easy subset 可评估性。
