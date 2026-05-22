# Stage 13 Final Report

Stage 13 ran the first overnight deterministic repair loop. It executed bounded deterministic residual training/search, benchmarked the trials, ran Stage 13 gates, and mined failures. It did not enable latent generative modeling and did not enable SMC.

## Direct Answers

1. 本轮是否真的执行了训练，而不是只 planned：是。
2. 跑了多少 trials：24。
3. 哪个模型最好：`residual_no_alpha` for the best hard/failure t+50 aggregate.
4. eth_ucy_ewap t+100 是否改善：不可评估于 Stage 13 per-agent causal mask；没有可评估 t+100 rows。
5. HardBench 是否改善：部分，best improvement = 0.013127，未达到 10% gate。
6. BaselineFailureBench 是否改善：部分，best improvement = 0.013127，未达到 10% gate。
7. Easy subset 是否保持：是。
8. Scene/goal 是否有效：否，未证明优于 no-scene/no-goal。
9. Interaction 是否有效：否，interaction family 不优于 no-interaction。
10. Stage 5C 是否 ready：否。
11. SMC 是否 ready：否。

## Gate Summary

Stage 13 gates passed 5 / 12. Data, no-leakage, strong-baseline, easy-preservation, and physical-validity gates passed. EWAP t+100, HardBench, BaselineFailureBench, scene/goal, interaction, Stage 5C, and SMC gates failed.

## Honest Conclusion

Stage 13 moved the system from a planned auto-loop to an actually executed deterministic repair loop. However, it also exposed a stricter blocker: Stage 12 had source-level t+100 coverage, but Stage 13 per-agent causal-mask evaluation produced no evaluable EWAP t+100 rows. Therefore the project must not claim pedestrian t+100 improvement.

项目是否跑通：是
overnight loop 是否真正执行：是
训练 trial 数：24
best model：residual_no_alpha
best eth_ucy_ewap t+100 improvement：not_evaluable_under_stage13_per_agent_mask
best HardBench improvement：0.013127
best BaselineFailureBench improvement：0.013127
easy preservation：pass
latent generative ready：否
SMC ready：否
current verdict：stage13_deterministic_repair_loop_executed_not_stage5c_ready
expert audit score：84

verdict = stage13_deterministic_repair_loop_executed_not_stage5c_ready

## Next Fixes

1. Fix or rebuild Stage 12 EWAP t+100 episode construction so per-agent causal past and t+100 target masks are evaluable.
2. Continue deterministic repair with stronger fallback-to-baseline discipline and hard/failure weighting.
3. Add more verified pedestrian/drone long-horizon data, especially SDD/OpenTraj if the user provides legal local paths.
