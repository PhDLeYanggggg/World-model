# Stage 7 Final Report

Stage 7 upgraded the scaffold from pure trajectory residuals toward scene/goal-grounded deterministic prediction.

## Honest Current State

1. 当前不是 true 3D world model.
2. 当前不是 large-scale foundation world model.
3. 当前仍是 multi-source trajectory world-state benchmark scaffold.
4. Stage 7 不启用 latent generative，不启用 SMC.
5. traffic t+100 不能包装成 pedestrian world model.

## What Changed

- Built inferred scene packs with walkable bbox, boundary SDF, candidate goals, and route hypotheses.
- Built GoalBench from scene-level candidate goals and future endpoint labels for training/evaluation only.
- Trained a causal goal/intent predictor.
- Trained goal/scene-conditioned baseline failure predictors.
- Trained deterministic goal-conditioned gated residual variants.
- Added interaction auxiliary diagnostics without claiming graph-interaction success.

## Key Benchmark Rows
| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention | false_intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goal_only_residual | eth_ucy | all | 10 | 0.569134 | 0.713643 | 0.202495 | 6 | 0.158914 | 0.25 | 0.083333 |
| goal_only_residual | eth_ucy | easy | 10 | 0.028436 | 0.0 | -0.284361 | 1 | 0.040886 | 0.0 | 0.0 |
| goal_only_residual | eth_ucy | hard | 10 | 1.319933 | 1.746128 | 0.24408 | 2 | 0.208692 | 0.5 | 0.0 |
| goal_only_residual | eth_ucy | baseline_failure | 10 | 1.319933 | 1.746128 | 0.24408 | 2 | 0.208692 | 0.5 | 0.0 |
| goal_only_residual | eth_ucy | scene_grounded | 10 | 0.569134 | 0.713643 | 0.202495 | 6 | 0.158914 | 0.25 | 0.083333 |
| goal_only_residual | eth_ucy | pedestrian_drone | 10 | 0.569134 | 0.713643 | 0.202495 | 6 | 0.158914 | 0.25 | 0.083333 |
| goal_only_residual | tgsim | all | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | scene_grounded | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | traffic | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | verified_t50 | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim | verified_t100 | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| goal_only_residual | tgsim_i90 | all | 100 | 12.204016 | 10.327657 | -0.181683 | 6 | 0.321674 | 0.6 | 0.1 |
| goal_only_residual | tgsim_i90 | hard | 100 | 4.29189 | 6.388728 | 0.328209 | 1 | 0.344064 | 0.6 | 0.0 |
| goal_only_residual | tgsim_i90 | baseline_failure | 100 | 13.5678 | 11.775393 | -0.152216 | 5 | 0.316118 | 0.6 | 0.0 |
| goal_only_residual | tgsim_i90 | scene_grounded | 100 | 12.204016 | 10.327657 | -0.181683 | 6 | 0.321674 | 0.6 | 0.1 |
| goal_only_residual | tgsim_i90 | traffic | 100 | 12.204016 | 10.327657 | -0.181683 | 6 | 0.321674 | 0.6 | 0.1 |
| goal_only_residual | tgsim_i90 | verified_t50 | 100 | 12.204016 | 10.327657 | -0.181683 | 6 | 0.321674 | 0.6 | 0.1 |
| goal_only_residual | tgsim_i90 | verified_t100 | 100 | 12.204016 | 10.327657 | -0.181683 | 6 | 0.321674 | 0.6 | 0.1 |
| goal_only_residual | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| goal_only_residual | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 | 0.0 |
| goal_only_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_only_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| goal_only_residual | trajnet | scene_grounded | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| goal_only_residual | trajnet | pedestrian_drone | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| scene_only_residual | eth_ucy | all | 10 | 0.582876 | 0.713643 | 0.183239 | 6 | 0.124989 | 0.166667 | 0.083333 |
| scene_only_residual | eth_ucy | easy | 10 | 0.003398 | 0.0 | -0.033983 | 1 | 0.025 | 0.0 | 0.0 |
| scene_only_residual | eth_ucy | hard | 10 | 1.403177 | 1.746128 | 0.196407 | 2 | 0.117042 | 0.25 | 0.0 |
| scene_only_residual | eth_ucy | baseline_failure | 10 | 1.403177 | 1.746128 | 0.196407 | 2 | 0.117042 | 0.25 | 0.0 |
| scene_only_residual | eth_ucy | scene_grounded | 10 | 0.582876 | 0.713643 | 0.183239 | 6 | 0.124989 | 0.166667 | 0.083333 |
| scene_only_residual | eth_ucy | pedestrian_drone | 10 | 0.582876 | 0.713643 | 0.183239 | 6 | 0.124989 | 0.166667 | 0.083333 |
| scene_only_residual | tgsim | all | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | hard | 100 | 11.995052 | 11.995052 | 0.0 | 2 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | baseline_failure | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | scene_grounded | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | traffic | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | verified_t50 | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim | verified_t100 | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.0082 | 0.0 | 0.0 |
| scene_only_residual | tgsim_i90 | all | 100 | 10.40928 | 10.327657 | -0.007903 | 6 | 0.339407 | 0.6 | 0.1 |
| scene_only_residual | tgsim_i90 | hard | 100 | 4.246666 | 6.388728 | 0.335288 | 1 | 0.398127 | 0.6 | 0.0 |
| scene_only_residual | tgsim_i90 | baseline_failure | 100 | 11.438377 | 11.775393 | 0.02862 | 5 | 0.336895 | 0.6 | 0.0 |
| scene_only_residual | tgsim_i90 | scene_grounded | 100 | 10.40928 | 10.327657 | -0.007903 | 6 | 0.339407 | 0.6 | 0.1 |
| scene_only_residual | tgsim_i90 | traffic | 100 | 10.40928 | 10.327657 | -0.007903 | 6 | 0.339407 | 0.6 | 0.1 |
| scene_only_residual | tgsim_i90 | verified_t50 | 100 | 10.40928 | 10.327657 | -0.007903 | 6 | 0.339407 | 0.6 | 0.1 |
| scene_only_residual | tgsim_i90 | verified_t100 | 100 | 10.40928 | 10.327657 | -0.007903 | 6 | 0.339407 | 0.6 | 0.1 |
| scene_only_residual | trajnet | all | 10 | 1.445667 | 1.434586 | -0.007724 | 7 | 0.05 | 0.0 | 0.0 |
| scene_only_residual | trajnet | easy | 10 | 0.242769 | 0.18701 | -0.298164 | 2 | 0.05 | 0.0 | 0.0 |
| scene_only_residual | trajnet | hard | 10 | 2.392291 | 2.399685 | 0.003082 | 4 | 0.05 | 0.0 | 0.0 |
| scene_only_residual | trajnet | baseline_failure | 10 | 2.392291 | 2.399685 | 0.003082 | 4 | 0.05 | 0.0 | 0.0 |
| scene_only_residual | trajnet | scene_grounded | 10 | 1.445667 | 1.434586 | -0.007724 | 7 | 0.05 | 0.0 | 0.0 |
| scene_only_residual | trajnet | pedestrian_drone | 10 | 1.445667 | 1.434586 | -0.007724 | 7 | 0.05 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | all | 10 | 0.674653 | 0.713643 | 0.054635 | 6 | 0.03987 | 0.083333 | 0.083333 |
| interaction_scalar_residual | eth_ucy | easy | 10 | 0.007224 | 0.0 | -0.072244 | 1 | 0.009954 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | hard | 10 | 1.691571 | 1.746128 | 0.031244 | 2 | 0.019207 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | baseline_failure | 10 | 1.691571 | 1.746128 | 0.031244 | 2 | 0.019207 | 0.0 | 0.0 |
| interaction_scalar_residual | eth_ucy | scene_grounded | 10 | 0.674653 | 0.713643 | 0.054635 | 6 | 0.03987 | 0.083333 | 0.083333 |
| interaction_scalar_residual | eth_ucy | pedestrian_drone | 10 | 0.674653 | 0.713643 | 0.054635 | 6 | 0.03987 | 0.083333 | 0.083333 |
| interaction_scalar_residual | tgsim | all | 100 | 6.101541 | 6.062032 | -0.006517 | 4 | 0.132282 | 0.2 | 0.1 |
| interaction_scalar_residual | tgsim | hard | 100 | 12.033661 | 11.995052 | -0.003219 | 2 | 0.215999 | 0.2 | 0.0 |
| interaction_scalar_residual | tgsim | baseline_failure | 100 | 21.036198 | 20.967461 | -0.003278 | 1 | 0.403995 | 0.4 | 0.0 |
| interaction_scalar_residual | tgsim | scene_grounded | 100 | 6.101541 | 6.062032 | -0.006517 | 4 | 0.132282 | 0.2 | 0.1 |
| interaction_scalar_residual | tgsim | traffic | 100 | 6.101541 | 6.062032 | -0.006517 | 4 | 0.132282 | 0.2 | 0.1 |
| interaction_scalar_residual | tgsim | verified_t50 | 100 | 6.101541 | 6.062032 | -0.006517 | 4 | 0.132282 | 0.2 | 0.1 |
| interaction_scalar_residual | tgsim | verified_t100 | 100 | 6.101541 | 6.062032 | -0.006517 | 4 | 0.132282 | 0.2 | 0.1 |
| interaction_scalar_residual | tgsim_i90 | all | 100 | 10.256576 | 10.327657 | 0.006883 | 6 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | hard | 100 | 6.20554 | 6.388728 | 0.028674 | 1 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | baseline_failure | 100 | 11.681197 | 11.775393 | 0.007999 | 5 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | scene_grounded | 100 | 10.256576 | 10.327657 | 0.006883 | 6 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | traffic | 100 | 10.256576 | 10.327657 | 0.006883 | 6 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | verified_t50 | 100 | 10.256576 | 10.327657 | 0.006883 | 6 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | tgsim_i90 | verified_t100 | 100 | 10.256576 | 10.327657 | 0.006883 | 6 | 0.0282 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | scene_grounded | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| interaction_scalar_residual | trajnet | pedestrian_drone | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 | 0.0 |
| goal_interaction_residual | eth_ucy | all | 10 | 0.712771 | 0.713643 | 0.001223 | 6 | 0.098017 | 0.166667 | 0.083333 |
| goal_interaction_residual | eth_ucy | easy | 10 | 0.006343 | 0.0 | -0.063433 | 1 | 0.032436 | 0.0 | 0.0 |

## Direct Answers

pedestrian/drone long horizon 是否补上：否.
scene packs 是否建立：是；数量=4.
candidate goals 是否建立：是.
GoalBench 是否有意义：是 .
goal predictor 是否超过 majority baseline：否/部分 .
goal/scene-conditioned failure predictor 是否超过 Stage 6：是 .
goal-conditioned residual 是否在 BaselineFailureBench 上赢：是 .
goal-conditioned residual 是否在 HardBench-v1 上赢：是 .
easy subset 是否没有被破坏：是 .
interaction auxiliary tasks 是否有效：否/diagnostic only .
verified long-horizon 是否改善：否 .
是否可以进入 latent generative Stage 5C：否.
是否可以启用 SMC：否.
当前是否仍只是 trajectory forecasting scaffold：是，但现在加入了 scene/goal grounding.
当前是否更接近 world model：部分，更接近 scene/goal-grounded state-space model，但不是 true 3D.

## Final Verdict

项目是否跑通：是
scene/goal grounding 是否建立：是
pedestrian/drone long-horizon 是否补上：否
GoalBench 是否可靠：部分
goal predictor 是否有效：部分/弱
goal-conditioned failure predictor 是否有效：是
goal-conditioned residual 是否有效：是
BaselineFailureBench 是否改善：是
HardBench 是否改善：是
verified long-horizon 是否改善：否
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage7_scene_goal_grounding_built_but_not_stage5c_ready
expert audit score：71
如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone scene+homography+t50/t100；人工/半自动 walkable/exit/goal 标注；多智能体 episodes 而不是 single-primary-agent windows.
