# Stage 6 Final Report

Stage 6 built HardBench-v1, BaselineFailureBench, a causal baseline-failure predictor, and a deterministic failure-aware gated residual model. It still does not unlock Stage 5C.

## Honest Current State

1. 当前不是 true 3D world model.
2. 当前不是 large-scale foundation world model.
3. 当前仍是 multi-source trajectory world-state benchmark scaffold.
4. 不启用 latent generative，不启用 SMC.
5. traffic t+100 不能包装成 pedestrian world model.

## Key Metrics

| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha | intervention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| failure_predictor_only_gate | eth_ucy | all | 10 | 0.713643 | 0.713643 | 0.0 | 6 | 0.0 | 0.0 |
| failure_predictor_only_gate | eth_ucy | easy | 10 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 0.0 |
| failure_predictor_only_gate | eth_ucy | hard | 10 | 1.746128 | 1.746128 | 0.0 | 2 | 0.0 | 0.0 |
| failure_predictor_only_gate | eth_ucy | baseline_failure | 10 | 1.746128 | 1.746128 | 0.0 | 2 | 0.0 | 0.0 |
| failure_predictor_only_gate | tgsim | all | 100 | 6.12394 | 6.062032 | -0.010212 | 4 | 0.041716 | 0.2 |
| failure_predictor_only_gate | tgsim | hard | 100 | 12.052359 | 11.995052 | -0.004778 | 2 | 0.045844 | 0.2 |
| failure_predictor_only_gate | tgsim | baseline_failure | 100 | 21.068335 | 20.967461 | -0.004811 | 1 | 0.06021 | 0.2 |
| failure_predictor_only_gate | tgsim | verified_t50 | 100 | 6.12394 | 6.062032 | -0.010212 | 4 | 0.041716 | 0.2 |
| failure_predictor_only_gate | tgsim | verified_t100 | 100 | 6.12394 | 6.062032 | -0.010212 | 4 | 0.041716 | 0.2 |
| failure_predictor_only_gate | tgsim_i90 | all | 100 | 10.214488 | 10.327657 | 0.010958 | 6 | 0.044 | 0.0 |
| failure_predictor_only_gate | tgsim_i90 | hard | 100 | 6.096299 | 6.388728 | 0.045773 | 1 | 0.044 | 0.0 |
| failure_predictor_only_gate | tgsim_i90 | baseline_failure | 100 | 11.622993 | 11.775393 | 0.012942 | 5 | 0.044 | 0.0 |
| failure_predictor_only_gate | tgsim_i90 | verified_t50 | 100 | 10.214488 | 10.327657 | 0.010958 | 6 | 0.044 | 0.0 |
| failure_predictor_only_gate | tgsim_i90 | verified_t100 | 100 | 10.214488 | 10.327657 | 0.010958 | 6 | 0.044 | 0.0 |
| failure_predictor_only_gate | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 |
| failure_predictor_only_gate | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 |
| failure_predictor_only_gate | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| failure_predictor_only_gate | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| learned_alpha_gate | eth_ucy | all | 10 | 0.636288 | 0.713643 | 0.108395 | 6 | 0.046228 | 0.166667 |
| learned_alpha_gate | eth_ucy | easy | 10 | 0.007458 | 0.0 | -0.074583 | 1 | 0.010655 | 0.0 |
| learned_alpha_gate | eth_ucy | hard | 10 | 1.57341 | 1.746128 | 0.098915 | 2 | 0.036182 | 0.25 |
| learned_alpha_gate | eth_ucy | baseline_failure | 10 | 1.57341 | 1.746128 | 0.098915 | 2 | 0.036182 | 0.25 |
| learned_alpha_gate | tgsim | all | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.139822 | 0.25 |
| learned_alpha_gate | tgsim | hard | 100 | 12.052357 | 11.995052 | -0.004777 | 2 | 0.222112 | 0.3 |
| learned_alpha_gate | tgsim | baseline_failure | 100 | 21.068338 | 20.967461 | -0.004811 | 1 | 0.405995 | 0.4 |
| learned_alpha_gate | tgsim | verified_t50 | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.139822 | 0.25 |
| learned_alpha_gate | tgsim | verified_t100 | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.139822 | 0.25 |
| learned_alpha_gate | tgsim_i90 | all | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.05405 | 0.0 |
| learned_alpha_gate | tgsim_i90 | hard | 100 | 6.097229 | 6.388728 | 0.045627 | 1 | 0.05405 | 0.0 |
| learned_alpha_gate | tgsim_i90 | baseline_failure | 100 | 11.614855 | 11.775393 | 0.013633 | 5 | 0.05405 | 0.0 |
| learned_alpha_gate | tgsim_i90 | verified_t50 | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.05405 | 0.0 |
| learned_alpha_gate | tgsim_i90 | verified_t100 | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.05405 | 0.0 |
| learned_alpha_gate | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 |
| learned_alpha_gate | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 |
| learned_alpha_gate | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| learned_alpha_gate | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | eth_ucy | all | 10 | 0.68748 | 0.713643 | 0.036662 | 6 | 0.025675 | 0.083333 |
| hybrid_failure_predictor_plus_learned_gate | eth_ucy | easy | 10 | 0.009843 | 0.0 | -0.098428 | 1 | 0.013861 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | eth_ucy | hard | 10 | 1.684307 | 1.746128 | 0.035404 | 2 | 0.023387 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | eth_ucy | baseline_failure | 10 | 1.684307 | 1.746128 | 0.035404 | 2 | 0.023387 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | tgsim | all | 100 | 6.124096 | 6.062032 | -0.010238 | 4 | 0.057194 | 0.2 |
| hybrid_failure_predictor_plus_learned_gate | tgsim | hard | 100 | 12.052214 | 11.995052 | -0.004765 | 2 | 0.070597 | 0.2 |
| hybrid_failure_predictor_plus_learned_gate | tgsim | baseline_failure | 100 | 21.068108 | 20.967461 | -0.0048 | 1 | 0.107119 | 0.2 |
| hybrid_failure_predictor_plus_learned_gate | tgsim | verified_t50 | 100 | 6.124096 | 6.062032 | -0.010238 | 4 | 0.057194 | 0.2 |
| hybrid_failure_predictor_plus_learned_gate | tgsim | verified_t100 | 100 | 6.124096 | 6.062032 | -0.010238 | 4 | 0.057194 | 0.2 |
| hybrid_failure_predictor_plus_learned_gate | tgsim_i90 | all | 100 | 10.212168 | 10.327657 | 0.011183 | 6 | 0.044 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | tgsim_i90 | hard | 100 | 6.091389 | 6.388728 | 0.046541 | 1 | 0.044 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | tgsim_i90 | baseline_failure | 100 | 11.620231 | 11.775393 | 0.013177 | 5 | 0.044 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | tgsim_i90 | verified_t50 | 100 | 10.212168 | 10.327657 | 0.011183 | 6 | 0.044 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | tgsim_i90 | verified_t100 | 100 | 10.212168 | 10.327657 | 0.011183 | 6 | 0.044 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| hybrid_failure_predictor_plus_learned_gate | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| no_interaction_ablation | eth_ucy | all | 10 | 0.67609 | 0.713643 | 0.052622 | 6 | 0.149585 | 0.5 |
| no_interaction_ablation | eth_ucy | easy | 10 | 0.01147 | 0.0 | -0.114701 | 1 | 0.095673 | 0.5 |
| no_interaction_ablation | eth_ucy | hard | 10 | 1.634765 | 1.746128 | 0.063777 | 2 | 0.145939 | 0.5 |
| no_interaction_ablation | eth_ucy | baseline_failure | 10 | 1.634765 | 1.746128 | 0.063777 | 2 | 0.145939 | 0.5 |
| no_interaction_ablation | tgsim | all | 100 | 6.088911 | 6.062032 | -0.004434 | 4 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim | hard | 100 | 12.03135 | 11.995052 | -0.003026 | 2 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim | baseline_failure | 100 | 21.038969 | 20.967461 | -0.00341 | 1 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim | verified_t50 | 100 | 6.088911 | 6.062032 | -0.004434 | 4 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim | verified_t100 | 100 | 6.088911 | 6.062032 | -0.004434 | 4 | 0.016 | 0.0 |
| no_interaction_ablation | tgsim_i90 | all | 100 | 8.607503 | 10.327657 | 0.166558 | 6 | 0.414962 | 0.8 |
| no_interaction_ablation | tgsim_i90 | hard | 100 | 4.098118 | 6.388728 | 0.358539 | 1 | 0.469517 | 0.8 |
| no_interaction_ablation | tgsim_i90 | baseline_failure | 100 | 10.088638 | 11.775393 | 0.143244 | 5 | 0.428319 | 0.8 |
| no_interaction_ablation | tgsim_i90 | verified_t50 | 100 | 8.607503 | 10.327657 | 0.166558 | 6 | 0.414962 | 0.8 |
| no_interaction_ablation | tgsim_i90 | verified_t100 | 100 | 8.607503 | 10.327657 | 0.166558 | 6 | 0.414962 | 0.8 |
| no_interaction_ablation | trajnet | all | 10 | 1.451655 | 1.434586 | -0.011899 | 7 | 0.082165 | 0.071429 |
| no_interaction_ablation | trajnet | easy | 10 | 0.335025 | 0.18701 | -0.791484 | 2 | 0.081878 | 0.0 |
| no_interaction_ablation | trajnet | hard | 10 | 2.351784 | 2.399685 | 0.019962 | 4 | 0.085581 | 0.125 |
| no_interaction_ablation | trajnet | baseline_failure | 10 | 2.351784 | 2.399685 | 0.019962 | 4 | 0.085581 | 0.125 |
| scalar_interaction_ablation | eth_ucy | all | 10 | 0.610945 | 0.713643 | 0.143907 | 6 | 0.127172 | 0.5 |
| scalar_interaction_ablation | eth_ucy | easy | 10 | 0.039443 | 0.0 | -0.394431 | 1 | 0.074364 | 0.5 |
| scalar_interaction_ablation | eth_ucy | hard | 10 | 1.493165 | 1.746128 | 0.144871 | 2 | 0.122912 | 0.5 |
| scalar_interaction_ablation | eth_ucy | baseline_failure | 10 | 1.493165 | 1.746128 | 0.144871 | 2 | 0.122912 | 0.5 |
| scalar_interaction_ablation | tgsim | all | 100 | 6.122798 | 6.062032 | -0.010024 | 4 | 0.033592 | 0.2 |
| scalar_interaction_ablation | tgsim | hard | 100 | 12.044144 | 11.995052 | -0.004093 | 2 | 0.040353 | 0.2 |
| scalar_interaction_ablation | tgsim | baseline_failure | 100 | 21.098898 | 20.967461 | -0.006269 | 1 | 0.056448 | 0.2 |
| scalar_interaction_ablation | tgsim | verified_t50 | 100 | 6.122798 | 6.062032 | -0.010024 | 4 | 0.033592 | 0.2 |
| scalar_interaction_ablation | tgsim | verified_t100 | 100 | 6.122798 | 6.062032 | -0.010024 | 4 | 0.033592 | 0.2 |
| scalar_interaction_ablation | tgsim_i90 | all | 100 | 8.637013 | 10.327657 | 0.163701 | 6 | 0.291334 | 0.533333 |
| scalar_interaction_ablation | tgsim_i90 | hard | 100 | 4.291121 | 6.388728 | 0.328329 | 1 | 0.328218 | 0.6 |
| scalar_interaction_ablation | tgsim_i90 | baseline_failure | 100 | 10.107469 | 11.775393 | 0.141645 | 5 | 0.298276 | 0.56 |
| scalar_interaction_ablation | tgsim_i90 | verified_t50 | 100 | 8.637013 | 10.327657 | 0.163701 | 6 | 0.291334 | 0.533333 |
| scalar_interaction_ablation | tgsim_i90 | verified_t100 | 100 | 8.637013 | 10.327657 | 0.163701 | 6 | 0.291334 | 0.533333 |
| scalar_interaction_ablation | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 |
| scalar_interaction_ablation | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 |
| scalar_interaction_ablation | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| scalar_interaction_ablation | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| graph_interaction_ablation | eth_ucy | all | 10 | 0.688839 | 0.713643 | 0.034757 | 6 | 0.024863 | 0.083333 |
| graph_interaction_ablation | eth_ucy | easy | 10 | 0.009412 | 0.0 | -0.094123 | 1 | 0.013447 | 0.0 |
| graph_interaction_ablation | eth_ucy | hard | 10 | 1.687786 | 1.746128 | 0.033412 | 2 | 0.022297 | 0.0 |
| graph_interaction_ablation | eth_ucy | baseline_failure | 10 | 1.687786 | 1.746128 | 0.033412 | 2 | 0.022297 | 0.0 |
| graph_interaction_ablation | tgsim | all | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.056407 | 0.2 |
| graph_interaction_ablation | tgsim | hard | 100 | 12.052357 | 11.995052 | -0.004777 | 2 | 0.070449 | 0.2 |
| graph_interaction_ablation | tgsim | baseline_failure | 100 | 21.068338 | 20.967461 | -0.004811 | 1 | 0.10712 | 0.2 |
| graph_interaction_ablation | tgsim | verified_t50 | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.056407 | 0.2 |
| graph_interaction_ablation | tgsim | verified_t100 | 100 | 6.123932 | 6.062032 | -0.010211 | 4 | 0.056407 | 0.2 |
| graph_interaction_ablation | tgsim_i90 | all | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.054 | 0.0 |
| graph_interaction_ablation | tgsim_i90 | hard | 100 | 6.097229 | 6.388728 | 0.045627 | 1 | 0.054 | 0.0 |
| graph_interaction_ablation | tgsim_i90 | baseline_failure | 100 | 11.614855 | 11.775393 | 0.013633 | 5 | 0.054 | 0.0 |
| graph_interaction_ablation | tgsim_i90 | verified_t50 | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.054 | 0.0 |
| graph_interaction_ablation | tgsim_i90 | verified_t100 | 100 | 10.207618 | 10.327657 | 0.011623 | 6 | 0.054 | 0.0 |
| graph_interaction_ablation | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 | 0.0 |
| graph_interaction_ablation | trajnet | easy | 10 | 0.18701 | 0.18701 | 0.0 | 2 | 0.0 | 0.0 |
| graph_interaction_ablation | trajnet | hard | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |
| graph_interaction_ablation | trajnet | baseline_failure | 10 | 2.399685 | 2.399685 | 0.0 | 4 | 0.0 | 0.0 |

## Interaction Ablation

| model | mean_hard_improvement | mean_failure_improvement |
| --- | --- | --- |
| no_interaction_ablation | 0.109813 | 0.055893 |
| scalar_interaction_ablation | 0.117277 | 0.070062 |
| graph_interaction_ablation | 0.018565 | 0.010558 |

## Direct Answers

是否补上 pedestrian/drone verified t+50/t+100：否.
HardBench-v1 是否可靠：是；hard episodes=53, eligibility=official.
BaselineFailureBench 是否建立：是；failure samples=48.
failure predictor 是否有效：是；test AUROC=0.899098.
alpha 是否学会何时介入：是.
failure-aware model 是否在 baseline failure cases 上赢：否.
easy cases 是否没有被破坏：是.
interaction encoder 是否有效：否.
verified long-horizon 是否改善：否.
是否可以进入 latent generative Stage 5C：否.
是否可以启用 SMC：否.
当前是否仍只是 trajectory forecasting scaffold：是.
当前是否更接近 world model：部分，更接近 failure-aware benchmarked world-state model，但不是 true world model.

## Final Verdict

项目是否跑通：是
pedestrian/drone long-horizon 是否补上：否
HardBench-v1 是否可靠：是
BaselineFailureBench 是否可靠：是
failure predictor 是否有效：是
failure-aware gated residual 是否有效：部分
interaction encoder 是否有效：否
verified long-horizon 是否改善：否
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage6_failure_bench_built_but_not_stage5c_ready
expert audit score：70
如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone verified t+50/t+100；真正多智能体 interaction episodes；failure-aware model 在 BaselineFailureBench 上稳定超过 baseline 10%。
