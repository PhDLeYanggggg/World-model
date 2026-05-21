# Stage 5B.6 Final Report

Stage 5B.6 repaired the benchmark reliability logic and trained baseline-aware gated residual models. It did not make the system ready for Stage 5C.

## Current State

1. 当前不是 true 3D world model.
2. 当前不是 large-scale foundation world model.
3. 当前仍是 multi-source trajectory world-state benchmark scaffold.
4. Stage 5B.5 hard subsets 已建立，Stage 5B.6 进一步证明其统计可靠性不足.
5. PyTorch deterministic temporal-interaction models 已经跑通，但 deterministic gate 仍失败.
6. actual verified t+100 pedestrian/drone source 数量仍为 0.
7. latent generative Stage 5C 和 SMC 仍不 ready.

## Official Gated Residual Results

| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_residual_all_data | eth_ucy | all | 10 | 0.63948 | 0.713643 | 0.103923 | 6 | 0.04003 |
| gated_residual_all_data | eth_ucy | easy | 10 | 0.170413 | 0.197401 | 0.136718 | 4 | 0.043119 |
| gated_residual_all_data | tgsim | all | 100 | 6.531357 | 6.062032 | -0.07742 | 4 | 0.155318 |
| gated_residual_all_data | tgsim | hard | 100 | 22.674988 | 20.967461 | -0.081437 | 1 | 0.500192 |
| gated_residual_all_data | tgsim | easy | 100 | 0.208527 | 0.129012 | -0.616337 | 2 | 0.044317 |
| gated_residual_all_data | tgsim_i90 | all | 100 | 10.252016 | 10.327657 | 0.007324 | 6 | 0.04 |
| gated_residual_all_data | tgsim_i90 | hard | 100 | 10.778029 | 10.793798 | 0.001461 | 3 | 0.04 |
| gated_residual_all_data | tgsim_i90 | easy | 100 | 6.2075 | 6.388728 | 0.028367 | 1 | 0.04 |
| gated_residual_all_data | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 |
| gated_residual_all_data | trajnet | hard | 10 | 2.287502 | 2.287502 | 0.0 | 1 | 0.0 |
| gated_residual_all_data | trajnet | easy | 10 | 1.315717 | 1.315717 | 0.0 | 4 | 0.0 |
| gated_residual_failure_classifier_aux | eth_ucy | all | 10 | 0.663455 | 0.713643 | 0.070327 | 6 | 0.02721 |
| gated_residual_failure_classifier_aux | eth_ucy | easy | 10 | 0.180313 | 0.197401 | 0.086568 | 4 | 0.028141 |
| gated_residual_failure_classifier_aux | tgsim | all | 100 | 6.306686 | 6.062032 | -0.040358 | 4 | 0.149895 |
| gated_residual_failure_classifier_aux | tgsim | hard | 100 | 21.814943 | 20.967461 | -0.040419 | 1 | 0.450191 |
| gated_residual_failure_classifier_aux | tgsim | easy | 100 | 0.191341 | 0.129012 | -0.48313 | 2 | 0.057431 |
| gated_residual_failure_classifier_aux | tgsim_i90 | all | 100 | 10.261156 | 10.327657 | 0.006439 | 6 | 0.03 |
| gated_residual_failure_classifier_aux | tgsim_i90 | hard | 100 | 10.801836 | 10.793798 | -0.000745 | 3 | 0.03 |
| gated_residual_failure_classifier_aux | tgsim_i90 | easy | 100 | 6.197868 | 6.388728 | 0.029874 | 1 | 0.03 |
| gated_residual_failure_classifier_aux | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 |
| gated_residual_failure_classifier_aux | trajnet | hard | 10 | 2.287502 | 2.287502 | 0.0 | 1 | 0.0 |
| gated_residual_failure_classifier_aux | trajnet | easy | 10 | 1.315717 | 1.315717 | 0.0 | 4 | 0.0 |
| gated_residual_hard_weighted | eth_ucy | all | 10 | 0.673919 | 0.713643 | 0.055664 | 6 | 0.022551 |
| gated_residual_hard_weighted | eth_ucy | easy | 10 | 0.182403 | 0.197401 | 0.075978 | 4 | 0.024237 |
| gated_residual_hard_weighted | tgsim | all | 100 | 6.310842 | 6.062032 | -0.041044 | 4 | 0.152487 |
| gated_residual_hard_weighted | tgsim | hard | 100 | 21.813395 | 20.967461 | -0.040345 | 1 | 0.450189 |
| gated_residual_hard_weighted | tgsim | easy | 100 | 0.200773 | 0.129012 | -0.556236 | 2 | 0.063866 |
| gated_residual_hard_weighted | tgsim_i90 | all | 100 | 10.252297 | 10.327657 | 0.007297 | 6 | 0.04 |
| gated_residual_hard_weighted | tgsim_i90 | hard | 100 | 10.782509 | 10.793798 | 0.001046 | 3 | 0.04 |
| gated_residual_hard_weighted | tgsim_i90 | easy | 100 | 6.199121 | 6.388728 | 0.029678 | 1 | 0.04 |
| gated_residual_hard_weighted | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 |
| gated_residual_hard_weighted | trajnet | hard | 10 | 2.287502 | 2.287502 | 0.0 | 1 | 0.0 |
| gated_residual_hard_weighted | trajnet | easy | 10 | 1.315717 | 1.315717 | 0.0 | 4 | 0.0 |

## Best Official Model By Dataset

| model | dataset | subset | target | FDE | baseline_FDE | improvement | episodes | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_residual_all_data | eth_ucy | all | 10 | 0.63948 | 0.713643 | 0.103923 | 6 | 0.04003 |
| gated_residual_failure_classifier_aux | tgsim | all | 100 | 6.306686 | 6.062032 | -0.040358 | 4 | 0.149895 |
| gated_residual_all_data | tgsim_i90 | all | 100 | 10.252016 | 10.327657 | 0.007324 | 6 | 0.04 |
| gated_residual_all_data | trajnet | all | 10 | 1.434586 | 1.434586 | 0.0 | 7 | 0.0 |

## Interaction Ablation

| ablation | mean_all_target_improvement | mean_hard_target_improvement | note |
| --- | --- | --- | --- |
| graph attention interaction | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| graph attention + temporal neighbor history | 0.00735 | -0.013418 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| nearest-neighbor scalar features only | 0.068819 | 0.054096 | quick deterministic ablation; interaction is past-only from kNN world-state table |
| no interaction | 0.037748 | 0.04006 | quick deterministic ablation; interaction is past-only from kNN world-state table |

## Direct Answers

hard benchmark 是否可靠：否；official hard-gate eligible datasets=0.
hard subsets 样本量是否足够：否；所有 hard subsets 都低于 30 或 50 的强 gate 阈值.
新增 pedestrian/drone 长时程数据了吗：否.
是否有 pedestrian/drone verified t+50：否.
是否有 pedestrian/drone verified t+100：否.
gated residual model 是否训练成功：是，7 个 official/ablation checkpoint 已生成.
alpha gate 是否学会什么时候介入：部分；easy_alpha=0.029783, hard_alpha=0.091664, corr=0.207346.
interaction encoder 是否真的带来提升：否；graph hard improvement=-0.013418, no-interaction=0.04006.
all-test 是否超过 strongest causal baseline：部分；official gated residual target wins=1.
hard-test 是否超过 strongest causal baseline：否，no official hard subset is reliable enough for gate.
verified t+100 是否超过 strongest causal baseline：否，official gated variants did not beat verified t+100 by 5%.
是否可以进入 Stage 5C latent generative：否.
是否可以启用 SMC：否.
当前 expert audit score：68.

## Final Verdict

项目是否跑通：是
hard benchmark 是否可靠：否
真实 pedestrian/drone long horizon 是否补上：否
gated residual 是否超过 strongest causal baseline：部分
interaction encoder 是否有效：否
alpha gate 是否有效：部分
verified long-horizon 是否改善：否
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage5b6_reliability_repaired_but_deterministic_gate_failed
expert audit score：68
如果不能进入 Stage 5C，下一步先修什么：真实 pedestrian/drone t+50/t+100；>=50 hard episodes per dataset；真正多智能体 interaction episode 输入。
