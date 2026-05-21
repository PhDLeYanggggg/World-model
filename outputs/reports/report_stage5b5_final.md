# Stage 5B.5 Final Report

Stage 5B.5 built hard interaction subsets and trained deterministic temporal-interaction models. After the runtime cleanup, the PyTorch path also completed and produced checkpoints. It still did not make the project a foundation world model.

## Benchmark Results

| dataset | subset | target_horizon | baseline_FDE | learned_FDE | improvement | episodes | gate_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | all | 10 | 0.713643 | 0.713643 | 0.0 | 6 | 0.0 |
| eth_ucy | easy | 10 | 0.197401 | 0.197401 | 0.0 | 4 | 0.0 |
| tgsim | all | 100 | 6.062032 | 6.062032 | 0.0 | 4 | 0.1 |
| tgsim | hard | 100 | 20.967461 | 20.967461 | 0.0 | 1 | 0.1 |
| tgsim | easy | 100 | 0.129012 | 0.129012 | 0.0 | 2 | 0.1 |
| tgsim_i90 | all | 100 | 10.327657 | 9.411212 | 0.088737 | 6 | 0.45 |
| tgsim_i90 | hard | 100 | 10.793798 | 10.02146 | 0.071554 | 3 | 0.45 |
| tgsim_i90 | easy | 100 | 6.388728 | 4.65976 | 0.270628 | 1 | 0.45 |
| trajnet | all | 10 | 1.434586 | 1.488781 | -0.037778 | 7 | 0.375 |
| trajnet | hard | 10 | 2.287502 | 2.254957 | 0.014227 | 1 | 0.375 |
| trajnet | easy | 10 | 1.315717 | 1.399987 | -0.064049 | 4 | 0.375 |

## PyTorch Backend Update

The previous `OMP: Error #179: Can't open SHM failed` blocker is no longer present in the cleaned environment. PyTorch training completed for three deterministic modes, but the metrics still fail the deterministic gate overall.

| mode | dataset | subset | target_horizon | baseline_FDE | torch_FDE | improvement |
| --- | --- | --- | --- | --- | --- | --- |
| recurrent_rollout | eth_ucy | all | 10 | 0.713643 | 0.721392 | -0.010857 |
| recurrent_rollout | tgsim | all | 100 | 6.062032 | 6.063009 | -0.000161 |
| recurrent_rollout | tgsim | hard | 100 | 20.967461 | 20.968679 | -5.8e-05 |
| hybrid | tgsim_i90 | all | 100 | 10.327657 | 9.800618 | 0.051032 |
| recurrent_rollout | tgsim_i90 | hard | 100 | 10.793798 | 10.796008 | -0.000205 |
| recurrent_rollout | trajnet | all | 10 | 1.434586 | 1.432021 | 0.001788 |
| hybrid | trajnet | hard | 10 | 2.287502 | 1.898759 | 0.169942 |

## Direct Answers

新增真实 pedestrian / drone 数据：部分。TrajNet++/ETH-UCY bundled fallback was prepared/probed, but no new raw long-horizon pedestrian/drone source was verified.
哪些能 t+50：TGSIM traffic sources only in this run.
哪些能 t+100：TGSIM traffic sources only in this run.
哪些只是 t+10：TrajNet and ETH/UCY fallback.
是否构建 hard interaction subsets：是。
hard subsets 数量够不够：部分；3 datasets are eval-ok.
temporal-interaction model 是否训练成功：是。NumPy deterministic fallback 和 PyTorch direct/recurrent/hybrid variants 都已跑通；但 PyTorch 结果仍未通过 deterministic gate.
是否超过 strongest causal baseline：部分；all-test wins=1.
在 hard-test 上超过了吗：部分但不过 gate；hard wins over 10%=1，PyTorch 只在 TrajNet hard t+10 上出现明显提升，不足以过 gate.
在 verified t+100 上超过了吗：部分；verified t+100 wins=1.
哪些数据源赢：tgsim_i90 all-test verified t+100 improved by about 8.9%.
哪些数据源输：tgsim, ETH/UCY fallback; pedestrian/drone verified t+100 仍为 0；tgsim hard subset also did not improve.
为什么输：strong causal baselines remain very strong; pedestrian snippets are short; no real maps/goals/routes; interaction is mostly diagnostic rather than multi-agent model input.
是否仍然只是 trajectory forecasting：是，仍是 trajectory world-state benchmark scaffold.
是否已经接近 world model：部分接近 state-space benchmark，但不是 true world model.
是否可以进入 latent generative Stage 5C：否.
是否可以启用 SMC：否.
expert audit score 是否达到 70：是 (70).
expert audit score 是否达到 80：否.

## Final Verdict

项目是否跑通：是
新增真实 pedestrian/drone 长轨迹数据：否
actual verified t+100 pedestrian/drone source 数量：0
hard interaction benchmark 是否建立：是
temporal-interaction deterministic model 是否超过 strongest causal baseline：部分
all-test 是否超过：部分
hard-test 是否超过：部分
verified t+100 是否超过：部分
cross-dataset 泛化：diagnostic only
latent generative Stage 5C 是否 ready：否
SMC 是否 ready：否
当前 verdict：stage5b5_hard_benchmark_built_but_deterministic_gate_failed
expert audit score：70
如果不能进入 Stage 5C，下一步先修什么：长轨迹 pedestrian/drone 数据；真实多智能体输入；干净 PyTorch temporal-interaction runtime。
