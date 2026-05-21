# Stage 5B Final Report

Stage 5B moved the project from a registry-heavy data lake scaffold to an actually executable multi-source benchmark. It still does not produce a large-scale foundation world model.

## Honest Current State

1. The project is not a true 3D world model.
2. It is not a large-scale foundation world model.
3. It is a partial but usable real-trajectory data lake plus deterministic residual pretraining gate.
4. Latent generative modeling remains disabled.
5. SMC remains disabled because deterministic learned proposals are not strong enough.

## Actual Converted Sources

| dataset | domain | actual_verified_t100 | horizons | episodes | metric |
| --- | --- | --- | --- | --- | --- |
| eth_ucy | pedestrian | False | [1, 10] | 15/2/6 | False |
| tgsim | traffic | True | [1, 10, 25, 50, 100] | 23/5/4 | True |
| tgsim_i90 | traffic | True | [1, 10, 25, 50, 100] | 18/7/6 | True |
| trajnet | pedestrian | False | [1, 10] | 19/6/7 | True |

## Baseline vs Learned

| dataset | domain | actual_verified_t100 | official_horizons | target_horizon | strongest_causal_baseline | baseline_FDE_target | best_learned | learned_FDE_target | learned_improvement | learned_beats_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | pedestrian | False | [1, 10] | 10 | constant_velocity_causal_fd | 0.713643 | deterministic_residual_multistep | 0.575894 | 0.193022 | True |
| tgsim | traffic | True | [1, 10, 25, 50, 100] | 100 | constant_velocity_causal_fd | 6.062032 | deterministic_residual_one_step | 6.060675 | 0.000224 | False |
| tgsim_i90 | traffic | True | [1, 10, 25, 50, 100] | 100 | constant_velocity_causal_fd | 10.327657 | deterministic_residual_one_step | 10.329037 | -0.000134 | False |
| trajnet | pedestrian | False | [1, 10] | 10 | constant_velocity_causal_fd | 1.434586 | deterministic_residual_one_step | 1.439945 | -0.003736 | False |

## Direct Answers

实际下载或接入真实数据源：4
实际转换真实数据源：4
actual verified t+100 数据源：2
registry-estimated 数据不计入 actual verified t+100。
失败/placeholder：SDD、OpenDD、NGSIM 在本轮保持 license/manual placeholder，未作为 converted dataset。
deterministic learned model 已训练：是，quick linear residual one-step 和 multistep 两个版本。
deterministic learned model 超过 strongest causal baseline 的数据源数量：1
cross-dataset generalization：已执行诊断矩阵，但未完成真正 leave-one-dataset-out learned transfer。
no-leakage audit：pass
现在是否可以称为 large-scale world model：否。
现在是否仍只是 trajectory forecasting model：是，更准确说是 multi-source trajectory world-state benchmark scaffold。
是否可以进入 latent generative Stage 5C：否。
是否可以启用 SMC：否。
当前 expert audit score：68 / 100
当前 verdict：stage5b_usable_data_lake_but_deterministic_gate_failed

## Final Verdict

项目是否跑通：是
数据湖是否从 partial 变成 usable：部分
实际转换真实数据源数量：4
actual verified t+100 数据源数量：2
no-leakage audit：pass
strongest causal baselines：见上表逐数据源结果
best learned deterministic model：deterministic_residual_multistep on ETH/UCY fallback for t+10, but not enough for Stage 5C
learned model 是否超过 strongest causal baseline：部分
跨数据集泛化：弱 / diagnostic only
是否启用 latent generative：否
是否启用 SMC：否
当前 verdict：stage5b_usable_data_lake_but_deterministic_gate_failed
expert audit score：68
是否达到 70：否
是否达到 80：否
是否进入 Stage 5C latent generative：否
如果否，下一步先修什么：补长轨迹真实行人/无人机数据；训练真正 deterministic temporal-interaction model；加入真实 scene/map/goal geometry。
