# M3W 长期目标研究总账：路线、失败、成功与当前结论

更新时间：2026-05-26  
汇总状态：`cached_verified` 汇总 Stage18-Stage42 已生成报告；Stage42-A 到 Stage42-S 中明确标注 `fresh_run` 的实验按原报告记录。  
用途：这是给用户看的一个总 README，把“这个长期目标内我做了什么、尝试了哪些路线、哪些失败、为什么失败、哪些成功、当前最强可部署是谁、还不能声称什么”集中写清楚。

这不是宣传稿，也不是论文 claim。它是研究状态账本。

## 0. 先写清楚不能夸大的边界

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame / pixel-space 的 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external top-down pedestrian 数据仍是 dataset-local / unverified weak-metric diagnostic，不能写成统一真实物理米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有完成全局验证。
- self-audited / visual-prior / auto-silver labels 不是 human gold。
- JEPA 在本项目里是 representation / auxiliary pretraining，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 仍不安全；当前可部署路线需要 Stage37 / teacher safety floor。

一句话现状：

```text
current best deployable =
  M3W-Neural v1 composite-tail safe-switch bounded neural dynamics

safety floor =
  Stage37 selector / teacher floor

honest claim =
  protected 2.5D external world-state candidate

not claim =
  true 3D / metric / seconds-level / foundation world model
```

## 1. 总体路线：从 SDD selector 到 protected neural world-state candidate

这个目标里实际推进的是一条长期研究链：

1. 先验证早期 JEPA / selector / correction 在 SDD 上为什么没有 downstream lift。
2. 把 SDD 建成 pixel raw-frame official benchmark。
3. 训练出 Stage26 cost-aware selector，成为 SDD best deployable。
4. 尝试把 SDD 结果迁移到外部 top-down pedestrian 数据，zero-shot 先大失败。
5. 补 external row geometry、train-only goals、history windows、goal prototypes 和 selective transfer。
6. Stage37 第一次把 external all / t+50 / hard-failure / easy safety 同时修到可部署。
7. Stage38 尝试 bounded correction，但没有安全超过 Stage37，不部署。
8. Stage39/40 开始真正 neural dynamics，但 ungated neural 不安全，不能替代 Stage37。
9. Stage41/42 把 neural 做成受保护的世界状态候选：full-waypoint、sequence ablation、static-gated repair、gain/harm selector 等逐步推进。
10. 到 Stage42-P，t+50 gain/harm selector 均值修复成功，但 seed-level t50 稳定性还不足，不能包装成完整 A刊级 t+50 结论。
11. Stage42-Q 发现 Stage42-J static expert 与 Stage42-P gain/harm selector 互补，但只是 report-level preflight；Stage42-R 随后建立 row prediction cache，完成 validation-only combo eval，并把 cached combo t+50 CI low 修到正数。
12. Stage42-S 将 Stage42-R combo 冻结成轻量 policy artifact，记录 policy hash/cache hash/schema hash，并完成 per-domain/per-horizon stress audit。

核心规律：

```text
纯模型堆大没有解决问题。
真正有用的是：
  causal history
  train-only goals / goal prototypes
  hard/easy/failure 分层
  gain/harm/risk-aware selection
  conservative fallback
  full-waypoint dynamics under safety floor
  validation-only policy selection
  row-level prediction cache for combo evaluation
```

## 2. 当前最强可部署结果与 Stage42-R 最新分支证据

当前主候选来自 M3W-Neural v1 / Stage41-42 证据链。相对 Stage37 / teacher safety floor，主包报告如下：

| 指标 | 数值 |
| --- | ---: |
| gates | 41 / 41 |
| evaluated rows | 55,528 |
| all ADE improvement | 0.2103 |
| t+50 ADE improvement | 0.1365 |
| t+100 raw-frame diagnostic ADE improvement | 0.1469 |
| hard/failure ADE improvement | 0.2038 |
| easy degradation | 0.0000 |
| positive external domains | 3 |
| all-agent composite FDE improvement | 0.1982 |
| all-agent composite FDE@50 improvement | 0.1739 |

Bootstrap 支持：

| slice | 2000-bootstrap low | mid | high |
| --- | ---: | ---: | ---: |
| all | 0.2067 | 0.2102 | 0.2139 |
| t+50 | 0.1306 | 0.1366 | 0.1426 |
| t+100 raw-frame diagnostic | 0.1396 | 0.1469 | 0.1537 |
| hard/failure | 0.1999 | 0.2039 | 0.2076 |

多 seed 支持：

```text
composite_tail_multiseed_pass = true
all_mean = 0.2095
t50_mean = 0.1383
t100_raw_frame_mean = 0.1445
hard_mean = 0.2031
easy_max = 0.0
positive_domain_counts = [3, 3, 3]
```

严格 pure-UCY neural retrain / select / test 证据：

```text
gate = true
best_trial = pure_ucy_transformer
best_mode = bounded_endpoint_residual
all = 0.0901
t50 = 0.0880
t100 raw-frame diagnostic = 0.0831
hard/failure = 0.0936
easy = 0.0
bootstrap lows = all 0.0889 / t50 0.0863 / t100 0.0807 / hard 0.0923
```

Endpoint-to-full trajectory bridge 证据：

```text
two_domain_endpoint_to_full_gate = true
positive_domains = ETH_UCY, TrajNet
ETH_UCY bootstrap lows: ADE all 0.0150, ADE t50 0.0014, FDE all 0.0154, FDE t50 0.0020
TrajNet bootstrap lows: ADE all 0.0338, ADE t50 0.0186, FDE all 0.0339, FDE t50 0.0258
```

注意：Stage37 用百分比风格汇报 improvement；Stage42 多数报告使用 ADE/FDE improvement 数值。不要把两个量纲直接混成同一个单位。

Stage42-R 最新 full-waypoint combo 分支结果：

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
cached_combo_ADE_all = 0.052387
cached_combo_ADE_t50 = 0.037934
cached_combo_ADE_t50_CI_low = 0.027740
cached_combo_ADE_t100_raw_frame_diagnostic = 0.041846
cached_combo_ADE_hard_failure = 0.054792
cached_combo_easy_degradation = 0.001102
cached_combo_FDE_t50 = 0.100059
source_choices = Stage42-P t50 gain/harm 14, floor 4, Stage42-J static expert 6
cache_dir = data/stage42_row_prediction_cache (not committed)
```

解释：

- Stage42-R 不是一个新 metric/seconds-level 结果，也不是 Stage5C/SMC。
- 它把 Stage42-Q 的“报告级互补信号”变成了 row-cache-backed evaluation。
- 它说明 Stage42-J 的 full-waypoint static expert 和 Stage42-P 的 t+50 gain/harm selector 可以通过 validation-only row-level combo 同时保留 t+50、hard/failure 和 easy safety。
- 它不改变全局边界：当前仍是 dataset-local raw-frame 2.5D protected world-state candidate。

## 3. 做过的主要路线、结果和原因

### 3.1 Stage18-19：JEPA / WAM-style representation

尝试：

- SAM-JEPA-2.5D representation pretraining。
- WAM-style data registry：simulation、real top-down trajectories、human/egocentric video。
- JEPA non-collapse 检查。
- 用 JEPA latent 辅助 selector、failure predictor、goal predictor、correction、official t+50。

结果：

```text
JEPA non-collapse = yes
downstream lift = no
deployable contribution = no
```

失败原因：

- JEPA latent 有 variance，但没有对齐部署目标。
- 真正卡住的是 causal history、hard/easy/failure、gain/harm、fallback safety，而不是单纯 representation variance。
- 当时没有足够强的外部 top-down scene/trajectory 对齐，也没有可靠的 hard/failure 部署标签。
- JEPA 没有启用 latent rollout，也不能说成生成式 world model。

结论：

```text
JEPA 不能作为主 claim，除非证明 downstream lift。
后续 JEPA 只能作为 auxiliary representation，而不是部署核心。
```

### 3.2 Stage22-26：SDD pixel-space benchmark 与 Stage26 selector

尝试：

- 把 SDD 转成 pixel raw-frame official benchmark。
- 构建 SDD scene packs、lazy medium index、no-leakage audit。
- 计算 strongest causal baseline。
- 建立 HardBench / BaselineFailureBench / GoalBench。
- 从 hard-class selector 改为 expected-FDE / cost-aware selector。

关键成功：

```text
Stage26 selector t50 improvement ≈ 14.58%
hard/failure improvement ≈ 11.23%
easy degradation ≈ 1.81%
Stage26 = 当时 SDD best deployable
```

失败过的路线：

```text
Stage24 selector oracle headroom ≈ 46.2%
trained hard-class selector t+50 improvement = -43.3%
easy degradation = 11.33%
```

失败原因：

- 直接分类“哪个 baseline 最好”太粗糙。
- 很多样本 oracle best 与 second best margin 很小，one-hot label 噪声大。
- selector 没有显式考虑 harm over fallback，导致 easy case 被大量错误切换。
- oracle headroom 大不等于真实 selector 可学；如果学习目标错，会越学越伤。

成功原因：

- expected-FDE / gain-harm / regret-aware 目标更接近部署需求。
- conservative fallback 到 strongest causal baseline 是必要安全机制。
- 这奠定了后面 Stage37/Stage42 的核心思想：先判断能不能安全切换，再考虑预测更复杂的轨迹。

### 3.3 Stage31-36：从 SDD 迁移到外部 top-down pedestrian 数据

尝试：

- OpenTraj / ETH-UCY / TrajNet / UCY external feature store。
- external strongest baseline。
- external latent cache。
- zero-shot transfer。
- domain normalization、relative targets、CORAL / latent adapter、coordinate-invariant features。
- external row geometry、train-only goals、scene packs、horizon split。
- selective transfer policy。

初始失败：

```text
Stage31 zero-shot external all improvement ≈ -92.67%
Stage31 zero-shot t50 ≈ -278.57%
```

失败原因：

- SDD pixel coordinates 与 external dataset-local coordinates 不兼容。
- scene/goal/interaction 上下文缺失。
- agent type 约定不一致。
- homography / scale 未验证。
- SDD selector 是 SDD-specific，不能直接当跨域模型。

Stage35 局部成功但不可部署：

```text
all improvement = 12.13%
hard/failure improvement = 13.98%
easy degradation = 0.041%
t50 improvement = 0.0
verdict = not deployable
```

Stage36 诊断：

- t+50 rows 足够。
- t+50 oracle headroom 约 22.98%。
- t+50 不是没有可学空间，而是特征/goal/context 不足以支持安全切换。
- 只调 threshold 不够，因为模型没有看见足够的 past-only history 和 scene-agnostic goal pattern。

结论：

```text
跨域失败的根因主要不是模型容量，而是坐标/目标/历史/安全切换建模不足。
```

### 3.4 Stage37：外部 t+50 修复成功

核心改变：

- 构建 past-only history windows。
- 构建 scene-agnostic goal prototypes。
- 训练 t+50 专用 failure/gain/harm/switchability 模型。
- 用 conformal / conservative safety 控制 easy degradation。

关键结果：

```text
all improvement = +13.48%
t50 improvement = +8.46%
t50 bootstrap CI = [+7.69%, +9.15%]
hard/failure improvement = +15.54%
easy degradation = 0.041%
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

为什么成功：

- t+50 不再被 all-horizon objective 淹没。
- goals 不来自 test endpoints，而来自 train 相对运动模式和 scene-agnostic prototypes。
- 只在 gain 高、harm 低、confidence 高时切换，否则 fallback。

结论：

```text
Stage37 是第一个 external deployable selector candidate。
```

### 3.5 Stage38：bounded correction / dynamics head 没有部署

尝试：

- 冻结 Stage37 policy。
- 训练 bounded correction / dynamics head：

```text
prediction = selected_baseline + alpha * bounded_delta
```

结果：

```text
bounded correction = not deployable
current external best remained Stage37 selector
```

失败原因：

- correction 可以改善少量 hard/tail rows，但不能稳定超过 Stage37。
- easy preservation 很脆弱。
- correction_with_fallback 没有同时满足 all/t50/hard 超过 Stage37 和 easy <=2%。

结论：

```text
不要部署 Stage38 correction。
Stage37 继续作为 external best safety floor。
```

### 3.6 Stage39-40：真正神经网络世界动力学训练，初期失败

尝试：

- Causal Temporal Transformer。
- JEPA auxiliary。
- Hybrid JEPA + Transformer。
- neural without fallback。
- neural with Stage37 fallback。
- teacher distillation、horizon heads、hard/failure oversampling、t50 curriculum。

结果：

```text
Transformer / JEPA / Hybrid trained = yes
neural without fallback = unsafe
neural with fallback did not clearly beat Stage37
deploy neural = no
current best remained Stage37
```

失败原因：

- 无保护 neural 容易伤 easy cases。
- JEPA non-collapse 仍不等于 downstream lift。
- Hybrid 的 latent/dynamics 没有稳定转化为部署增益。
- 很多模型其实在学习“近似 Stage37”，而不是学习更强的可部署切换/动力学。

结论：

```text
神经网络方向不能直接替代 selector。
必须把 Stage37/teacher floor 当作安全地板，再训练 protected neural dynamics。
```

### 3.7 Stage41-42：protected neural world-state candidate

Stage41/42 的目标是把神经网络从“裸模型失败”改造成“受保护的 world-state candidate”。

主要成功：

- composite-tail safe-switch bounded neural dynamics 达到 41/41 gates。
- external source-level validation fresh run 通过。
- full-waypoint dynamics fresh run 通过。
- pure-UCY strict retrain 通过。
- endpoint-to-full bridge 在 ETH_UCY / TrajNet 有正证据。
- sequence history ablation 证明 causal history tokens 有强贡献。
- static/context 不是无用，而是必须 gated 使用。
- explicit gain/harm selector 明显优于 alpha-only distillation。

### 3.8 Stage42-A/B/C：数据审计、外部验证、full-waypoint

Stage42-A 数据/标定审计：

```text
source = fresh_run
datasets_audited = 7
raw_paths_found = 6
converted_paths_found = 7
external_domains_ready = OpenTraj, ETH/UCY, TrajNet++, UCY
metric_claim_ready_datasets = TGSIM diagnostic only
seconds_claim_ready_datasets = none
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
stage42_a_gates = 7 / 7
```

Stage42-B 外部 source-level 验证：

```text
source = fresh_run
source_level_split_rebuilt = true
frozen_eval_pool_rows = 66,303
evaluated_rows = 55,528
protected_M3W_all_ADE_improvement = 0.2103
protected_M3W_t50_ADE_improvement = 0.1365
protected_M3W_t100_raw_frame_diagnostic_ADE_improvement = 0.1469
protected_M3W_hard_failure_ADE_improvement = 0.2038
protected_M3W_easy_degradation = -0.1451
ungated_neural_all_ADE_improvement = 0.2966
ungated_neural_easy_degradation = 1.2459
stage42_b_gates = 10 / 10
verdict = protected neural passes; ungated neural unsafe
```

Stage42-C full-waypoint dynamics：

```text
source = fresh_run
full_waypoint_sequence_model = full_trajectory_ensemble
positive_full_waypoint_domains = ETH_UCY, TrajNet
protected_full_waypoint_ADE_all = 0.1858
protected_full_waypoint_ADE_t50 = 0.1480
protected_full_waypoint_ADE_t100_raw_frame_diagnostic = 0.2286
protected_full_waypoint_ADE_hard_failure = 0.1952
protected_full_waypoint_easy_degradation = 0.0000
protected_full_waypoint_FDE_all = 0.1938
protected_full_waypoint_FDE_t50 = 0.2158
stage42_c_gates = 12 / 12
```

意义：

- Stage42-C 把证据从 endpoint-only / linear bridge 推向 reconstructed future waypoints。
- 但它仍然依赖 safety floor，不是无保护生成式 rollout。

### 3.9 Stage42-D/E/F：论文包与证据边界

Stage42-D：

```text
causal ablation evidence = pass_with_retrain_boundary
gates = 12 / 12
```

边界：

- 有 component evidence，但不是每个组件都完成 full fresh retraining。

Stage42-E：

```text
safety floor research = pass
gates = 12 / 12
```

结论：

- safety floor 不是临时补丁，而是当前部署方法的一部分。
- 去掉 floor 的 neural 还不安全。

Stage42-F：

```text
paper package = complete_not_full_a_journal_ready
gates = 12 / 12
```

结论：

- 已经有 strong protected 2.5D manuscript evidence package。
- 但还不是 full A-journal-ready / foundation-ready 结果。

缺口：

- full retrained ablation 不完整。
- 更多独立外部数据不足。
- metric/time calibration 不足。
- floor-free neural safety mechanism 不足。

### 3.10 Stage42-G/H：重训消融与 causal sequence history

Stage42-G Phase1 fresh retrained ablation：

```text
source = fresh_run
verdict = stage42_g_retrained_ablation_phase1_pass
gates = 11 / 11
variants = full, no_history, no_neighbor, no_goal, no_scene_goal,
           no_interaction, no_domain_expert, no_transformer_proxy_history_sequence,
           no_safe_switch, no_teacher_floor_proxy
seeds_per_variant = 3
full_all = 0.8122
full_t50 = 0.8462
full_t100_raw_frame_diagnostic = 0.9527
full_hard_failure = 0.8459
full_easy_degradation = -0.8413
```

重要负结果：

- flattened history / transformer-proxy history 在 ridge selector 协议里不稳定。
- 不能因此说 history 无用，只能说这个表示方式太弱。

Stage42-H causal temporal sequence encoder：

```text
source = fresh_run
verdict = stage42_h_sequence_ablation_pass
gates = 10 / 10
sequence_full_all = 0.7785
sequence_full_t50 = 0.7834
sequence_full_t100_raw_frame_diagnostic = 0.9166
sequence_full_hard_failure = 0.8081
sequence_full_easy_degradation = -0.7684
history_t50_delta_full_minus_no_history = 0.4578
history_hard_delta_full_minus_no_history = 0.4708
```

结论：

```text
history tokens 有真实贡献，但需要 causal sequence encoder。
flattened history + ridge 不足以表达这个信号。
```

### 3.11 Stage42-I/J/K/L/M/N/O/P/Q/R：full-waypoint static/context 与 gain/harm 修复链

Stage42-I：

```text
verdict = stage42_i_sequence_full_waypoint_partial
sequence_waypoint_full ADE all/t50/hard = -0.0106 / -0.0321 / -0.0116
sequence_waypoint_no_static_context ADE all/t50/hard = 0.0115 / 0.0199 / 0.0129
FDE t50 = 0.0611
```

结论：

- causal sequence history 有用。
- 但 static/context 不能无条件混入；全局混入会伤 full-waypoint。

Stage42-J：

```text
source = cached_verified_checkpoints_fresh_static_gate_eval
verdict = stage42_j_static_gated_full_waypoint_pass
ADE all/t50/hard = 0.0362 / 0.0369 / 0.0397
t100 raw-frame diagnostic ADE = 0.0267
FDE all/t50 = 0.0633 / 0.1166
easy degradation = 0.0
```

结论：

- static/context 不是没用，而是必须按 domain/horizon validation-gated。
- Stage42-J 是强的 policy-level static-gated full-waypoint evidence。

Stage42-K：

```text
source = fresh_run
verdict = stage42_k_fresh_static_gated_checkpoint_pass
ADE all = 0.0136
ADE t100 raw-frame diagnostic = 0.0159
ADE hard/failure = 0.0148
FDE all = 0.0312
FDE t50 = 0.0358
ADE t50 = -0.0122
easy degradation = 0.0
```

结论：

- fresh checkpoint 方向可行，但 t+50 ADE 仍失败。

Stage42-L：

```text
source = fresh_run
verdict = stage42_l_horizon_static_gate_repair_pass
ADE all = 0.0219
ADE t50 = 0.0020
ADE hard/failure = 0.0240
FDE t50 = 0.0532
easy degradation = 0.0
```

结论：

- horizon-aware static gate 修复了 Stage42-K 的 t+50 ADE 负号。
- 但仍弱于 Stage42-J policy-level static gate。

Stage42-M：

```text
source = fresh_run
verdict = stage42_m_policy_distilled_static_gate_partial
ADE all = 0.0161
ADE t50 = -0.0015
ADE hard/failure = 0.0177
FDE t50 = 0.0729
easy degradation = 0.0
gates = 10 / 12
```

失败原因：

- slice-level alpha distillation 太粗。
- domain/horizon alpha 增加 static usage，但没有教会 row-level gain/harm。

Stage42-N：

```text
source = fresh_run
verdict = stage42_n_row_gain_static_gate_partial
ADE all = 0.0250
ADE t50 = -0.0278
ADE hard/failure = 0.0269
easy degradation = 0.0
```

失败原因：

- row-level static gain/floor gain/harm/switchability 有帮助，但仍不能选对 t+50 rows。
- alpha-style static-gate supervision 不足。

Stage42-O：

```text
source = fresh_run
verdict = stage42_o_explicit_gain_harm_selector_partial
ADE all = 0.052646
ADE t50 = -0.000776
ADE t100 raw-frame diagnostic = 0.060206
ADE hard/failure = 0.053527
easy degradation = 0.015491
FDE t50 = 0.057614
gates = 13 / 14
```

重要修复：

- Stage42-O 最初看起来 13/13 pass，但发现 split-local normalization 风险后，改成 train-only normalization，并增加 `no_test_statistics_normalization` gate。
- 严格协议后 t+50 ADE 仍微负，所以必须标 partial。

结论：

- explicit gain/harm prediction 明显比 alpha-only distillation 好。
- 但还需要 t+50-specific 训练/校准。

Stage42-P：

```text
source = fresh_run
verdict = stage42_p_t50_gain_harm_selector_pass
gates = 14 / 14
ADE all = 0.051537
ADE t50 = 0.006596
ADE t100 raw-frame diagnostic = 0.059254
ADE hard/failure = 0.053256
easy degradation = 0.008580
FDE all = 0.080118
FDE t50 = 0.057431
switch rate = 0.139443
```

重要边界：

```text
3-seed t50 CI low = -0.0179
```

结论：

- Stage42-P 修复了 Stage42-O 的 mean ADE t+50 负号。
- 它是 gate-passing t+50 repair。
- 但 t+50 seed-level 稳定性还不够，不能作为 paper-stable t+50 claim。
- 下一步应该把 Stage42-P row-level gain/harm selector 与 Stage42-J static expert policy 组合，并做更多 seed/bootstrap。

Stage42-Q：

```text
source = cached_verified_report_level_preflight
verdict = stage42_q_preflight_partial_row_cache_required
gates = 7 / 7
diagnostic_ADE_t50_best_available = 0.036875
diagnostic_FDE_t50_best_available = 0.116638
row_level_combo_status = attempted_not_completed
```

结论：

- Stage42-Q 只证明 Stage42-J static expert 和 Stage42-P gain/harm selector 有互补可能。
- 因为没有 row-level prediction cache，它不能当作 deployable combo。

Stage42-R：

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
cached_combo_ADE_all = 0.052387
cached_combo_ADE_t50 = 0.037934
cached_combo_ADE_t50_CI_low = 0.027740
cached_combo_ADE_hard_failure = 0.054792
cached_combo_easy_degradation = 0.001102
cached_combo_FDE_t50 = 0.100059
```

成功原因：

- 它不再只在 report-level 拼结果，而是把 floor / Stage42-J / Stage42-P 的 row-level prediction errors 缓存到本地 NPZ，再做 validation-only source selection。
- combo 只用 validation domain/horizon slice 选 source，test 只最终评估一次。
- 它同时用到了 Stage42-J 的 static expert 稳定性和 Stage42-P 的 t+50 gain/harm 信号。

边界：

- cache 是本地 derived array，不提交 GitHub。
- 这是 full-waypoint combo 分支证据，不是 true 3D、不是 metric、不是 seconds-level、不是 foundation。

## 4. 失败路线总表

| 路线 | 结果 | 失败原因 | 后续处理 |
| --- | --- | --- | --- |
| JEPA-only / Stage18 | non-collapse 但无 downstream lift | latent variance 没有对齐 selector/failure/goal/correction | 降级为 auxiliary，不作为主 claim |
| Stage24 hard-class selector | t50 -43.3%，easy degr 11.33% | one-hot oracle label 噪声、margin 小、过度切换 | 改 expected-FDE / gain-harm / fallback |
| SDD -> external zero-shot | all -92.67%，t50 -278.57% | 坐标尺度、scene/goal、agent type、domain mismatch | 做 row geometry、relative target、scene goals |
| 普通 normalization / latent adapter | 不足以正迁移 | 分布距离缩小不等于预测有用 | 改成 selective transfer 与 history/goal prototypes |
| Stage35 selective transfer | all/hard 正但 t50=0 | all objective 淹没 t50，缺历史/目标原型 | Stage37 t50-specific 修复 |
| Stage38 bounded correction | not deployable | hard 少量改善但 easy/safety 不稳 | 不部署，保留 Stage37 |
| Stage39/40 ungated neural | unsafe | easy degradation 大，复制/扰乱 floor | 必须 protected neural |
| Stage42-I full static/context sequence | negative | static/context 全局混入伤 full-waypoint | Stage42-J static-gated |
| Stage42-M alpha distillation | t50 负 | slice-level teacher 太粗 | Stage42-N/O row-level gain/harm |
| Stage42-O strict gain/harm | t50 微负 | t50 仍未被足够重视 | Stage42-P t50-weighted repair |
| Stage42-Q report-level combo | 只能 preflight | 没有逐行预测 cache，无法做真实 combo eval | Stage42-R row prediction cache |

## 5. 成功路线总表

| 路线 | 成功点 | 证据 |
| --- | --- | --- |
| Stage26 cost-aware selector | SDD t50/hard/easy 同时过 gate | t50 +14.58%，hard +11.23%，easy +1.81% |
| Stage37 t50 transfer repair | external 可部署 selector | all +13.48%，t50 +8.46%，hard +15.54%，easy 0.041%，16/16 gates |
| Stage42-B protected external validation | source-level external validation pass | all ADE +0.2103，t50 +0.1365，hard +0.2038 |
| Stage42-C protected full-waypoint dynamics | 从 endpoint 推进到 full-waypoint | ADE all +0.1858，t50 +0.1480，FDE t50 +0.2158 |
| Stage42-H causal sequence history | 证明 history tokens 有贡献 | no-history 下降 t50 0.4578、hard 0.4708 |
| Stage42-J static-gated full-waypoint | static/context gated 后有效 | ADE all/t50/hard +0.0362/+0.0369/+0.0397 |
| Stage42-L horizon static gate | fresh checkpoint 修复 t50 符号 | ADE t50 从 -0.0122 到 +0.0020 |
| Stage42-P t50 gain/harm selector | explicit gain/harm t50 mean repair | ADE t50 +0.0066，14/14 gates |
| Stage42-R row-cache combo | Stage42-J/P combo 从 preflight 变成 row-level eval | ADE all/t50/hard +0.0524/+0.0379/+0.0548，t50 CI low +0.0277，15/15 gates |

## 6. 为什么当前 best 不是纯 neural，也不是纯 selector

纯 selector 的问题：

- 能部署，但不是完整 world dynamics。
- 对 full future waypoint / all-agent dynamics 解释力不足。
- 依赖手工 baseline family 和 policy 结构。

纯 neural 的问题：

- 无保护时 easy degradation 大。
- 可能产生不物理或过度修正的输出。
- 在 dataset-local / raw-frame 外部数据中很容易受 domain/scale/goal 缺失影响。

当前 best 的折中：

```text
neural world-state signal + Stage37/teacher floor + safe switch
```

它的意义是：

- neural 学习 history / goal / interaction / full-waypoint 的额外信号；
- floor 保证 easy case 不被破坏；
- gain/harm selector 决定什么时候允许 neural 接管或辅助。

这比纯 selector 更接近 world-state modeling，但仍不是 true 3D / foundation。

## 7. 当前最重要的研究结论

1. **Oracle headroom 不能直接变成 selector 成功。**  
   Stage24 就证明了 oracle headroom 大但 hard classification 会大失败。

2. **跨域泛化首先是数据几何和目标定义问题。**  
   Stage31-34 的失败说明，坐标/scale/scene/goal/horizon 不对齐时，模型越复杂越容易负迁移。

3. **t+50 需要专门建模，并且需要 row-level combo 才能稳定。**  
   Stage35 all/hard 正但 t50=0；Stage37 通过 history window + goal prototype 修复；Stage42-P 证明 full-waypoint t50 需要 t50-weighted gain/harm；Stage42-R 进一步证明 Stage42-J static expert 与 Stage42-P gain/harm 可以通过 row prediction cache 组合，得到正 t50 CI low。

4. **history 有用，但必须用 causal sequence encoder。**  
   flattened history 失败，Stage42-H sequence history 成功。

5. **scene/static/context 有用，但必须 gated。**  
   Stage42-I 全局 static 失败，Stage42-J static-gated 成功。

6. **JEPA 目前不是主贡献。**  
   它 non-collapse，但没有稳定 downstream lift；不能包装成 world model 成功。

7. **安全 floor 是当前方法的一部分。**  
   不是临时 hack。去掉 floor 的 neural 仍不安全。

## 8. 当前还没解决的问题

1. **Stage42-R 还需要更强统计和部署固化。**  
   Stage42-R row-cache combo 的 t50 CI low 已为正，但它仍需要更多 seeds/per-domain stress、固定 policy artifact、cache hash/schema freeze，才能作为更强论文级分支 claim。

2. **full A-journal-ready 仍未达到。**  
   Stage42-F 是强 paper package，但不是 full A-journal-ready：还缺完整 fresh retrained ablation、更多独立外部数据、metric/time calibration、floor-free safety。

3. **metric / seconds-level claim 仍不允许。**  
   没有全局 homography/scale/effective seconds 验证。

4. **Stage5C / SMC 仍不能执行。**  
   当前没有足够证据证明 latent generative 或 SMC readiness。

5. **JEPA downstream lift 仍未证明。**  
   后续如果继续 JEPA，必须以 downstream head lift 为目标，而不是只看 non-collapse。

6. **external domains 还需要更多独立验证。**  
   当前已有 ETH_UCY / TrajNet / UCY 证据，但还不能称 foundation 或广泛真实世界泛化。

## 9. 当前 best deployable 与不部署清单

当前 best deployable：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
with Stage37 / teacher safety floor
```

可作为候选但需注明边界：

```text
Stage42-C protected full-waypoint dynamics
Stage42-J static-gated full-waypoint policy evidence
Stage42-L fresh horizon-aware static-gated checkpoint
Stage42-P t50 gain/harm selector repair
Stage42-R row-cache static/gain-harm combo evidence
```

不部署：

```text
JEPA-only
ungated neural
Stage38 correction
Stage39/40 neural without fallback
Stage42-I full static/context sequence
Stage42-M alpha-distilled static gate
Stage42-N row static-gate pilot
Stage42-O strict gain/harm as t50 success
```

## 10. 下一步最短路径

1. **Stage42-R policy 固化与 stress test。**  
   Stage42-R 已完成 row-cache combo eval；下一步应冻结 combo policy/hash/schema，增加 seeds/per-domain stress，并确认它能否成为 full-waypoint 分支默认策略。

2. **继续统计加固。**  
   Stage42-R t50 CI low 已为正，但仍需要更多 seeds、bootstrap、held-out domain stress，避免只依赖当前 cache/eval split。

3. **继续 full retrained ablation。**  
   把 JEPA / full Transformer / full-waypoint-shape / no-scene / no-goal / no-interaction 这些未完全 fresh retrain 的部分补齐。

4. **做 metric/time audit，但不急着声称 metric。**  
   只有 homography/scale/effective seconds 证据足够时，才允许从 raw-frame/dataset-local 升级 claim。

5. **保持 Stage5C / SMC 禁止。**  
   除非未来 gates 明确通过且用户确认，否则只生成 plan，不执行。

## 11. 最关键文件

主 README：

```text
outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md
outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md
outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md
README_RESULTS.md
research_state.json
```

Stage42 关键报告：

```text
outputs/stage42_long_research/report_stage42_final.md
outputs/stage42_long_research/external_validation_stage42.md
outputs/stage42_long_research/full_waypoint_dynamics_stage42.md
outputs/stage42_long_research/retrained_ablation_stage42.md
outputs/stage42_long_research/sequence_ablation_stage42.md
outputs/stage42_long_research/sequence_full_waypoint_stage42.md
outputs/stage42_long_research/static_gated_full_waypoint_stage42.md
outputs/stage42_long_research/fresh_static_gated_checkpoint_stage42.md
outputs/stage42_long_research/horizon_static_gate_repair_stage42.md
outputs/stage42_long_research/policy_distilled_static_gate_stage42.md
outputs/stage42_long_research/row_gain_static_gate_stage42.md
outputs/stage42_long_research/explicit_gain_harm_selector_stage42.md
outputs/stage42_long_research/t50_gain_harm_selector_stage42.md
outputs/stage42_long_research/t50_static_expert_combo_stage42.md
outputs/stage42_long_research/row_prediction_cache_stage42.md
```

## 12. 给用户的最终简明结论

这个目标里成功的不是“直接训练一个无保护大模型”，而是逐步建立了一个受保护的、跨外部 top-down pedestrian 数据有正证据的 2.5D world-state candidate。

最重要的成功：

```text
Stage37 修复 external t+50，形成 deployable selector。
Stage42-B/C 把 protected neural 推到 external validation 和 full-waypoint dynamics。
Stage42-H/J 证明 causal history 和 gated static/context 有真实贡献。
Stage42-P 修复 explicit gain/harm selector 的 mean t+50。
Stage42-R 用 row prediction cache 完成 Stage42-J/P combo，并让 cached combo t+50 CI low 为正。
```

最重要的失败：

```text
JEPA-only 没有 downstream lift。
hard-class selector 会严重伤 easy。
SDD zero-shot external 会崩。
普通 domain alignment / latent adapter 不足。
ungated neural 不安全。
global static/context mixing 会伤 full-waypoint。
Stage42-P 的 t50 seed stability 还不够；Stage42-R 已修复 combo 分支的 t50 CI，但还需要更多 seed/domain stress 才能写成更强论文 claim。
```

当前 honest verdict：

```text
M3W 已经不是早期 SDD-only selector demo。
它现在是 protected 2.5D external world-state candidate。
但它仍不是 true 3D，不是 metric/seconds-level，不是 foundation world model。
如果写论文，主 claim 应该是：
  safety-floor-protected neural world-state dynamics for dataset-local top-down multi-agent trajectories
而不是：
  true 3D foundation world model
```

## Stage42-Q T50 Static Expert + Gain/Harm Combo

```text
source = cached_verified_report_level_preflight
verdict = stage42_q_preflight_partial_row_cache_required
gates = 7 / 7
diagnostic_ade_all_best_available = 0.0526457864037421
diagnostic_ade_t50_best_available = 0.036875348395170704
diagnostic_ade_hard_best_available = 0.0535270529782426
diagnostic_fde_t50_best_available = 0.11663789673246368
row_level_combo_status = attempted_not_completed
stage5c_executed = false
smc_enabled = false
```

Stage42-Q targets the complementarity between Stage42-J static-gated full-waypoint experts and Stage42-P t+50 gain/harm selector. If it is a preflight result, it is diagnostic only and not a deployable combo claim; a row-level NPZ prediction cache is required before a full validation-only combo can be treated as pipeline evidence.

## Stage42-R Row Prediction Cache + Combo Eval

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
cached_combo_ade_all = 0.05238704221741153
cached_combo_ade_t50 = 0.03793420310086152
cached_combo_ade_t50_ci_low = 0.02774018469754745
cached_combo_ade_hard_failure = 0.05479172593908743
cached_combo_ade_easy_degradation = 0.001101978371627214
cached_combo_fde_t50 = 0.10005888767615174
cache_dir = data/stage42_row_prediction_cache (not committed)
stage5c_executed = false
smc_enabled = false
```

Stage42-R builds a local NPZ row prediction cache for floor / Stage42-J static expert / Stage42-P t+50 gain-harm selected errors, then performs validation-only combo evaluation from cache. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

## Stage42-S Frozen Row Combo Policy

```text
source = fresh_run_from_stage42r_row_cache
verdict = stage42_s_frozen_row_combo_policy_pass
gates = 13 / 13
policy_hash = 33450e033e14b10293b8a10796d934d7689e39358ab5eaa338d684a36b015d3f
cache_hash = f338f5c57b735b013ca210e30e9a6bbcfeebb646d4e0bc2e7f9e799006ac4ed6
ade_all = 0.05238704221741153
ade_t50 = 0.03793420310086152
ade_t50_ci_low = 0.02774018469754745
ade_hard_failure = 0.05479172593908743
ade_easy_degradation = 0.001101978371627214
stage5c_executed = false
smc_enabled = false
```

Stage42-S freezes the Stage42-R row-cache combo into a lightweight policy artifact and reports per-domain/per-horizon stress. It remains dataset-local raw-frame 2.5D evidence and not a metric, seconds-level, Stage5C, or SMC result.

## Stage42-T UCY Unseen-Domain Transfer Attempt

```text
source = fresh_run
verdict = stage42_t_ucy_transfer_blocked_no_candidate_predictions
gates = 8 / 11
ucy_ade_all = 0.0
ucy_ade_t50 = 0.0
ucy_hard_failure = 0.0
ucy_easy_degradation = 0.0
available_nonfloor_source_for_ucy = False
stage5c_executed = false
smc_enabled = false
```

Stage42-T attempts a validation-only unseen-domain transfer rule for UCY. The current row cache has no non-floor Stage42-J/P UCY predictions, so UCY remains fallback-only; this is reported as a blocker, not as a success.
