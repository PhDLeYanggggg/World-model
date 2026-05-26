# M3W 长期目标研究总结：做过什么、哪些路线失败、哪些成功、为什么

更新时间：2026-05-26  
汇总状态：`cached_verified` 汇总 Stage18-Stage42 已生成报告；每个阶段内的新实验按原报告标注 `fresh_run` / `cached_verified` / `not_run`。  
用途：给长期目标状态机使用的中文总 README。它不是论文宣传稿，而是把真实尝试、失败原因、成功证据、当前 best deployable 和不能声称的边界集中写清楚。

## 0. 必须先写清楚的结论边界

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame / pixel-space 的 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external top-down pedestrian 数据仍是 dataset-local / unverified weak-metric diagnostic，不能写成统一真实物理米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 还没有全局验证。
- self-audited / visual-prior / auto-silver labels 不是 human gold。
- JEPA 在本项目里是 representation / auxiliary pretraining，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 仍不安全，当前可部署路径仍需要 Stage37 / teacher safety floor。

一句话现状：

```text
当前 best deployable = M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
safety_floor = Stage37 selector / teacher floor
claim = protected 2.5D external world-state candidate
not_claim = true 3D / metric / seconds-level / foundation world model
```

## 1. 当前最强可部署结果

当前主包来自 M3W-Neural v1 / Stage41-42 证据链，相对 Stage37 / teacher safety floor：

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

## 2. 做过的主路线

### 2.1 Stage18-19：JEPA / WAM-style representation

尝试：

- SAM-JEPA-2.5D representation pretraining。
- WAM-style data registry：simulation、real top-down trajectories、human/egocentric video。
- 检查 JEPA non-collapse。
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
- 没有启用 latent rollout，也不能把 JEPA 说成生成式 world model。

结论：

```text
继续盲目加 JEPA 不是主线；JEPA 必须证明 downstream lift 才能成为贡献。
```

### 2.2 Stage22-26：SDD pixel-space official benchmark 与 Stage26 selector

尝试：

- 把 SDD 转成 pixel raw-frame official benchmark。
- 构建 SDD scene packs、lazy medium index、no-leakage audit。
- 计算 strongest causal baseline。
- 建立 HardBench / BaselineFailureBench / GoalBench。
- 从 hard-class selector 改为 expected-FDE / cost-aware selector。

关键结果：

```text
Stage26 selector t50 improvement ≈ 14.58%
hard/failure improvement ≈ 11.23%
easy degradation ≈ 1.81%
Stage26 = 当时 SDD best deployable
```

失败过的路线：

- Stage24 hard-class selector 有 46.2% oracle headroom，但训练 selector t+50 improvement = -43.3%。
- easy degradation = 11.33%，说明 selector 大量伤害 easy cases。

失败原因：

- hard one-hot oracle label 噪声大，很多样本 best 和 second best margin 很小。
- 直接分类“哪个 baseline 最好”没有考虑切换代价。
- 没有 conservative fallback 时会过度切换 easy 样本。

成功原因：

- expected-FDE / gain/harm / regret-aware selector 更符合部署目标。
- fallback 到 strongest causal baseline 是必要安全机制。

### 2.3 Stage31-36：从 SDD 迁移到外部 top-down pedestrian 数据

尝试：

- OpenTraj / ETH-UCY / TrajNet / UCY 外部 feature store。
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

- 坐标尺度不兼容。
- scene/goal/interaction 缺失。
- agent type 约定不一致。
- homography / scale 未验证。
- SDD selector 明显 SDD-specific。

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
- 不是没有可学空间，而是现有特征/goal/context 不足以支持安全切换。
- threshold tuning 不能解决根因。

### 2.4 Stage37：外部 t+50 修复成功

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

- 它不再把 t+50 当作 all-horizon objective 的附属品。
- 不使用 test endpoints 建 goals，而是用 train 相对运动模式生成 scene-agnostic prototypes。
- 只在 gain 高、harm 低、confidence 高时切换，否则 fallback。

结论：

```text
Stage37 是第一个 external deployable selector candidate。
```

### 2.5 Stage38：bounded correction 失败

尝试：

- 冻结 Stage37 policy。
- 训练 bounded correction / dynamics head：
  `prediction = selected_baseline + alpha * bounded_delta`

结果：

```text
bounded correction = not deployable
current external best remained Stage37 selector
```

失败原因：

- correction 可以改善少量 hard/tail rows，但不能稳定超过 Stage37。
- easy preservation 很脆弱。
- 无法安全证明 correction_with_fallback all/t50/hard 超过 Stage37。

结论：

```text
不要部署 Stage38 correction；保留 Stage37。
```

### 2.6 Stage39-40：真正神经网络世界动力学训练，初期失败

尝试：

- Causal Transformer。
- JEPA auxiliary。
- JEPA + Transformer Hybrid。
- teacher distillation。
- horizon-specific heads。
- failure/gain/harm multitask targets。
- hard/t50 curriculum。

结果：

```text
Transformer-only = trained but did not beat Stage37
JEPA-only = non-collapse historically, but downstream lift remained absent/negative
Hybrid = did not beat Stage37
ungated neural = unsafe
Stage37 remained external best
```

失败原因：

- neural without fallback 会伤 easy cases。
- weak neural proposals 被 Stage37 fallback 吃掉，无法形成净提升。
- JEPA latent 没有稳定 downstream lift。
- Transformer 学到局部动态，但没有学到“何时安全切换”。
- Hybrid 继承了 JEPA downstream 无效的问题。

结论：

```text
神经网络必须在 Stage37 safety floor 保护下学习 bounded contribution，而不是直接替代 selector。
```

### 2.7 Stage41：protected neural dynamics 突破

成功路线：

- 重新建立 external split。
- 构建 seq2seq world-model dataset。
- 构建 all-agent dataset。
- endpoint dynamics + safety floor。
- composite-tail safe-switch bounded neural dynamics。
- endpoint-to-full bridge。
- all-agent composite world-state evaluation。

同协议 protected endpoint candidate：

```text
all = 0.4196
t50 = 0.4062
t100 raw-frame diagnostic = 0.4573
hard/failure = 0.4361
easy = 0.0
positive domains = 3
```

冻结 packaged deployable candidate：

```text
all = 0.2103
t50 = 0.1365
t100 raw-frame diagnostic = 0.1469
hard/failure = 0.2038
easy = 0.0
positive domains = 3
```

为什么成功：

- 不让 neural dynamics 全局替代 safety floor。
- neural 只提供 bounded tail improvement。
- policy frozen 后再 test，不用 test 调 threshold。
- bootstrap / multiseed / pure-UCY / all-agent / endpoint-to-full 都给了正证据。

结论：

```text
M3W 从 selector-only 进入 protected neural world-state candidate。
```

## 3. Stage42 长研究模式：证据补强与 full-waypoint 分支

### Stage42-A：数据/标定审计

```text
source = fresh_run
datasets_audited = 7
raw_paths_found = 6
converted_paths_found = 7
external_domains_ready = OpenTraj, ETH/UCY, TrajNet++, UCY
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
gates = 7 / 7
```

意义：

- 外部验证和 full-waypoint dynamics 可以继续。
- 仍不能做 metric 或 seconds-level claim。

### Stage42-B：外部 source-level 验证

```text
source = fresh_run
source_level_split_rebuilt = true
evaluated_rows = 55,528
protected_M3W_all_ADE_improvement = 0.2103
protected_M3W_t50_ADE_improvement = 0.1365
protected_M3W_t100_raw_frame_diagnostic_ADE_improvement = 0.1469
protected_M3W_hard_failure_ADE_improvement = 0.2038
protected_M3W_easy_degradation = -0.1451
ungated_neural_all_ADE_improvement = 0.2966
ungated_neural_easy_degradation = 1.2459
gates = 10 / 10
```

解释：

- protected M3W 稳定正迁移。
- ungated neural 即使 all 更高，也因 easy degradation 过大不可部署。

### Stage42-C：full-waypoint dynamics

```text
source = fresh_run
positive_full_waypoint_domains = ETH_UCY, TrajNet
protected_full_waypoint_ADE_all = 0.1858
protected_full_waypoint_ADE_t50 = 0.1480
protected_full_waypoint_ADE_t100_raw_frame_diagnostic = 0.2286
protected_full_waypoint_ADE_hard_failure = 0.1952
protected_full_waypoint_easy_degradation = 0.0000
protected_full_waypoint_FDE_all = 0.1938
protected_full_waypoint_FDE_t50 = 0.2158
gates = 12 / 12
```

意义：

- 不只是 endpoint selector，有 reconstructed full future waypoint evidence。
- 但仍需要 protection；ungated full-waypoint neural 不安全。

### Stage42-D/E/F：ablation、安全地板、论文包

结果：

```text
Stage42-D causal ablation evidence = pass_with_retrain_boundary
Stage42-D gates = 12 / 12
Stage42-E safety floor research = pass
Stage42-E gates = 12 / 12
Stage42-F paper package = complete_not_full_a_journal_ready
Stage42-F gates = 12 / 12
full_a_journal_ready = false
```

意义：

- 论文包结构完整。
- 但 full retrained ablation、更多外部数据、metric/time calibration、floor-free neural safety 仍不足。

### Stage42-G：fresh retrained selector ablation

```text
source = fresh_run
verdict = stage42_g_retrained_ablation_phase1_pass
gates = 11 / 11
variants = full, no_history, no_neighbor, no_goal, no_scene_goal, no_interaction, no_domain_expert, no_transformer_proxy_history_sequence, no_safe_switch, no_teacher_floor_proxy
seeds_per_variant = 3
full_all = 0.8122
full_t50 = 0.8462
full_t100_raw_frame_diagnostic = 0.9527
full_hard_failure = 0.8459
full_easy_degradation = -0.8413
```

意义：

- fresh retrained selector ablation 证明多个组件有贡献。
- 但 flattened history + ridge 协议下 history 信号不稳定，因此不能把它当成最终 history 结论。

### Stage42-H：causal temporal sequence ablation

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

意义：

- history 不是没用，flattened history 太弱。
- 真正 causal temporal encoder 可以提取强 t+50 / hard 信息。

### Stage42-I：sequence-to-full-waypoint 失败与诊断

结果：

```text
source = fresh_run
verdict = stage42_i_sequence_full_waypoint_partial
sequence_waypoint_full ADE all/t50/hard = negative
sequence_waypoint_no_static_context ADE all/t50/hard = positive
```

失败原因：

- 静态/scene/context 全局混入会伤害 full-waypoint ADE。
- 需要 static/context gating 或 static dropout。

### Stage42-J：policy-level static-gated repair 成功

```text
source = cached_verified_checkpoints_fresh_static_gate_eval
verdict = stage42_j_static_gated_full_waypoint_pass
ADE all = 0.0362
ADE t50 = 0.0369
ADE hard/failure = 0.0397
ADE easy degradation = 0.0
FDE all = 0.0633
FDE t50 = 0.1166
```

意义：

- static/context 不是没用，而是不能全局强混。
- validation-gated partial static experts 能修复 Stage42-I。

### Stage42-K/L/M/N：fresh static-gate checkpoint 分支

Stage42-K：

```text
fresh static-gated checkpoint = pass
ADE all = 0.0136
ADE t50 = -0.0122
hard = 0.0148
easy = 0.0
```

Stage42-L：

```text
horizon-aware repair = pass
ADE all = 0.0219
ADE t50 = 0.0020
hard = 0.0240
FDE t50 = 0.0532
easy = 0.0
```

Stage42-M：

```text
policy-distilled static gate = partial
ADE all = 0.0161
ADE t50 = -0.0015
hard = 0.0177
FDE t50 = 0.0729
easy = 0.0
```

Stage42-N：

```text
row-level gain/harm static-gate distillation = partial
ADE all = 0.0250
ADE t50 = -0.0278
hard = 0.0269
FDE t50 = 0.0555
easy = 0.0
```

分支结论：

- Stage42-J policy-level static gate 仍是最强 static-gated full-waypoint evidence。
- Stage42-L 是更好的 fresh checkpoint 修复，但 t+50 仍很弱。
- Stage42-M/N 证明 alpha-style static distillation 不够；需要显式 row-level gain/harm/switchability selector。

### Stage42-O：显式 gain/harm/switchability selector

Stage42-O 是对 Stage42-N 的直接修复：不再只监督 static-gate alpha，而是显式训练 row-level switch probability、expected gain、harm probability、uncertainty，再用 validation 选择 conservative policy。

严格修复点：

- 初版 val/test 特征标准化使用了各自 split stats，容易被视为 test-stat normalization 风险。
- 已改为只使用 train-split mean/std 标准化 val/test。
- gate 中新增 `no_test_statistics_normalization`。
- 以下结果是 train-only normalization 后的结果。

```text
source = fresh_run
verdict = stage42_o_explicit_gain_harm_selector_partial
gates = 13 / 14
ADE all = 0.0526
ADE t50 = -0.0008
ADE t100 raw-frame diagnostic = 0.0602
ADE hard/failure = 0.0535
ADE easy degradation = 0.0155
FDE t50 = 0.0576
feature_normalization = train_split_stats_only
no_test_statistics_normalization = true
```

解释：

- 显式 row-level selector 明显优于 Stage42-N：ADE all 从 `0.0250` 到 `0.0526`，hard/failure 从 `0.0269` 到 `0.0535`，t100 raw-frame diagnostic 也为正。
- 它没有通过 t+50 gate：ADE t50 是 `-0.0008`，虽然比 Stage42-N 的 `-0.0278` 大幅改善，但仍不能写成 t50 修复成功。
- easy degradation mean 为 `0.0155`，低于 2% mean gate，但仍需要后续 bootstrap/CI 检查，不能只看 mean。
- 下一步不是再做 alpha distillation，而是做 t+50-specific gain/harm teacher ensemble 或 per-domain horizon policy。

## 4. 失败路线总表

| 路线 | 状态 | 失败原因 |
| --- | --- | --- |
| JEPA-only | 失败 | non-collapse 没有转化为 downstream lift。 |
| Stage24 hard-class selector | 失败 | oracle margin 小、label ambiguity、大量 easy over-switch。 |
| SDD-to-external zero-shot | 失败 | 坐标、scale、goal/scene、agent type、horizon domain gap 太大。 |
| raw normalization / CORAL / latent alignment alone | 失败 | 缩小分布距离不等于预测增益。 |
| mixed-domain selector without safety | 失败 | all 有时提升，但 easy preservation 不过 gate。 |
| Stage34/35 global external transfer | partial/失败 | hard/t50 有局部信号，但 all/easy/t50 不能同时过。 |
| Stage38 bounded correction | 失败 | 不能安全超过 Stage37。 |
| Stage39 pure Transformer | 失败 | 不会稳定超过 Stage37，且易被 fallback 吃掉。 |
| JEPA+Transformer Hybrid | 失败 | JEPA downstream 无 lift，Hybrid 没有变成主贡献。 |
| no-fallback neural | 失败 | easy degradation 太大，不可部署。 |
| continuous full-row bounded blend | 失败 | all/t50/hard 正，但 easy degradation 约 0.207，远超 2% gate。 |
| fixed-prior / dynamic source switching | 大多失败 | residual oracle headroom 太小，真正可切换正样本极少。 |
| unconditional static/context full-waypoint model | 失败 | scene/static context 全局混入导致 ADE t50/hard 受损。 |
| alpha-only static-gate distillation | partial/失败 | 学到“用多少 static”，没学到“哪一行切换会 gain/harm”。 |

## 5. 成功路线总表

| 路线 | 状态 | 证明了什么 |
| --- | --- | --- |
| Stage26 expected-FDE SDD selector | SDD 成功 | selector 必须 cost-aware、regret-aware、fallback-safe。 |
| Stage37 history + goal prototype selector | external 成功 | past-only history 与 scene-agnostic goals 可修复 external t+50。 |
| Stage41 protected endpoint neural | 成功 | neural dynamics 在 safety floor 下能超过 Stage37。 |
| Stage41 composite-tail safe-switch | 当前主成功 | bounded neural tail 能稳定增益且保持 easy。 |
| pure-UCY retrain | 成功 | 独立 source 上 neural branch 可 bootstrap-stable 正迁移。 |
| endpoint-to-full bridge | 成功 | endpoint neural dynamics 的增益能转成 full future waypoint evidence。 |
| Stage42-C full-waypoint dynamics | 成功 | 不只是 endpoint，有 reconstructed full-waypoint raw-frame 证据。 |
| Stage42-H causal sequence encoder | 成功 | history tokens 对 t+50 / hard 有强贡献。 |
| Stage42-J validation-gated static experts | 成功 | static/context 有用，但必须 gated，不可全局混入。 |
| Stage42-L horizon-aware static gate | 局部成功 | fresh checkpoint 分支能把 t+50 从负修成小正。 |

## 6. 现在到底是什么质量

现在质量可以诚实描述为：

```text
strong protected 2.5D external multi-agent world-state candidate
not full A-journal-ready
not true 3D
not metric/seconds-level
not foundation model
not ungated neural deployment
```

它已经比早期 SDD-only selector 强很多，因为：

- 有 external dataset-local top-down pedestrian evidence。
- 有 protected neural dynamics，而不只是 selector。
- 有 all-agent world-state evidence。
- 有 endpoint-to-full waypoint evidence。
- 有 bootstrap / multiseed / ablation / no-leakage 报告。

但还不能写成 A刊完整主张，因为：

- 仍依赖 Stage37 / teacher floor。
- JEPA 没有稳定 downstream lift。
- metric/time calibration 未完成。
- external 数据广度仍有限。
- full retrained ablation 仍未完全覆盖所有神经组件。
- Stage42-O 严格 train-only normalization 后是 partial，不是 t+50 pass；说明 t50-specific switchability 仍是未解决点。

## 7. 下一步最短路径

1. 做 t+50-specific gain/harm teacher ensemble，避免 Stage42-O 在 train-only normalization 后 t50 仍小负。
2. 对 Stage42-O 类策略增加 bootstrap / seed variance，并检查 easy degradation CI，而不只看 mean。
3. 组合 Stage42-J policy static-gated experts 与 Stage42-O row-level gain/harm selector，而不是继续 alpha-only distillation。
4. 补 full retrained ablation：JEPA、Transformer、full-waypoint shape、no-fallback、no-static、no-history、no-neighbor、no-goal/scene 全部同协议重训。
5. 做 metric/time calibration：FPS、annotation stride、homography、meter-per-pixel。没有证据就继续写 raw-frame / dataset-local。
6. 增加外部 top-down 数据集，尤其带 scene image / homography / longer horizon 的合法数据。

## 8. 文件索引

- 主目标英文总结：`outputs/m3w_neural_v1/README_GOAL_SUMMARY_M3W_NEURAL_V1.md`
- 主目标中文详细总结：`outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md`
- 本文件：`outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md`
- Stage42 final package：`outputs/stage42_long_research/report_stage42_final.md`
- Stage42 paper gap：`outputs/stage42_long_research/a_journal_gap_stage42.md`
- Stage42 full-waypoint：`outputs/stage42_long_research/full_waypoint_dynamics_stage42.md`
- Stage42 static-gated repair：`outputs/stage42_long_research/static_gated_full_waypoint_stage42.md`
- Stage42 horizon-aware repair：`outputs/stage42_long_research/horizon_static_gate_repair_stage42.md`
- Stage42 row-gain static gate：`outputs/stage42_long_research/row_gain_static_gate_stage42.md`
- Stage42 explicit gain/harm selector：`outputs/stage42_long_research/explicit_gain_harm_selector_stage42.md`
