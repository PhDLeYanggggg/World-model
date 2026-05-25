# M3W 目标内研究总结：尝试路线、失败原因、成功结果与当前结论

来源状态：`cached_verified` 汇总 Stage18 到 Stage41 的已生成报告与 package evidence，外加 Stage42-A 数据/标定审计的 `fresh_run` 结果。  
本文件是给项目长期目标使用的中文总 README：它不是论文结论包装，而是把真实做过的路线、失败原因、成功证据和仍然不能声称的内容集中写清楚。

## 0. 当前一句话结论

当前最强可部署候选不是原来的纯 selector，也不是无保护神经网络，而是：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
safety floor = Stage37 selector / teacher floor
policy = composite_tail
calibrated external domains = ETH_UCY, TrajNet, UCY
uncalibrated domain rule = fallback_to_stage37_floor
```

它已经从 SDD-only selector 走到了受保护的 2.5D 外部多智能体世界状态候选：有外部 dataset-local raw-frame 结果、bootstrap、多 seed、pure-UCY、all-agent、endpoint-to-full bridge 和 ablation evidence。

但是必须继续诚实承认：

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external 数据是 dataset-local / unverified weak-metric diagnostic，不可包装成统一真实物理米制。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 仍未全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation pretraining / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 仍不安全，当前部署仍需要 Stage37/teacher safety floor。

## 1. 当前最强结果

主 package 指标，相对于 Stage37/teacher safety floor：

| 指标 | 数值 |
| --- | ---: |
| gates | 41 / 41 |
| evaluation rows | 55,528 |
| all ADE improvement | 0.2103 |
| t+50 ADE improvement | 0.1365 |
| t+100 raw-frame diagnostic ADE improvement | 0.1469 |
| hard/failure ADE improvement | 0.2038 |
| easy degradation | 0.0000 |
| positive external domains | 3 |
| all-agent composite FDE improvement | 0.1982 |
| all-agent composite FDE@50 improvement | 0.1739 |

Composite-tail 2000-bootstrap 支持：

| slice | low | mid | high |
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

Strict pure-UCY neural retrain / select / test 证据：

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

Stage42-A 最新数据/标定审计：

```text
source = fresh_run
datasets_audited = 7
raw_paths_found = 6
converted_paths_found = 7
external_domains_ready_from_existing_state = OpenTraj, ETH/UCY, TrajNet++, UCY
metric_claim_ready_datasets = TGSIM diagnostic only
seconds_claim_ready_datasets = none
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
stage42_b_external_validation_ready = true
stage42_c_full_waypoint_prereq_ready = true
Stage5C_executed = false
SMC_enabled = false
stage42_a_gates = 7 / 7
```

## 2. 我们尝试过的主要路线

### 2.1 Stage18-19：SAM-JEPA-2.5D 与 WAM-style representation

尝试内容：

- SAM-JEPA-2.5D representation pretraining。
- WAM-style 数据策略拆分：simulation、real top-down trajectories、human/egocentric video。
- JEPA non-collapse 检查。
- downstream heads：selector、failure predictor、goal predictor、correction、official t+50。

结果：

```text
JEPA non-collapse = yes
downstream lift = no
deployable contribution = no
```

失败原因：

- JEPA latent 虽然没有 collapse，但没有对齐真正的部署目标。
- 下游任务需要的是 causal history、hard/easy/failure 结构、gain/harm 选择和 conservative fallback，而不是单纯 representation variance。
- 没有启用 latent generative rollout，也不能把 JEPA 说成生成式 world model。
- 因此继续堆 JEPA 不会自动解决 selector、failure、goal、correction 或 t+50。

结论：

JEPA 可以保留为 research auxiliary，但不是当前 deployable path。

### 2.2 Stage22-26：SDD official pixel-space benchmark 与 cost-aware selector

尝试内容：

- 用 SDD 构建 official pixel-space benchmark。
- 建立 scene packs、lazy medium index、no-leakage audit、strongest causal baselines。
- 构建 GoalBench、HardBench、BaselineFailureBench。
- 诊断 hard-class selector 为什么失败。
- 改成 expected-FDE / cost-aware / gain-harm / fallback selector。

重要结果：

```text
Stage26 t50 improvement ~= 14.58%
Stage26 hard/failure improvement ~= 11.23%
Stage26 easy degradation ~= 1.81%
Stage26 selector = 当时 SDD best deployable
```

成功点：

- expected-FDE selector 优于 hard baseline-class 分类。
- conservative fallback 能保护 easy cases。
- failure predictor、gain predictor、harm predictor 是有效结构。

失败过的路线：

- Stage24 validation-selected hard-class selector 有很大 oracle headroom，但部署结果反而崩。
- t+50 improvement 为负，easy degradation 高。

失败原因：

- oracle best baseline 的硬标签低 margin、噪声大、很多样本 best 与 second-best 差距很小。
- hard classification 会在 easy cases 上过度切换。
- 学会“哪个 baseline 是 oracle best”不等于学会“什么时候值得切换”。

结论：

Stage26 证明了 selector 必须是 cost-aware / regret-aware，并且必须 fallback-safe。

### 2.3 Stage31-34：SDD 到 external 的 zero-shot 和 domain alignment

尝试内容：

- 把 OpenTraj、ETH-UCY、TrajNet、UCY 等外部 top-down pedestrian 数据转成 M3W schema。
- external no-leakage audit。
- external strongest baseline。
- SDD -> external zero-shot transfer。
- normalization、CORAL、latent alignment、coordinate-invariant features、relative-FDE target。
- external row geometry、train-only goals、scene packs。

初始失败结果：

```text
Stage31 zero-shot external all improvement ~= -92.67%
t50 ~= -278.57%
```

失败原因：

- SDD pixel 与 external local coordinates 不兼容。
- scene/goal/interaction context 缺失。
- agent type convention 不一致。
- scale / homography 未验证。
- horizon 与 track length 分布不匹配。
- SDD selector 太 SDD-specific。

Stage32-34 的结果：

- normalization 和 latent alignment 能缩小分布距离，但没有稳定 predictive lift。
- external row geometry 和 train-only goals 带来局部信号。
- t+50 / hard 上开始有正信号，但 all-test 和 easy safety 还不过。

结论：

单纯域对齐不够。必须补 causal row geometry、history window、goal prototype，并做 selective transfer。

### 2.4 Stage35-36：external selective transfer 与 t+50 blocker 定位

Stage35 尝试内容：

- 扩容 external 数据。
- 建 external official split v2。
- 建 external hard/easy/failure labels。
- 训练 selective transfer policy。

Stage35 结果：

```text
external test rows = 66,303
t+50 rows = 16,263
t+100 rows = 10,008
all improvement = 12.13%
hard/failure improvement = 13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
verdict = not deployable
```

为什么不算成功：

- all 和 hard/failure 正，但 t+50 仍 0。
- 长时程是世界模型核心 gate 之一，t+50 不过不能说 external deployable。

Stage36 诊断：

- t+50 rows 足够。
- t+50 oracle headroom 约 22.98%，不是没有学习空间。
- 现有特征和策略不敢在 t+50 上安全切换。
- all-test objective 把 t+50 长时程目标淹没了。
- 继续调 threshold 不够。

结论：

需要 t+50 专用 past-only history window、scene-agnostic goal prototype 和 switchability model。

### 2.5 Stage37：external t+50 修复成功

尝试内容：

- 构建 external past-only history windows：K=8/16/32/64。
- 构建 scene-agnostic goal prototypes：straight、slow_stop、left_turn、right_turn、u-turn、group_follow、density_avoid、exit_like_direction。
- 为 t+50 单独训练 failure/gain/harm switchability。
- 使用 conformal / conservative safety guard。

结果：

```text
all improvement = +13.48%
t+50 improvement = +8.46%
t+50 bootstrap CI = [+7.69%, +9.15%]
hard/failure improvement = +15.54%
easy degradation = 0.041%
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

为什么成功：

- 只使用过去，不用 future endpoint 做输入。
- 不用 held-out test endpoints 建 goals，而是用相对运动 goal prototypes。
- t+50 不再被 all-horizon selector 淹没。
- 只有 gain 高、harm 低、confidence 足够时才切换，否则 fallback。

结论：

Stage37 是第一个 external deployable positive transfer。它仍主要是 selector-level / policy-level 成功，但为后续 neural dynamics 提供了可靠 safety floor。

### 2.6 Stage38：bounded correction / dynamics head 失败

尝试内容：

- 冻结 Stage37 policy。
- 训练 bounded correction head：

```text
prediction = selected_baseline + alpha * bounded_delta
```

- 比较 correction without fallback、with fallback、hard-only、t50-only。

结果：

```text
Stage37 policy frozen = yes
bounded correction = not deployable
current best remained Stage37 selector
```

失败原因：

- correction 在 hard/tail 局部可能有提升，但不能稳定超过 Stage37。
- easy-case preservation 脆弱。
- bounded residual 不足以证明真正安全的 dynamics lift。

结论：

不能部署普通 residual/correction。必须先有更强神经 dynamics evidence，并保持 Stage37 fallback。

### 2.7 Stage39-40：Transformer / JEPA / Hybrid 神经动力学初期失败

尝试内容：

- Causal Temporal Transformer。
- JEPA auxiliary representation。
- JEPA + Transformer hybrid。
- teacher distillation。
- horizon-specific heads。
- failure/gain/harm targets。
- hard/t50 curriculum。
- 自动优化 trials。

结果：

```text
Transformer-only = trained but did not beat Stage37
JEPA = non-collapse historically, but downstream lift negative or absent
Hybrid = did not beat Stage37
Stage37 remained current external best
```

失败原因：

- weak neural proposals 被 safety fallback 吃掉，无法贡献部署提升。
- no-fallback neural 在 easy cases 上不安全。
- JEPA latent 没有转化成 selector/failure/trajectory downstream lift。
- Transformer 学到局部动力学，但没有学好安全切换/gain/harm 结构。
- Hybrid 继承了 JEPA downstream 弱的问题。
- t+100 仍缺少强 long-horizon scene/domain context。

结论：

“训练了神经网络”不等于“可部署 world dynamics”。需要让 neural 在 Stage37 保护下只做有把握的 bounded improvement。

### 2.8 Stage41：protected neural dynamics breakthrough

尝试内容：

- 重建 external split 和 world-model dataset。
- 构建 past-only seq2seq / all-agent dataset。
- 训练 protected endpoint neural dynamics。
- 训练 teacher-guided repair。
- 训练 group consistency / all-agent consistency。
- 训练 composite-tail safe-switch bounded neural dynamics。
- 做 endpoint-to-full bridge 和 full-waypoint evidence。
- 做 bootstrap、多 seed、pure-UCY、ablation、no-fallback 对照。

成功路线：

```text
Stage37/teacher floor
protected endpoint neural dynamics
self-gated endpoint candidate
composite-tail safe-switch bounded neural dynamics
endpoint-to-full bridge
all-agent composite world-state
```

最佳 same-protocol protected endpoint candidate：

```text
all = 0.4196
t50 = 0.4062
t100 raw-frame diagnostic = 0.4573
hard/failure = 0.4361
easy = 0.0
positive domains = 3
```

最终 packaged deployable candidate：

```text
all = 0.2103
t50 = 0.1365
t100 raw-frame diagnostic = 0.1469
hard/failure = 0.2038
easy = 0.0
positive domains = 3
```

为什么 Stage41 成功：

- 没让 ungated neural 替代 safety floor。
- neural 只在 safe-switch / composite-tail 下贡献。
- 用 bounded tail，而不是 unbounded residual。
- 把 endpoint evidence 扩展到 full-waypoint bridge 和 all-agent world-state evidence。
- 保留了 bootstrap、多 seed、pure-UCY、ablation 证据。

结论：

M3W-Neural v1 是当前最强 protected 2.5D neural world-state candidate。它不是 foundation，不是 true 3D，但已经不是单纯 selector demo。

### 2.9 Stage42-A：数据/标定长线审计

尝试内容：

- 复核现有 local converted state。
- 审计 external domains 是否可继续做 Stage42-B external validation 和 Stage42-C full-waypoint dynamics。
- 检查 metric/seconds claim 是否允许。

结果：

```text
datasets_audited = 7
raw_paths_found = 6
converted_paths_found = 7
stage42_a_gates = 7 / 7
external validation ready = true
full-waypoint prereq ready = true
global metric claim allowed = false
global seconds claim allowed = false
```

结论：

可以继续做外部验证和 full-waypoint dynamics，但仍不能声称 metric/seconds-level 或 foundation。

## 3. 失败路线总表

| 路线 | 状态 | 失败原因 |
| --- | --- | --- |
| Stage18/19 JEPA-only representation | 不可部署 | non-collapse 没有变成 selector/failure/goal/correction/t+50 lift。 |
| Stage24 hard-class selector | 失败 | oracle headroom 大，但 hard label 低 margin，导致 over-switch 和 easy degradation。 |
| SDD -> external zero-shot | 失败 | 坐标、scale、scene/goal、agent type、horizon 全部存在 domain gap。 |
| raw normalization / CORAL / latent alignment only | 失败 | 分布距离变小不等于 predictive lift。 |
| mixed-domain selector without safety | 失败 | 平均值局部改善，但 easy cases 被破坏，不可部署。 |
| Stage34/35 global external transfer | 部分失败 | hard/t50 有信号，但 all/easy/t50 不能同时过 gate。 |
| Stage38 bounded correction | 不可部署 | 不能稳定超过 Stage37，easy preservation 脆弱。 |
| Stage39 pure Transformer | 不可部署 | 学到局部动态，但不能在安全部署策略下超过 Stage37。 |
| Stage39/41 JEPA-only | 不可部署 | Stage41 JEPA auxiliary: all -0.0268, t50 -0.0151, hard -0.0243, easy degradation 0.0269。 |
| Stage39/41 Hybrid JEPA+Transformer | 不可部署 | fallback-only 或负贡献，没有稳定 dynamics lift。 |
| mixture selector | 不可部署 | 安全但没有 lift，等价 fallback-only。 |
| no-fallback neural | 安全失败 | hard/all 有时好，但 easy degradation 太大。 |
| continuous full-row bounded blend | 安全失败 | all/t50/t100/hard 正，但 easy degradation 约 0.207，高于 <=2% gate。 |
| dynamic/calibrated/pairwise source switching | 大多失败 | fixed composer 后剩余 oracle headroom 很小，positive residual rows 约 0.1%。 |
| learned full-waypoint shape alone | 弱 | 安全正贡献存在，但 shape gain 很小，主要是 tail-specific。 |

## 4. 成功路线总表

| 路线 | 状态 | 证明了什么 |
| --- | --- | --- |
| Stage26 cost-aware SDD selector | SDD 成功 | expected-FDE、gain/harm、fallback 是 robust selector 的必要结构。 |
| Stage37 causal history + goal prototype selector | external 成功 | external t+50 可以用 past-only history、goal prototype 和 conservative switching 修复。 |
| Stage41 external split + no-leakage dataset | 成功 | external train/val/test discipline 可以建立，且无 future/test leakage。 |
| Stage41 self-gated endpoint candidate | 神经证据成功 | protected neural endpoint dynamics 可以超过 Stage37。 |
| Stage41 composite-tail safe-switch | 当前 packaged success | bounded neural tail 在保护 easy 的同时提供稳定 lift。 |
| strict pure-UCY neural retrain | 成功 | source-heldout neural branch 可在 UCY-style 数据上 bootstrap-stable positive。 |
| endpoint-to-full trajectory bridge | 成功 | endpoint neural dynamics 投影到 full future waypoints 仍在 ETH_UCY/TrajNet 正。 |
| all-agent composite world-state | 成功 | 证据不是单 agent endpoint 选择，all-agent future world-state metrics 也为正。 |
| ablation coverage | 成功 | no-history、no-neighbor、no-scene/goal、no-interaction、no-JEPA、no-Transformer、no-fallback 都已覆盖。 |
| Stage42-A data calibration | 成功 | 当前数据足够继续 Stage42-B/C，但 metric/seconds claim 仍不允许。 |

## 5. 当前模型到底是什么

当前可部署路径不是普通 JEPA，也不是普通 Transformer，也不是无保护 residual。

实际部署逻辑：

1. Stage37/teacher floor 给出安全基线。
2. neural endpoint dynamics 提出 bounded improvement。
3. composite-tail policy 只在安全阈值下允许 neural switch 或小幅 tail contribution。
4. easy rows 回退到 safety floor。
5. 未校准 domain 回退到 Stage37 floor。

输入仍然是 causal-only：

- past history windows
- causal velocity / acceleration
- neighbor history / same-frame interaction context
- scene-agnostic goal prototype features
- horizon / domain metadata
- selected baseline rollout diagnostics computed without future labels

future endpoints / future waypoints 只用于训练 loss 和 evaluation label，不作为 inference input。

## 6. 为什么现在还不能叫 true world foundation model

可以称为：

```text
protected 2.5D multimodal multi-agent world-state candidate
dataset-local raw-frame external top-down pedestrian evidence
safe-switch bounded neural dynamics under Stage37/teacher floor
```

不能称为：

```text
true 3D world model
metric world model
seconds-level long-horizon model
large-scale foundation world model
ungated neural world dynamics deployment
Stage5C latent generative model
SMC-ready model
```

原因：

- metric scale / homography / FPS / annotation stride 仍未全局验证。
- external 数据仍是 dataset-local / weak-metric diagnostic。
- 当前 neural 成功依赖 safety floor。
- no-fallback neural 不安全。
- JEPA 没有证明 deployable downstream lift。
- 跨传感器、跨场景、跨大规模 video pretraining 还不够。

## 7. 当前最重要的剩余差距

1. **metric/time calibration**：必须验证 FPS、annotation stride、homography、meter-per-pixel，才能写 metric 或 seconds-level。
2. **ungated neural safety**：当前 neural 不能无保护部署，仍依赖 Stage37/teacher floor。
3. **JEPA contribution**：JEPA 仍是 diagnostic-only，没有 deployable downstream lift。
4. **full-waypoint dynamics**：endpoint-to-full bridge 有效，但 learned waypoint-shape contribution 小。
5. **external breadth**：ETH_UCY、TrajNet、UCY 为正，但还不是 foundation-scale 外部泛化。
6. **source/domain robustness**：fixed composer 之后的 dynamic source switching headroom 很小，需要新增 causal scene/domain features，而不是继续堆 source-switch learner。
7. **Stage42-B/C**：需要继续做 external validation source-level split 和 full-waypoint dynamics 加固。

## 8. 直接回答

```text
是否训练了神经世界模型：是，protected neural dynamics 已训练和评估。
是否超过 Stage37：是，在 Stage37/teacher safety floor 与 composite-tail policy 保护下超过。
是否可以无保护替代 Stage37：否。
当前 best deployable 是谁：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。
JEPA 是否可部署：否。
Transformer 是否有贡献：纯 Transformer 不可部署；protected endpoint neural dynamics 有贡献。
当前是否只是 selector：不是，已有 protected neural dynamics、all-agent world-state、endpoint-to-full evidence，但仍依赖 safety floor。
是否 true 3D：否。
是否 foundation：否。
Stage5C 是否可执行：否。
SMC 是否可启用：否。
```

## 9. 关键证据文件

- 主 package：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/report_m3w_neural_v1.md`
- 英文总账本：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_GOAL_SUMMARY_M3W_NEURAL_V1.md`
- 当前中文总账本：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md`
- completion audit：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/goal_completion_audit_m3w_neural_v1.md`
- evidence matrix：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.md`
- architecture ablation：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/neural_architecture_ablation_m3w_neural_v1.md`
- ablation coverage：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/ablation_coverage_m3w_neural_v1.md`
- model card：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/model_card_m3w_neural_v1.md`
- data card：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/data_card_m3w_neural_v1.md`
- reproducibility：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/reproducibility_m3w_neural_v1.md`
- paper gap：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/paper_gap_m3w_neural_v1.md`
- frozen policy：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/selector_policy_m3w_neural_v1.json`
- Stage42-A data calibration：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/data_calibration_stage42.md`

## 10. 下一步最值得做

1. **Stage42-B external validation**：做 source-level external validation，明确 frozen M3W-Neural v1 在不同 source/domain/scene/agent 切片上的稳定性。
2. **Stage42-C full-waypoint dynamics**：继续从 endpoint evidence 推进到更强 full future waypoint / all-agent dynamics，但保持 safety floor。
3. **Metric/time audit**：补 FPS、annotation stride、homography、scale；不完成前继续禁止 metric/seconds claims。
4. **新增外部 top-down 数据**：优先 legal scene image + trajectory 的 pedestrian/drone top-down 数据，扩大 external breadth。
5. **停止低收益方向**：不继续把 JEPA-only、hard-class selector、ungated residual、source-switch 微小 headroom 当主线。

## 11. 当前 GitHub/报告口径

这个中文 README 的结论应作为后续 GitHub README 和报告的口径：

- 可以说：M3W-Neural v1 是当前 strongest protected 2.5D neural world-state candidate。
- 必须说：它不是 true 3D、不是 metric、不是 seconds-level、不是 foundation。
- 可以说：Stage37/teacher floor 下的 composite-tail neural dynamics 有稳定正贡献。
- 必须说：ungated neural dynamics 不安全，JEPA 不可部署，Stage5C/SMC 未启用。

