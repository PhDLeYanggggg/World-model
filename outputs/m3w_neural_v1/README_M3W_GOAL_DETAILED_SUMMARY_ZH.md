# M3W 目标内研究总结：尝试路线、失败原因、成功结果与当前结论

来源状态：`cached_verified` 汇总 Stage18 到 Stage41 的已生成报告与 package evidence；Stage42-A/B/C/D/E/F/G/H/I/J/K/L/M 的新增审计、外部验证、full-waypoint、安全地板、论文包、重训消融、sequence ablation、sequence-to-waypoint、static-gated repair、fresh static-gated checkpoint、horizon-aware repair 和 policy-distilled static gate 结果按报告内 source 字段标记为 `fresh_run` / `cached_verified` / `not_run`。  
本文件是给项目长期目标使用的中文总 README：它不是论文结论包装，而是把真实做过的路线、失败原因、成功证据和仍然不能声称的内容集中写清楚。

## 0.0 给用户的目标级摘要

这个目标内我实际推进的是一条从 **SDD pixel-space selector** 到 **external dataset-local protected neural world-state candidate** 的长期路线。核心不是一次性训练出 foundation model，而是不断用 gate 排除错误路线：

1. 先证明原始 JEPA / selector / correction 在 SDD 上哪里无效。
2. 再把 SDD 数据做成 official pixel raw-frame benchmark，训练出 Stage26 cost-aware selector。
3. 然后把模型迁移到 external top-down pedestrian 数据，先失败，再通过 row geometry、train-only goals、history windows、goal prototypes、selective transfer 修复。
4. Stage37 形成第一个 external deployable selector：all、t+50、hard/failure 都为正，easy 安全。
5. Stage39/40 开始训练神经动力学，但无保护 neural 不安全，不能替代 Stage37。
6. Stage41/42 把神经动力学做成 protected candidate：Stage37/teacher floor 保护下，composite-tail / endpoint-to-full / full-waypoint / sequence ablation 都有正证据。
7. 目前最强可部署仍需要 safety floor；它比 selector-only 更像 world-state dynamics，但仍不是 true 3D / metric / foundation。

最关键的成功是：**M3W-Neural v1 composite-tail safe-switch bounded neural dynamics** 在 external dataset-local raw-frame benchmark 上形成稳定正提升，并且 Stage42-H/J/K/L 进一步证明 causal history、gated static/context、fresh static-gated checkpoint 与 horizon-aware t50 repair 对 full-waypoint world-state dynamics 有用。

最关键的失败是：**JEPA-only、ungated neural、hard-class selector、raw domain alignment、无保护 residual、全局 static/context mixing** 都不能部署。失败原因不是单一模型小，而是 leakage-safe 部署下 easy preservation、domain scale、goal/scene availability、horizon objective、static context overfit 同时约束了模型。

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

Stage42-F 已经整理出论文级证据包，但结论是：**这是强的 protected 2.5D external world-state manuscript package，不是 full A-journal-ready / foundation / true-3D 结论**。A刊候选还缺 full retrained ablation、更多独立外部数据、metric/time calibration，以及减少 Stage37/teacher floor 依赖的安全神经门控。

Stage42-G Phase1 之后，full retrained ablation gap 被推进了一步：已经 fresh 重训 external expected-FDE selector 的 history、neighbor、goal/scene、interaction、domain、safe-switch/floor 关键消融；但 JEPA / full Transformer / full-waypoint-shape 仍未在 Stage42-G 内重训，所以还不能说 Stage42-D 全组件 A刊级消融完成。

Stage42-H 进一步修复了 Stage42-G 的一个负结果：flattened history 在 ridge selector 下不稳定，但 causal temporal sequence encoder 证明 history tokens 对 t+50 和 hard/failure 有强正贡献。也就是说，history 不是没用，而是不能用简单 flatten + ridge 方式证明；它需要真正的时序编码器。

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

Stage42-B 最新外部 source-level 验证：

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
verdict = stage42_b_external_validation_pass_protected_neural_not_ungated
```

Stage42-C 最新 full-waypoint dynamics 证据：

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
verdict = stage42_c_full_waypoint_dynamics_pass
```

Stage42-D 到 F 的证据边界：

```text
Stage42-D causal ablation evidence = pass_with_retrain_boundary
Stage42-D gates = 12 / 12
Stage42-E safety floor research = pass
Stage42-E gates = 12 / 12
Stage42-F paper package = complete_not_full_a_journal_ready
Stage42-F gates = 12 / 12
full_a_journal_ready = false
```

Stage42-G Phase1 fresh retrained ablation：

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
phase1_not_full_stage42_d_completion = true
```

Stage42-H causal sequence ablation：

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

### 2.10 Stage42-B：外部 source-level 验证

尝试内容：

- 使用冻结后的 M3W-Neural v1 / Stage37 safety floor 组合策略，重新做外部 source-level split 和 frozen evaluation。
- 明确区分 protected policy 与 ungated neural。
- 对 external all、t+50、t+100 raw-frame diagnostic、hard/failure、easy degradation 重新统计。

结果：

```text
source = fresh_run
evaluated_rows = 55,528
all ADE improvement = 0.2103
t+50 ADE improvement = 0.1365
t+100 raw-frame diagnostic ADE improvement = 0.1469
hard/failure ADE improvement = 0.2038
easy degradation = -0.1451
ungated neural easy degradation = 1.2459
gates = 10 / 10
```

为什么这一步成功：

- 不是把 Stage37 结果简单重复，而是在冻结策略、source/fold stress 下重新验证。
- protected M3W 在 external dataset-local raw-frame 上保持正提升。
- 同时证明了 ungated neural 虽然 raw lift 更高，但 easy-case 安全失败，不能部署。

结论：

Stage42-B 支持 “protected neural world-state dynamics 有外部正贡献”，但也确认 Stage37/teacher floor 仍是方法的一部分，不可悄悄拿掉。

### 2.11 Stage42-C：full-waypoint / all-agent dynamics

尝试内容：

- 从 endpoint / tail evidence 推进到 future waypoint sequence。
- 训练 / 评估 full_trajectory_ensemble。
- 比较 ADE/FDE、t+50、t+100 raw-frame diagnostic、hard/failure、easy preservation、near-collision proxy。

结果：

```text
source = fresh_run
positive_domains = ETH_UCY, TrajNet
ADE all = 0.1858
ADE t+50 = 0.1480
ADE t+100 raw-frame diagnostic = 0.2286
ADE hard/failure = 0.1952
easy degradation = 0.0000
FDE all = 0.1938
FDE t+50 = 0.2158
near_collision_delta_005 = 0.0086
gates = 12 / 12
```

为什么这一步重要：

- 它把证据从“只选 endpoint / baseline policy”推进到 “future waypoint sequence / all-agent world-state”。
- 说明当前 M3W 已经不只是 selector demo，有了受保护的神经 dynamics 证据。

仍然的限制：

- full-waypoint model 还没有完全替代 composite-tail linear bridge 的 all-ADE。
- ungated full-waypoint neural 仍不安全。
- 仍然是 raw-frame / dataset-local，不是 metric / seconds-level。

### 2.12 Stage42-D：causal ablation evidence

尝试内容：

- 汇总并复核 no-history、no-neighbor、no-scene/goal、no-interaction、no-JEPA、no-Transformer、no-fallback 等 ablation evidence。
- 对 fresh safety / waypoint rows 做重新计算。
- 明确哪些是 Stage42 本轮 fresh，哪些是 Stage30/41 cached-verified。

结果：

```text
verdict = stage42_d_causal_ablation_evidence_pass_with_retrain_boundary
gates = 12 / 12
ablation_coverage = true
all_components_retrained_inside_stage42_d = false
```

为什么只能算 partial support：

- 组件覆盖齐全，但不是每个组件都在 Stage42-D 内重新训练。
- scene/goal/interaction/history/neighbor 的贡献可以作为当前证据链的一部分，但还不能当成最终论文级 full retrained ablation。

结论：

Stage42-D 足够支撑“当前证据链有 ablation coverage”，但 A刊最终稿还需要同一 protocol 下全量 retrained ablation。

### 2.13 Stage42-E：safety floor research

尝试内容：

- 研究能否去掉 Stage37/teacher floor。
- 比较 no-teacher、internal self-gate、uncertainty gate、harm gate、conformal gate、bounded no-switch 等策略。
- 检查 raw lift、easy degradation、proximity/collision safety。

结果：

```text
best_deployable_policy = current_composite_tail_policy
all = 0.2103
t+50 = 0.1365
t+100 raw-frame diagnostic = 0.1469
hard/failure = 0.2038
easy degradation = 0.0000
floor_conclusion = teacher_floor_required_for_current_deployment
gates = 12 / 12
```

失败路线：

- ungated endpoint/full-waypoint neural raw lift 很高，但 easy degradation 约 1.2459，不可部署。
- internal self-gate / uncertainty / harm / conformal gates 可以产生 raw lift，但违反 proximity/collision ceiling。
- bounded no-switch 不能稳定替代 Stage37 floor。

结论：

当前最安全的部署方式仍然需要 Stage37/teacher floor。去掉 teacher floor 是未来研究任务，不是当前可部署结论。

### 2.14 Stage42-F：论文证据包

尝试内容：

- 把 Stage42-A 到 E 的证据整理成 paper outline、method draft、experiment table、ablation table、failure taxonomy、model/data card、reproducibility 和 A-journal gap analysis。
- 明确哪些 claim 支持，哪些 claim 不支持。

结果：

```text
verdict = stage42_f_paper_package_complete_not_full_a_journal_ready
gates = 12 / 12
full_a_journal_ready = false
```

支持的 claim：

- protected external raw-frame 2.5D world-state dynamics improves over Stage37 / strongest floor。
- full-waypoint sequence dynamics exists beyond endpoint-only linear bridge。

不支持或只能部分支持的 claim：

- ungated neural can replace safety floor：不支持。
- metric or seconds-level pedestrian world model：不支持。
- true 3D / foundation world model：不支持。
- scene/goal/interaction/history/neighbor contribution：部分支持，还需要 full retrained ablation。

结论：

Stage42-F 已经形成可写论文草稿的证据包，但诚实结论是 `not yet full A-journal ready`。当前是强 protected 2.5D external world-state candidate，而不是最终 A刊级完成态。

### 2.15 Stage42-G Phase1：fresh retrained ablation

尝试内容：

- 不再只读 Stage30/41 的 cached ablation coverage。
- 在 external combined dataset 上重新 assemble causal feature matrix。
- 对每个 variant 在 train split 上 fresh refit expected-FDE selector。
- 在 val split 上选择 conservative safety policy。
- test split 只评一次。
- 每个 variant 做 3 seeds：11、17、23。

重训 variants：

```text
full_retrained_external
no_history
no_neighbor
no_goal
no_scene_goal
no_interaction
no_domain_expert
no_transformer_proxy_history_sequence
no_safe_switch
no_teacher_floor_proxy
```

结果：

```text
source = fresh_run
gate = 11 / 11
verdict = stage42_g_retrained_ablation_phase1_pass
full_all = 0.8122
full_t50 = 0.8462
full_t100_raw_frame_diagnostic = 0.9527
full_hard_failure = 0.8459
full_easy_degradation = -0.8413
```

主要贡献结论：

- `no_goal` / `no_scene_goal` 低于 full：goal/scene proxy 对 all 和 hard/failure 有正贡献，t50 贡献小但为正。
- `no_neighbor` / `no_interaction` 低于 full：neighbor / interaction 对 all、t50 和 hard/failure 有小但稳定正贡献。
- `no_safe_switch` / `no_teacher_floor_proxy` 在 all/hard 上低于 full，但 t50 更高，说明无保护切换可提高长时程但会改变安全/部署权衡；不能用它替代 protected policy。
- `no_history` 和 `no_transformer_proxy_history_sequence` 在这个 ridge expected-FDE selector 协议下反而强于 full：这说明 raw flattened history sequence 对该轻量 selector 不是稳定正贡献，可能需要真正 sequence model 或更强 regularization，而不能把 history contribution 硬写成已证明。
- `no_domain_expert` 几乎等于 full：当前 external source-level feature 下 domain one-hot 贡献很小。

仍然未完成：

- `no_jepa` 未在 Stage42-G Phase1 重训，因为当前 deployable path 本来不使用 JEPA；JEPA 仍是 diagnostic-only。
- full Transformer checkpoint retrain 未在 Phase1 执行；Phase1 是 ridge expected-FDE selector ablation，不是 Torch Transformer ablation。
- `no_endpoint_bridge` / `no_full_waypoint_shape` 仍依赖 Stage42-C 的 fresh comparison，尚未做全组件 waypoint-shape retraining。

结论：

Stage42-G Phase1 把 Stage42-D 的 “cached ablation coverage” 往 “fresh retrained ablation” 推进了一大步，但还不能宣布 A刊级 full retrained ablation 全完成。最诚实的下一步是继续做 JEPA/Transformer/full-waypoint-shape 的同协议重训消融。

### 2.16 Stage42-H：causal sequence ablation

尝试内容：

- 训练真正的 causal temporal sequence encoder，而不是把 history window flatten 后喂给 ridge selector。
- 输入只包含 current/past history sequence、static causal features、goal/scene proxy、neighbor/interaction proxy、horizon/domain metadata。
- family_fde / future endpoint 只作为 supervised label 和 evaluation label，不作为 inference input。
- 每个 variant 3 seeds：31、37、43。
- val 选择 safety policy，test 只评一次。

重训 variants：

```text
sequence_full_safe_switch
sequence_no_history_tokens
sequence_no_goal_scene_tokens
sequence_no_neighbor_interaction_tokens
sequence_no_domain_expert
sequence_full_no_safe_switch
```

结果：

```text
source = fresh_run
gate = 10 / 10
verdict = stage42_h_sequence_ablation_pass
sequence_full_all = 0.7785
sequence_full_t50 = 0.7834
sequence_full_t100_raw_frame_diagnostic = 0.9166
sequence_full_hard_failure = 0.8081
sequence_full_easy_degradation = -0.7684
```

主要贡献结论：

- 去掉 history tokens 后，all / t50 / hard 大幅下降：`t50_delta_full_minus_no_history = 0.4578`，`hard_delta = 0.4708`。这证明 history 对真正 sequence model 有强贡献。
- 去掉 domain expert 后 all / t50 / hard 也下降约 0.04，说明 domain conditioning 对 sequence encoder 比对 ridge selector 更有用。
- 去掉 goal/scene 后 all 和 hard 有小正贡献，但 t50 略负，说明 goal/scene 对短中程或 hard 更有帮助，对 t50 仍需更好建模。
- 去掉 neighbor/interaction 后 t50 有很小正贡献，但 hard 略负，说明当前 neighbor/interaction proxy 贡献弱且切片相关。
- `sequence_full_no_safe_switch` raw family-FDE 更高，但它没有完成 proximity/collision/deployment safety，因此不能直接替代 protected policy。

结论：

Stage42-H 纠正了 Stage42-G 对 history 的负面信号：history 贡献依赖时序模型。它把 “history 是否有效” 从不稳定推进到 fresh positive sequence evidence。但它仍是 dataset-local raw-frame 2.5D expected-FDE family selection evidence，不是 Stage5C、不是 SMC、不是 metric/seconds-level。

### 2.17 Stage42-I：sequence-to-full-waypoint dynamics

尝试内容：

- 把 Stage42-H 的 causal temporal sequence encoder 接到 Stage42-C 的 reconstructed full-waypoint labels。
- 预测 4 个 future waypoints 的 relative displacement，而不是只预测 endpoint / family-FDE。
- 输入是 past-only all-agent tokens、neighbor tokens、static causal context、horizon/domain metadata。
- future waypoints / endpoints 只作为 loss/eval label，不作为 inference input。
- 每个 variant 3 seeds：53、59、61。
- val 选择 risk/safe-switch policy，test 只评一次。

重训 variants：

```text
sequence_waypoint_full
sequence_waypoint_no_history
sequence_waypoint_no_neighbor
sequence_waypoint_no_static_context
```

结果：

```text
source = fresh_run
gate = 10 / 11
verdict = stage42_i_sequence_full_waypoint_partial
sequence_waypoint_full_ade_all = -0.0106
sequence_waypoint_full_ade_t50 = -0.0321
sequence_waypoint_full_ade_hard_failure = -0.0116
sequence_waypoint_full_ade_easy_degradation = 0.0
sequence_waypoint_full_fde_t50 = 0.0201
history_ade_t50_delta_full_minus_no_history = 0.0040
history_fde_t50_delta_full_minus_no_history = 0.0094
best_variant = sequence_waypoint_no_static_context
best_variant_ade_all = 0.0115
best_variant_ade_t50 = 0.0199
best_variant_ade_hard_failure = 0.0129
best_variant_fde_t50 = 0.0611
best_variant_easy_degradation = 0.0
```

主要贡献结论：

- full static+sequence model 没过 gate：ADE all / t50 / hard 为负，不能部署。
- history 对 full-waypoint ADE/FDE 有小但正的贡献：去掉 history 后 t50 ADE 和 t50 FDE 都下降。
- no-neighbor 基本接近 full，说明当前 neighbor token 对 full-waypoint 仍弱。
- `sequence_waypoint_no_static_context` 反而是最强变体，all/t50/hard/FDE t50 都为正且 easy degradation 为 0。
- 这说明当前 static/context features 在 full-waypoint head 中可能存在 domain/scale overfit；下一步应该训练 static-gated 或 static-dropout sequence-to-waypoint head，而不是无条件混入全部 static context。

结论：

Stage42-I 没有让 full sequence-to-waypoint 主模型过 gate，但它把失败原因定位得很具体：causal sequence history 是有用的，unconditional static/context mixing 是当前 full-waypoint ADE 负贡献的主要嫌疑。它是一个有价值的 partial/failure result，不能包装成成功。

### 2.18 Stage42-J：static-gated full-waypoint repair

尝试内容：

- 不再无条件把 static/context 特征混入 full-waypoint head。
- 使用 Stage42-I 的 cached-verified `sequence_waypoint_full` 和 `sequence_waypoint_no_static_context` checkpoints 作为 experts。
- 构建 static mix experts：`no_static`、`static_alpha025`、`static_alpha050`、`static_alpha075`、`full_static`。
- 在 validation 上按 domain+horizon 选择 expert 和 safe-switch policy。
- test 只评一次。
- 结果来源明确标为 `cached_verified_checkpoints_fresh_static_gate_eval`，不是新的 checkpoint 训练。

结果：

```text
source = cached_verified_checkpoints_fresh_static_gate_eval
gate = 10 / 10
verdict = stage42_j_static_gated_full_waypoint_pass
static_gated_ade_all = 0.0362
static_gated_ade_t50 = 0.0369
static_gated_ade_t100_raw_frame_diagnostic = 0.0267
static_gated_ade_hard_failure = 0.0397
static_gated_ade_easy_degradation = 0.0
static_gated_fde_all = 0.0633
static_gated_fde_t50 = 0.1166
```

对比：

```text
full_static ADE all/t50/hard = -0.0106 / -0.0321 / -0.0116
no_static ADE all/t50/hard = 0.0115 / 0.0199 / 0.0129
static_alpha025 ADE all/t50/hard = 0.0352 / 0.0344 / 0.0387
static_alpha050 ADE all/t50/hard = 0.0324 / 0.0134 / 0.0356
static_gated ADE all/t50/hard = 0.0362 / 0.0369 / 0.0397
```

主要贡献结论：

- Stage42-J 修复了 Stage42-I 的 static/context failure mode：问题不是 static/context 永远无用，而是不能全局强制使用。
- partial static expert，尤其 `alpha=0.25`，比 no-static 更强；full-static 仍然有害。
- static gate 在 val 上选择 partial-static / floor / occasional full-static slice，test 上取得 all/t50/hard/FDE 全部正，并保持 easy degradation 为 0。
- 这一步加强了 full-waypoint world-state evidence，但它是 cached checkpoint + fresh gate/eval，不是 fresh checkpoint retrain。

结论：

Stage42-J 是一个有效的 policy-level repair。它把 Stage42-I 的 partial/failure 推进为 static-gated full-waypoint pass，但更强论文证据仍需要下一步把 static gate / static dropout bake into training，训练 fresh checkpoint。

### 2.19 Stage42-K：fresh static-gated checkpoint training

尝试内容：

- 把 Stage42-J 的 static-gated / static-dropout 思路直接训练进 checkpoint，而不是只做 cached experts 的 policy gate。
- 模型是 `StaticGatedSequenceWaypoint`。
- 使用 3 seeds：71、73、79。
- 使用 learned static gate，初始偏向低 static 使用，并带 static dropout / gate regularization。
- 输入仍是 past-only causal features；future waypoints 只作为 loss/eval label。
- val 选择安全 policy，test 只评一次。

结果：

```text
source = fresh_run
gate = 9 / 9
verdict = stage42_k_fresh_static_gated_checkpoint_pass
fresh_static_gated_ade_all = 0.0136
fresh_static_gated_ade_t50 = -0.0122
fresh_static_gated_ade_t100_raw_frame_diagnostic = 0.0159
fresh_static_gated_ade_hard_failure = 0.0148
fresh_static_gated_ade_easy_degradation = 0.0
fresh_static_gated_fde_all = 0.0312
fresh_static_gated_fde_t50 = 0.0358
fresh_static_gate_mean_test = 0.1278
```

对比 Stage42-J：

```text
Stage42-J static_gated ADE all/t50/hard = 0.0362 / 0.0369 / 0.0397
Stage42-K fresh checkpoint ADE all/t50/hard = 0.0136 / -0.0122 / 0.0148
Stage42-J static_gated FDE t50 = 0.1166
Stage42-K fresh checkpoint FDE t50 = 0.0358
```

主要贡献结论：

- Stage42-K 是一个真正的 fresh checkpoint training，不再只是 cached experts gate。
- 它比 Stage42-I 失败的 full static+sequence head 更好：all / hard / t100 raw-frame diagnostic / FDE t50 为正，easy degradation 为 0。
- 它没有超过 Stage42-J policy-level static gate，尤其 ADE t50 仍为负。
- 这说明 static gate 可以被训练进模型，但当前训练目标还没有把 t+50 slice 学稳；下一步应该做 horizon-aware static gate 或 t50-weighted static-dropout objective。

结论：

Stage42-K 通过了 fresh checkpoint gate，但不是新 best deployable。当前最强 static-gated full-waypoint evidence 仍是 Stage42-J；Stage42-K 的价值是把 Stage42-J 的 policy-level 发现推进成可训练 checkpoint 证据，并暴露 t50-specific gap。

### 2.20 Stage42-L：horizon-aware / t50-weighted static gate repair

尝试内容：

- 专门修复 Stage42-K 的 `ADE t50 < 0` 问题。
- 在 static gate 中加入 horizon embedding，让模型按 `t10/t25/t50/t100` 学不同的 static/context 使用策略。
- 对 t+50 降低 static dropout、降低 gate penalty、提高 waypoint/FDE 末端权重。
- validation policy search 明确提高 t50 权重，test 仍只评一次。
- 仍然禁止 future endpoint / future waypoint 作为 inference input。

结果：

```text
source = fresh_run
gate = 11 / 11
verdict = stage42_l_horizon_static_gate_repair_pass
horizon_static_gate_ade_all = 0.0219
horizon_static_gate_ade_t50 = 0.0020
horizon_static_gate_ade_t100_raw_frame_diagnostic = 0.0002
horizon_static_gate_ade_hard_failure = 0.0240
horizon_static_gate_ade_easy_degradation = 0.0
horizon_static_gate_fde_t50 = 0.0532
horizon_static_gate_t50_mean = 0.1903
```

对比：

```text
Stage42-K ADE all/t50/hard = 0.0136 / -0.0122 / 0.0148
Stage42-L ADE all/t50/hard = 0.0219 / 0.0020 / 0.0240
Stage42-J ADE all/t50/hard = 0.0362 / 0.0369 / 0.0397
```

主要贡献结论：

- Stage42-L 修复了 Stage42-K 的 t50 ADE 负号问题，说明 horizon-aware gate 是必要修复。
- Stage42-L 同时提升 all、hard/failure、FDE t50，并保持 easy degradation 为 0。
- Stage42-L 仍然没有超过 Stage42-J policy-level static gate，所以它不是新 best deployable。
- 当前 static-gated branch 的最强 deployable evidence 仍是 Stage42-J；Stage42-L 是目前最强 fresh checkpoint evidence。

结论：

Stage42-L 是一次有效的 targeted repair：把 K 的 t50 失败变成正，但幅度还小。下一步不该再泛泛调 gate，而应该研究如何把 Stage42-J policy-level expert selection 蒸馏进 horizon/domain-aware fresh checkpoint，或做 t50 bootstrap/longer training 来确认 t50 正信号是否稳定。

### 2.21 Stage42-M：policy-distilled static gate checkpoint

尝试内容：

- 直接蒸馏 Stage42-J 的 domain/horizon static expert selection。
- teacher alpha 来自 Stage42-J validation-selected policy，不使用 test endpoints。
- 模型加入 domain embedding + horizon embedding，并用 teacher alpha 监督 static gate。
- 目标是缩小 Stage42-L fresh checkpoint 与 Stage42-J policy gate 之间的差距。

teacher alpha map：

```text
ETH_UCY|10 = 0.3333
ETH_UCY|25 = 0.0
ETH_UCY|50 = 0.1667
ETH_UCY|100 = 0.0
TrajNet|10 = 0.5
TrajNet|25 = 0.0
TrajNet|50 = 0.3333
TrajNet|100 = 0.6667
```

结果：

```text
source = fresh_run
gate = 10 / 12
verdict = stage42_m_policy_distilled_static_gate_partial
policy_distilled_ade_all = 0.0161
policy_distilled_ade_t50 = -0.0015
policy_distilled_ade_t100_raw_frame_diagnostic = 0.0021
policy_distilled_ade_hard_failure = 0.0177
policy_distilled_ade_easy_degradation = 0.0
policy_distilled_fde_t50 = 0.0729
policy_distilled_t50_gate_mean = 0.1805
```

对比：

```text
Stage42-L ADE all/t50/hard = 0.0219 / 0.0020 / 0.0240
Stage42-M ADE all/t50/hard = 0.0161 / -0.0015 / 0.0177
Stage42-M FDE t50 = 0.0729
Stage42-J ADE all/t50/hard = 0.0362 / 0.0369 / 0.0397
```

失败原因：

- Stage42-M 的 teacher 是 coarse domain/horizon alpha，不是 row-level gain/harm teacher。
- 它提高了 static usage，并改善了 FDE t50，但 ADE t50 重新变负。
- 说明简单蒸馏 Stage42-J 的 slice-level expert choice 不够；需要 row-level teacher，包括 expected ADE gain、harm risk、switchability，而不是只学 static alpha。

结论：

Stage42-M 是有价值的负结果：它证明“把 J 的 policy gate 直接蒸馏成 alpha”不能替代 Stage42-L，也不能追上 Stage42-J。当前 fresh checkpoint 最强仍是 Stage42-L；整体 static-gated full-waypoint 最强仍是 Stage42-J。

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
| ungated Stage42 full-waypoint / endpoint neural | 不可部署 | raw lift 高，但 easy degradation 和 proximity/collision safety 失败。 |
| internal self/uncertainty/harm/conformal gate 替代 teacher floor | 不可部署 | 可以产生 raw lift，但未通过 proximity/collision ceiling。 |
| Stage42-D 全组件 fresh retraining | 未完成 | 当前是 fresh rows + cached-verified component evidence，不是全量同 protocol retrain。 |
| Stage42-G flattened history proxy | 不稳定/负贡献 | 在 ridge expected-FDE selector 下 no_history 和 no_transformer_proxy_history_sequence 反而更强，说明 raw history flattening 需要 sequence model 或更强正则，不能作为已证明贡献。 |
| Stage42-G domain one-hot expert | 弱 | no_domain_expert 几乎等于 full，说明当前 external protocol 下 domain embedding 贡献很小。 |
| Stage42-H no-safe-switch raw sequence | 不能部署 | raw family-FDE 更高，但没有 proximity/collision/deployment safety，不能替代 protected policy。 |
| Stage42-I full static+sequence waypoint head | 部分失败 | protected ADE all/t50/hard 为负；no-static-context 变体为正，说明 static/context 混入导致 full-waypoint head 过拟合或跨域尺度不稳。 |
| Stage42-K fresh static-gated checkpoint | 部分成功 | fresh checkpoint 过 gate，all/hard/FDE 为正并保护 easy；但 t50 ADE 仍为负，且弱于 Stage42-J policy gate。 |
| Stage42-L horizon-aware static gate | 成功但非 best | 修复 Stage42-K t50 ADE 负值，但幅度小且仍弱于 Stage42-J policy gate。 |
| Stage42-M policy-distilled static gate | 部分失败 | FDE t50 提升，但 ADE t50 仍负；coarse domain/horizon alpha teacher 不足以学习 row-level gain/harm。 |

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
| Stage42-B protected external validation | 成功 | frozen protected M3W 在 external source/fold stress 下仍为正，ungated neural 仍不安全。 |
| Stage42-C full-waypoint dynamics | 成功 | full future waypoint / all-agent world-state evidence 在 ETH_UCY、TrajNet 为正。 |
| Stage42-D causal ablation coverage | 部分成功 | ablation coverage 完成，但不是所有组件在 Stage42-D 内 fresh retrain。 |
| Stage42-E safety floor study | 成功 | 证明 teacher floor 当前仍必要，不能部署 ungated neural。 |
| Stage42-F paper package | 成功但未达最终 A刊 | 证据包完整，claim boundary 清楚，但 full retrained ablation、metric/time、外部扩展仍缺。 |
| Stage42-G Phase1 retrained ablation | 部分成功 | fresh refit 了 10 个 external selector 消融，goal/scene、neighbor/interaction、safe-switch/floor 有贡献；JEPA/Transformer/full-waypoint-shape 重训仍未完成。 |
| Stage42-H causal sequence ablation | 成功 | 证明 history tokens 在真正 sequence encoder 下对 t50 和 hard/failure 有强正贡献，修复了 flattened-history 负结果。 |
| Stage42-I no-static sequence-to-waypoint variant | 局部成功 | 证明 causal sequence 可以转化成 full-waypoint ADE/FDE 正信号，但需要 static/context gating 后才能变成主模型。 |
| Stage42-J static-gated full-waypoint repair | 成功 | 证明 partial static/context 在 validation-gated 使用时能提升 full-waypoint ADE/FDE，同时保护 easy。 |
| Stage42-K fresh static-gated checkpoint | 成功但非 best | 证明 static gate/dropout 可以训练进 checkpoint，并修复 Stage42-I full-static 的部分失败；但 t50 仍需 horizon-aware 修复。 |
| Stage42-L horizon-aware/t50-weighted repair | 成功但非 best | 证明 t50-specific horizon gate 可以把 fresh checkpoint 的 t50 ADE 从负修到正，并保持 easy 安全。 |
| Stage42-M policy distillation | 负结果有用 | 证明 Stage42-J 不能只用 slice-level alpha 蒸馏，需要 row-level gain/harm teacher。 |

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
7. **full retrained ablation**：Stage42-G Phase1 已完成 external selector 关键 feature/safety 消融，但 A刊级最终 claim 还需要同一 protocol 下重训 JEPA、Transformer、full-waypoint-shape、endpoint bridge 等 ablation。
8. **teacher-floor dependence**：Stage42-E 证明当前 teacher floor 必要；下一步要研究 proximity-safe internal gate，减少 floor 依赖。
9. **更多独立外部数据**：需要再接入合法 top-down pedestrian/drone 数据源，而不是只依赖当前 converted external 状态。
10. **sequence-to-full-waypoint bridge**：Stage42-H 证明 sequence history 对 family selection 有用，但还要把 causal sequence encoder 直接接到 full-waypoint all-agent dynamics。
11. **static-gated full-waypoint repair**：Stage42-J 已完成 policy-level repair，Stage42-K 已完成 fresh checkpoint training，Stage42-L 已修复 fresh checkpoint 的 t50 ADE 负号；Stage42-M 证明 coarse policy alpha distillation 不够，下一步需要 row-level gain/harm teacher。

## 8. 直接回答

```text
是否训练了神经世界模型：是，protected neural dynamics 已训练和评估。
是否超过 Stage37：是，在 Stage37/teacher safety floor 与 composite-tail policy 保护下超过。
是否可以无保护替代 Stage37：否。
当前 best deployable 是谁：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。
JEPA 是否可部署：否。
Transformer 是否有贡献：纯 Transformer 不可部署；protected endpoint neural dynamics 有贡献。
当前是否只是 selector：不是，已有 protected neural dynamics、all-agent world-state、endpoint-to-full evidence，但仍依赖 safety floor。
Stage42 论文包是否 full A刊 ready：否，是 strong protected 2.5D manuscript package，还不是最终 A刊完成态。
Stage42-G 是否完成 full retrained ablation：否，只完成 Phase1；goal/scene、neighbor/interaction、safe-switch/floor 有 fresh 正贡献，history/domain 贡献不稳定或弱，JEPA/Transformer/full-waypoint-shape 仍 not_run。
Stage42-H 是否证明 history：是，在 causal temporal sequence encoder 下证明；但这不是 metric、不是 seconds-level，也不是无保护 deployment。
Stage42-I 是否完成 full-waypoint sequence integration：部分完成。history 转成 full-waypoint 有小正贡献，no-static sequence 变体为正，但 full static+sequence 模型未过 gate。
Stage42-J 是否修复 static/context：是，policy-level 修复成功；但还不是 fresh static-gated checkpoint training。
Stage42-K 是否完成 fresh static-gated checkpoint：是，fresh_run gates 9/9；但它不是新 best deployable，t50 ADE 仍为负。
Stage42-L 是否修复 K 的 t50：是，ADE t50 从 -0.0122 修到 +0.0020；但仍未超过 Stage42-J policy gate。
Stage42-M 是否完成 policy distillation：部分失败。FDE t50 提升到 0.0729，但 ADE t50 仍为负，未超过 Stage42-L/J。
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
- Stage42-B external validation：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/external_validation_stage42.md`
- Stage42-C full-waypoint dynamics：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/full_waypoint_dynamics_stage42.md`
- Stage42-D causal ablation：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/causal_ablation_stage42.md`
- Stage42-E safety floor：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/safety_floor_stage42.md`
- Stage42-F final report：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/report_stage42_final.md`
- Stage42-F A-journal gap：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/a_journal_gap_stage42.md`
- Stage42-G retrained ablation Phase1：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/retrained_ablation_stage42.md`
- Stage42-H causal sequence ablation：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/sequence_ablation_stage42.md`
- Stage42-I sequence-to-full-waypoint dynamics：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/sequence_full_waypoint_stage42.md`
- Stage42-J static-gated full-waypoint repair：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/static_gated_full_waypoint_stage42.md`
- Stage42-K fresh static-gated checkpoint training：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/fresh_static_gated_checkpoint_stage42.md`
- Stage42-L horizon-aware t50 static-gate repair：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/horizon_static_gate_repair_stage42.md`
- Stage42-M policy-distilled static gate：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/policy_distilled_static_gate_stage42.md`

## 10. 下一步最值得做

1. **row-level gain/harm teacher distillation**：Stage42-M 证明 slice-level alpha teacher 不够；下一步要蒸馏 row-level expected ADE gain、harm risk 和 switchability，而不是只学 domain/horizon alpha。
2. **proximity-safe internal gate**：减少 Stage37/teacher floor 依赖，但不能牺牲 easy/proximity/collision safety。
3. **Metric/time audit**：补 FPS、annotation stride、homography、scale；不完成前继续禁止 metric/seconds claims。
4. **新增外部 top-down 数据**：优先 legal scene image + trajectory 的 pedestrian/drone top-down 数据，扩大 external breadth。
5. **停止低收益方向**：不继续把 JEPA-only、hard-class selector、ungated residual、source-switch 微小 headroom 当主线。

## 11. 当前 GitHub/报告口径

这个中文 README 的结论应作为后续 GitHub README 和报告的口径：

- 可以说：M3W-Neural v1 是当前 strongest protected 2.5D neural world-state candidate。
- 必须说：它不是 true 3D、不是 metric、不是 seconds-level、不是 foundation。
- 可以说：Stage37/teacher floor 下的 composite-tail neural dynamics 有稳定正贡献。
- 可以说：Stage42 形成了 strong protected 2.5D manuscript package。
- 必须说：它 not yet full A-journal ready；ungated neural dynamics 不安全，JEPA 不可部署，Stage5C/SMC 未启用。
