# M3W 长期目标总账：尝试路线、失败原因、成功证据与当前边界

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总 Stage18-Stage42 已生成报告、gate、README、`research_state.json`，并纳入最近 Stage42-CV/CW/CX/CY/CZ 的 runtime replay、paper refresh、provenance、worktree caveat、paper-freeze manifest 证据，Stage42 report-test isolation 的可复现性修复，以及 Stage42-DE/DF/DG/DH 对 full-waypoint primary deployment gap、all/hard/proximity repair、weighted-loss repair、proximity/occupancy-proxy loss repair 的最新 fresh evidence。
用途：这是当前 canonical 单文件中文总账，回答“在 M3W 这个长期目标内做了什么、试过什么路线、哪些失败、为什么失败、哪些成功、当前 best deployable 是谁、还有哪些不能 claim”。

这不是宣传稿，也不是论文最终版。凡是 `not_run`、technical dry-run、fallback-only、license-blocked、source-support 不足、metadata caveat，都不能写成完成或成功。

## 0. 一句话总判定

M3W 已经从 SDD-only 的 2.5D 轨迹 scaffold 推进到一个有外部 top-down 数据正迁移证据的 protected raw-frame / dataset-local 多智能体 world-state candidate。

当前最强可部署形态不是裸 Transformer、裸 JEPA、裸 Hybrid，也不是 Stage5C，而是：

```text
best deployable:
  M3W-Neural v1 / Stage42 proximity-guarded protected composer

safety floor:
  Stage37 selector / teacher floor

dominant effective mechanism:
  baseline-family rollout context
  + causal history window
  + validation-only gain/harm/easy guard
  + source/horizon-aware conservative fallback
  + proximity-aware safety guard
```

最重要边界：

```text
不是 true 3D world model
不是 large-scale foundation world model
不是 global metric predictor
不是 seconds-level long-horizon predictor
不是 ungated neural dynamics
不是 latent generative Stage5C execution
不是 SMC-ready model
```

当前最诚实的 claim：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

## 1. 必须一直保留的诚实边界

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 不可部署；当前强结果都依赖 safety floor / fallback / validation-only guard。
- 不能用 future endpoint、future waypoint、central velocity、test endpoint goals 或 test threshold tuning。

## 2. 当前最重要结果总表

| 阶段 | 结果 | 关键指标 | 当前解释 |
| --- | --- | ---: | --- |
| Stage26 | SDD cost-aware selector 成为 SDD best deployable | t+50 约 +14.58%；hard/failure 约 +11.23%；easy degradation 约 1.81% | SDD pixel/raw-frame 内成立，不是 metric/3D |
| Stage37 | external t+50 repaired deployable | all +13.48%；t+50 +8.46%；t50 CI [+7.69%, +9.15%]；hard/failure +15.54%；easy 0.041%；16/16 gates | 第一次把外部 t+50 从 0 修到可部署正迁移 |
| Stage40 | neural world model trials | best neural protected 指标接近 Stage37，但 `neural_exceeds_stage37=false`；部署保持 Stage37 | 裸 neural 灾难性失败，受保护 neural 没稳定超过 floor |
| Stage41/M3W-Neural v1 | protected neural package | all ADE +21.03%；t+50 +13.65%；t+100 raw diagnostic +14.69%；hard/failure +20.38%；easy 0.00%；41/41 gates | 有 protected neural evidence，但不是 ungated neural dynamics |
| Stage42-CO | common-validation bridge/shape composer | vs endpoint-linear：all +3.02%；t50 +1.50%；t100 raw +6.12%；hard +3.28%；easy +0.25%；14/14 gates | 补上 endpoint-linear vs full-waypoint common row alignment |
| Stage42-CP | composer safety/bootstrap | 2000 bootstrap：all CI [+2.64%, +3.37%]；t50 CI [+0.90%, +2.09%]；t100 CI [+5.39%, +6.94%]；hard CI [+2.90%, +3.68%] | 统计支持成立，但 near-collision 有小 caveat |
| Stage42-CQ | proximity-aware composer guard | all +1.77%；t50 +1.07%；t100 raw +3.48%；hard +1.93%；easy +0.25%；near@0.05 vs endpoint -0.06%；19/19 gates | 用 validation-only predicted-proximity guard 修复 CP proximity caveat |
| Stage42-CR | proximity guard Pareto audit | no-guard 更准但 near@0.05 +0.34%；guarded all/t50/t100/hard 仍为正且 near@0.05 -0.06%；19/19 gates | 明确 accuracy-priority vs safety-sensitive deployable 的 Pareto 边界 |
| Stage42-CV | frozen policy batch runtime replay | val 53256 rows，test 55528 rows；25/25 gates；runtime decisions 和 selected_xy/ADE/FDE 精确复现 | 冻结策略不是 report-only，能按 batch runtime 精确重放 |
| Stage42-CW | runtime replay paper refresh | 25/25 gate 证据写入 paper/reproducibility/model card | paper package 吸收 runtime replay，不隐藏部署形态 |
| Stage42-CX | evidence provenance verifier | 21 artifacts audited；21 gates passed；20 fresh_run，1 cached_verified | 证据来源和命令矩阵明确 |
| Stage42-CY | worktree caveat classifier | tracked dirty files 8；Stage42 dirty 0；Stage42 substantive dirty 0；11/11 gates | 当前 Stage42 tracked artifacts 已无 substantive dirty；剩余 dirty 为历史 Stage17-19 outside-scope 报告 |
| Stage42-CZ | paper freeze candidate manifest | 74 files hashed；14/14 gates；freeze_status = candidate_clean；final_immutable_release = true | 当前 paper evidence candidate 有 clean manifest，但 claim 仍限于 protected dataset-local/raw-frame 2.5D |
| Stage42 report-test isolation | pytest artifact hygiene | Stage42 report tests 改为写 `tmp_path`；focused tests 13 passed；full tests 615 passed | 修复 pytest 反复改写 tracked Stage42 evidence files 的 metadata churn |
| Stage42-DE | full-waypoint deployment gap audit | 17/17 gates；primary promotion blocked | full-waypoint 有辅助/guarded composer 价值，但不能升为 primary deployable；all/hard/proximity/source-support 仍是 blocker |
| Stage42-DF | all/hard/proximity repair search | 12/14 gates；test vs endpoint-linear all -0.67%；t50 -1.40%；hard -0.72%；easy +0.19% | validation-only threshold/proximity repair 没修好 primary gap；keep CQ/Stage37 floor |
| Stage42-DG | all/hard weighted-loss retraining | 13/15 gates；protected all +24.58%；t50 +22.02%；t100 raw +14.37%；hard +23.75%；easy -25.66%；delta vs AM 0 | 重新训练复现 AM，但没有超过 AM；单纯 loss weighting 不够 |
| Stage42-DH | proximity/occupancy-proxy weighted retraining | 15/16 gates；protected all +25.51%；t50 +22.14%；t100 raw +14.34%；hard +23.74%；easy -29.23%；delta vs AM all +0.93%、hard -0.01% | proximity/density weighting 有轻微 all gain，但没有修复 hard/failure primary blocker；不 promotion |

## 3. 路线复盘：试过什么，结果是什么

### 3.1 BPSG-MA / early 2.5D scaffold

做了什么：

- 建立 per-agent multi-agent trajectory world-state scaffold。
- 建立 causal baseline fallback。
- 建立 failure diagnostics、hard/failure bench、goal bench。
- 明确部署策略是 strongest causal baseline fallback + diagnostics。

结果：

- 成功成为稳定工程基座。
- 能跑、能审计、能 fallback。

失败 / 限制：

- 不是 true 3D。
- 不是 foundation。
- 不是大规模 representation world model。
- 主要解决可运行与可审计，不解决神经动力学主贡献。

### 3.2 Stage18/19 JEPA / WAM-style data

做了什么：

- Stage18 SAM-JEPA-2.5D representation pretraining。
- Stage19 WAM-style data registry：simulation、top-down、egocentric/video 分角色接入。
- 后续多次做 JEPA non-collapse、probe、downstream 检验。

结果：

```text
JEPA non-collapse = yes
downstream lift = not proven
deployable contribution = no
```

失败原因：

- latent variance 不等于 selector/failure/correction/t50 有 lift。
- JEPA 目标与部署目标错位。部署需要“何时切换、何时回退、easy 不伤、hard/failure 有收益”。
- scene/trajectory/interaction latent 没有稳定改善下游 heads。
- JEPA 不能包装成 latent generative rollout。

当前处理：

- JEPA 只保留为 auxiliary / diagnostic。
- 不能作为主贡献，也不能作为 Stage5C 或生成式 world model claim。

### 3.3 Stage20/21 数据采集与 SDD 转换

做了什么：

- 联网/本地数据发现和合法性审计。
- SDD 解压并转换为 per-video world-state shards。
- OpenTraj/外部数据进入后续 diagnostic / transfer 轨道。
- 明确 registry-only、download-failed、license-blocked 不算 converted。

结果：

- SDD 转换成功，成为 official pixel-space raw-frame benchmark。
- no-leakage、causal velocity、train/val/test split 被审计。

关键 SDD 数据事实：

- scenes = 8。
- videos = 60。
- tracks = 10300。
- world-state rows = 10616256。
- raw-frame t+50 samples = 10009005。
- raw-frame t+100 samples = 9497463。
- coordinate status = pixel-space。
- metric status = no verified homography / scale。

限制：

- SDD 不能写 metric。
- t+50/t+100 不能写 seconds-level。

### 3.4 Stage22/23/24：SDD benchmark 与 selector 失败

做了什么：

- SDD scene packs、lazy episodes、GoalBench、HardBench、BaselineFailureBench。
- strongest causal baseline 重新计算。
- true medium index 与 fast cache。
- validation-selected selector / failure predictor / JEPA probes。

关键成功：

- SDD official pixel-space benchmark 成立。
- fast cache 加速约 12.66x。
- true medium index 600,000 windows。
- no-leakage pass。
- selector oracle headroom 很大：约 46.2%。
- failure predictor 过 gate：AUROC 约 0.8715。

关键失败：

```text
Stage24 validation-selected selector:
  t+50 improvement = -43.3%
  easy degradation = 11.33%
```

失败原因：

- 直接 hard classification “哪个 baseline 最好”会过度切换。
- oracle best label 在低 margin 样本上噪声大。
- selector 学到了“切换”，但没学会“什么时候不要切”。
- easy 样本大量被错误切换。

结论：

- oracle headroom 大不代表训练 selector 成功。
- 需要 cost-aware / regret-aware / fallback-safe policy。

### 3.5 Stage25/26：cost-aware selector 修复 SDD

做了什么：

- 不再预测 one-hot best baseline。
- 对每个 candidate baseline 预测 expected FDE/risk。
- 加 confidence gate、gain margin、easy guard、harm guard。
- 使用 failure predictor 作为辅助。
- test 只最终评估一次。

结果：

| 指标 | 结果 |
| --- | ---: |
| SDD t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |

成功原因：

- 把任务从 hard label classification 改成 cost-aware decision。
- 明确“切换必须有足够收益，否则 fallback”。
- 修复了 low-margin 样本和 easy 样本过切换。

当前定位：

- Stage26 是 SDD 内部 best deployable baseline。
- 不能外推成 cross-domain 或 foundation success。

### 3.6 Stage31/32：external zero-shot / normalization / latent alignment 失败

做了什么：

- 建立 external feature store。
- 运行 external no-leakage 和 strongest baseline。
- 尝试 SDD -> external zero-shot。
- 尝试 normalization、CORAL、latent adapter、mixed-domain selector。

结果：

```text
Stage31 zero-shot M3W-LAS external transfer:
  all improvement = -92.67%
  t50 improvement = -278.57%

Stage32 external adapted selector:
  all/t50 improvement ~= 0
```

失败原因：

- SDD pixel-space 与 external dataset-local 坐标不兼容。
- horizon/track length 不匹配。
- scene/goal/interaction 缺失或定义不一致。
- agent type 不一致。
- scale/homography 未验证。
- latent distribution distance 变小不等于 decision target 变好。

结论：

- 普通 domain alignment 不够。
- 跨域要改成 coordinate-invariant / relative-error / train-only goals / selective transfer。

### 3.7 Stage33/34：external geometry 与 relative target，局部正但不可部署

做了什么：

- 坐标不变特征。
- relative-error baselines。
- external row geometry。
- train-only goals。
- scene packs / goal distance features。

结果：

- t+50 / hard 上出现局部正信号。
- all-test 仍为负，easy degradation 高。
- 不能部署。

失败原因：

- 外部目标、horizon、source 支持仍不足。
- easy 样本保护不够。
- 局部 t50/hard 提升不能覆盖 all/easy 风险。

结论：

- 不能把局部正信号包装成 deployable。
- 需要 selective transfer 和 hard/easy/failure 判别。

### 3.8 Stage35/36：selective transfer 有 all/hard，但 t+50 仍失败

Stage35 结果：

| 指标 | 结果 |
| --- | ---: |
| external all improvement | +12.13% |
| hard/failure improvement | +13.98% |
| easy degradation | 0.041% |
| t+50 improvement | 0.0 |

Stage36 发现：

- t+50 test rows = 16263。
- t+50 oracle headroom 约 22.98%。
- 所以不是没有可学空间。

失败原因：

- 一个 selector 混管 t10/t25/t50/t100，t50 objective 被 all-test objective 淹没。
- t50 没有足够 past-only history signal。
- held-out scene 缺 train-scene goal context。
- goal distance / horizon-specific switchability 不够。
- 保守 policy 对 t50 多数 fallback。

结论：

- 不能再只调 threshold。
- 要构建因果历史窗口和 scene-agnostic goal prototypes。

### 3.9 Stage37：external t+50 修复成功

做了什么：

- 构建 past-only history windows：K=8/16/32/64。
- 构建 scene-agnostic goal prototypes：
  - straight_continue
  - slow_stop
  - left_turn
  - right_turn
  - reverse_or_u_turn
  - group_follow
  - density_avoid
  - exit_like_direction_from_past_motion
- 构建 t50 failure/gain/harm predictors。
- 构建 t50 专用安全 selector。
- 使用 conformal safety 控制 easy degradation。
- 严禁 future endpoint 进入 input。

结果：

| 指标 | 结果 |
| --- | ---: |
| external all improvement | +13.48% |
| external t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

成功原因：

- t50 有专属 feature、专属 label、专属安全策略。
- history window 提供了过去运动形态。
- scene-agnostic goal prototypes 在 held-out scene 下提供了可迁移目标先验。
- gain/harm/easy guard 防止乱切。

当前定位：

- Stage37 是 external dataset-local raw-frame 下第一个可部署正迁移点。

### 3.10 Stage38：bounded correction 没有超过 Stage37

做了什么：

- 冻结 Stage37 policy。
- 尝试 bounded correction / dynamics head：

```text
prediction = selected_baseline + alpha * bounded_delta
```

结果：

- correction 没有稳定超过 Stage37。
- 不部署 correction。

失败原因：

- 直接 residual / correction 很容易伤 easy。
- hard/failure 有时改善，但 all/easy 安全不足。
- 受保护 policy 已经很强，普通 correction 没有足够稳定增益。

结论：

- 不训练普通 residual 作为部署模型。
- correction 必须被 Stage37 floor 保护，且过 gate 才能部署。

### 3.11 Stage39/40：真正 neural world dynamics 尝试，但未替代 Stage37

做了什么：

- Causal Temporal Transformer。
- JEPA auxiliary。
- JEPA + Transformer Hybrid。
- Stage37 teacher distillation。
- horizon-specific heads。
- hard/failure oversampling。
- t50-focused curriculum。
- 多 trial auto optimization。

Stage40 最好结果：

```text
best_stage40_neural = Stage40_causal_transformer_candidate_ranker
rows = 16000
all improvement = +13.20%
t50 improvement = +8.30%
hard/failure improvement = +15.24%
easy degradation = 0.068%
neural_exceeds_stage37 = false
deployment_decision = keep_stage37_selector
```

裸 neural without fallback：

```text
all improvement = -126.36%
t50 improvement = -292.10%
hard/failure improvement = -109.40%
easy degradation = 612.31%
```

失败原因：

- neural without fallback 不安全。
- Transformer/Hybrid 学会了部分 teacher/floor 机制，但没有稳定超过 Stage37。
- JEPA non-collapse 仍不等于 downstream lift。
- 神经模型很多时候是在复制 Stage37，而不是学到更强可部署 dynamics。

结论：

- 神经网络已经被训练和评估，但不能替代 Stage37。
- 当前神经贡献只能在 protected / teacher-floor 语境下报告。

### 3.12 Stage41/42：protected neural / full-waypoint / source-level evidence package

做了什么：

- composite-tail safe-switch bounded neural dynamics。
- full-waypoint prediction / row cache。
- source-level split。
- source-level full-waypoint eval。
- ablation：history、domain expert、baseline family、goal/scene、neighbor/interaction。
- safety floor necessity audit。
- paper claim evidence audit。
- runtime replay。
- evidence provenance。

当前主机制结论：

```text
dominant supported mechanism:
  baseline-family rollout context
  + causal history
  + guarded domain expert
  + conservative safety floor

not supported as independent main mechanisms:
  JEPA
  Transformer alone
  goal/scene context alone
  neighbor/interaction context alone
```

关键 Stage42 结果：

- Stage42-CI：context forensics 说明 baseline-family rollout context 是 dominant mechanism；history 是 core；domain expert 是 secondary；goal/scene、neighbor/interaction 不能作为主贡献。
- Stage42-CJ：goal/scene gated expert 没有超过 `baseline_family_control`。
- Stage42-CK：neighbor/interaction gated expert 没有超过 `baseline_family_control`。
- Stage42-CL：把 CJ/CK 负证据写入 paper package，避免 overclaim。
- Stage42-CO：common validation/test row alignment 完成，full-waypoint composer 可在 ETH_UCY t50/t100 使用。
- Stage42-CP：2000 bootstrap 证明 composer 对 endpoint-linear 有正 lower bound，但有 near-collision caveat。
- Stage42-CQ：proximity-aware guard 修复 near-collision caveat。
- Stage42-CR：明确 no-guard vs guarded 的 accuracy/safety Pareto。
- Stage42-CV：冻结 policy batch runtime replay 精确复现，25/25 gates。
- Stage42-CX：21 个 artifact provenance 审计通过。
- Stage42-CY：当前 Stage42 dirty caveat 中无 substantive metric 变化。

## 4. 最重要失败列表和根因

| 失败路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| JEPA as main contribution | non-collapse 但 downstream lift 不稳定 | 表征目标和部署目标错位 | 只作为 auxiliary / diagnostic |
| Stage24 hard-class selector | t50 -43.3%，easy degradation 11.33% | low-margin label noise，过度切换，无 harm/easy guard | Stage26 改为 expected-FDE / regret-aware / fallback-safe |
| SDD -> external zero-shot | all -92.67%，t50 -278.57% | 坐标、horizon、scene/goal、agent-type、scale 不兼容 | 改成 coordinate-invariant / relative / train-only goals / selective transfer |
| normalization / CORAL / latent adapter | 分布距离变小但预测没变好 | 分布对齐不是决策目标对齐 | 只保留 diagnostic |
| Stage35/36 t50 | all/hard 正，但 t50 = 0 | 缺 history window、goal prototype、t50-specific switchability | Stage37 修复 |
| ordinary residual / correction | hard 有时好，但 easy 容易受伤 | baseline already-good 样本多，残差无安全 gate | 不部署，除非 protected 且过 gate |
| ungated Transformer / Hybrid | 不安全，大幅负迁移 | 无 floor 输出会伤 easy 和 long horizon | 部署必须 Stage37 protected |
| goal/scene expert | 没超过 baseline-family control | 当前 scene/goal 特征对 deployment objective 增益不足 | 只写 auxiliary / negative evidence |
| neighbor/interaction expert | 没超过 baseline-family control | 当前 graph/context 不能提供稳定 incremental lift | 只写 auxiliary / negative evidence |
| t100 global claim | 多 source support 不足 | ETH_UCY/TrajNet 独立 source、terms、conversion、easy safety 未全闭环 | 只能 raw-frame diagnostic / blocker |
| metric/seconds claim | 未全局验证 | homography、scale、FPS/stride、license conversion 未闭环 | 禁止 metric / seconds-level claim |

## 5. 当前可以写的 claim

可以写：

```text
M3W 当前是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
Stage26 在 SDD pixel-space raw-frame 上形成 cost-aware deployable selector。
Stage37 在 external dataset-local raw-frame 上修复 t+50 并形成 deployable selector candidate。
Stage41/42 形成了 protected neural/full-waypoint/source-level evidence package。
Stage42 proximity-aware composer guard 在保持 all/t50/t100 raw-frame/hard-failure 正收益的同时修复 near-collision caveat。
当前最有效机制是 baseline-family rollout context + causal history + guarded domain expert + safety floor。
```

不能写：

```text
true 3D world model
foundation world model
global metric model
seconds-level long-horizon model
ungated neural dynamics success
JEPA is a generative world model
Stage5C executed
SMC ready/enabled
goal/scene or neighbor/interaction are independent main drivers
t100 global deployable success
```

## 6. 当前 best deployable 与可报告版本

### Safety-sensitive deployable

```text
policy:
  Stage42-CQ / CV proximity-aware composer runtime policy

evidence:
  Stage42-CQ 19/19 gates
  Stage42-CV 25/25 batch runtime replay gates

test vs endpoint-linear ADE:
  all +1.77%
  t50 +1.07%
  t100 raw diagnostic +3.48%
  hard/failure +1.93%
  easy degradation +0.25%

safety:
  near-collision@0.05 delta vs endpoint-linear = -0.06%
  no future endpoint / waypoint input
  no central velocity
  no test endpoint goals
  no test threshold tuning
```

### Accuracy-priority diagnostic

```text
policy:
  Stage42-CO/CP no-proximity-guard composer

test vs endpoint-linear ADE:
  all +3.02%
  t50 +1.50%
  t100 raw diagnostic +6.12%
  hard/failure +3.28%
  easy degradation +0.25%

caveat:
  near-collision@0.05 delta vs endpoint-linear = +0.34%
```

所以如果写部署/安全：

```text
用 proximity-aware guarded policy。
```

如果写 diagnostic upper-bound：

```text
可以报告 no-guard composer，但必须明确 near-collision caveat。
```

## 7. Git / provenance / reproducibility 当前状态

最近已完成并提交的有意义进展：

- `26c9345 Add Stage42 batch runtime policy replay`
  - Stage42-CV batch runtime replay。
  - 25/25 gates。
  - full tests 597 passed。
- `8f7bc3f Add Stage42 runtime replay paper refresh`
  - Stage42-CW paper/reproducibility/model-card refresh。
  - 25/25 gates。
  - full tests 600 passed。
- `1d2da72 Add Stage42 evidence provenance verifier`
  - Stage42-CX provenance verifier。
  - 21 artifacts audited，21 gates passed。
  - full tests 604 passed。
- `4b86379 Add Stage42 worktree caveat classifier`
  - Stage42-CY dirty caveat classifier。
  - 当时记录 Stage42 substantive dirty files = 0。
  - full tests 610 passed。
- `629c6fb Add Stage42 paper freeze candidate manifest`
  - Stage42-CZ paper-freeze candidate manifest。
  - 74 files hashed；14/14 gates。
- `98a3aff Refresh Stage42 clean paper freeze manifest`
  - Stage42-CZ clean manifest refresh。
  - `freeze_status = candidate_clean`；`final_immutable_release = true`。
- `08a8b2a Isolate Stage42 report tests from tracked artifacts`
  - 把 Stage42 report-writing tests 隔离到 `tmp_path`。
  - focused report tests 13 passed；full tests 615 passed。

当前 caveat：

- 仓库仍有历史遗留 dirty/untracked 数据和报告。
- Stage42-CY 当前分类结果：tracked dirty files = 8，Stage42 dirty = 0，Stage42 substantive dirty = 0；剩余 dirty 是历史 Stage17-19 outside-scope report drift，不应作为新 Stage42 证据引用。
- Stage42-CZ 当前结果：paper freeze candidate manifest clean，74 files hashed，14/14 gates，`final_immutable_release = true`。
- Stage42 report-test isolation 已修复 pytest 改写 tracked Stage42 report artifacts 的问题；这属于 reproducibility hygiene，不是新模型指标。

## 8. 下一步最值得做

1. 继续把 paper-freeze candidate 变成真正可交付 archive。
   - 当前 manifest 已 clean，但还需要外部 reviewer 可以按 manifest、runner、README 复现关键证据。
   - 不要把 clean manifest 写成 broader foundation / metric / seconds-level success。

2. 继续解决 t100 / source support。
   - ETH_UCY 本地 XML 技术路径有正信号，但 terms/conversion/source-CV 未闭环。
   - TrajNet 长 t100 source 仍不足。
   - 没有合法 independent source support 前，不写 global t100 deployable claim。

3. 若要推进“真正 world model”而不是 selector/composer：
   - 不要再堆普通 residual。
   - 需要让神经模型在不破坏 easy 的前提下，提供超过 baseline-family rollout context 的独立贡献。
   - 目前最可能方向是 source/horizon-aware protected dynamics + stronger causal sequence/waypoint supervision，而不是裸 JEPA 或裸 Transformer。

## 9. 最终当前结论

项目是否有实质进展：是。
是否训练/评估过神经世界模型：是。
神经模型是否无保护超过 Stage37：否。
当前 best deployable 是否仍受保护：是。
是否可以 claim true 3D：否。
是否可以 claim foundation：否。
是否可以 claim metric/seconds-level：否。
Stage5C 是否执行：否。
SMC 是否启用：否。

当前 best deployable：

```text
Stage42-CQ/CV proximity-aware guarded composer
under Stage37 / teacher safety floor
```

当前研究身份：

```text
M3W is a protected dataset-local/raw-frame 2.5D multi-agent world-state candidate with external positive-transfer evidence.
```

还不是：

```text
true 3D / foundation / global metric / seconds-level / ungated neural world dynamics.
```

<!-- STAGE42_DA_NEXT_ACTION_QUEUE:START -->
## Stage42-DA Next-Action Evidence Queue

- source: `fresh_synthesis_from_cached_verified_stage42_artifacts`
- role: convert current Stage42 paper gaps into prioritized executable next actions.
- gate: `15 / 15`; verdict `stage42_da_next_action_queue_pass`.
- top priority: `DA-1 Close legal/source support for ETH_UCY and TrajNet t100/t50 calibration`.
- user/external blockers remain explicit; no not_run item is counted complete.
- Current deployable claim remains protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_DA_NEXT_ACTION_QUEUE:END -->

<!-- STAGE42_DB_CONTEXT_RESCUE_DECISION:START -->
## Stage42-DB Context Rescue Decision Audit

- source: `fresh_synthesis_from_cached_verified_context_runs`
- role: decide whether existing goal/scene, neighbor/interaction, sequence, and graph context protocols should be repeated.
- gate: `13 / 13`; verdict `stage42_db_context_rescue_decision_pass`.
- decision: `stop_repeating_current_context_residual_or_gated_protocols`.
- best delta all/t50/hard vs baseline-family control: `-0.0230` / `-0.0831` / `-0.0262`.
- No safe positive context variant was found under the existing residual/gated protocols; next work must change target/model/data, not just rerun thresholds.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.
<!-- STAGE42_DB_CONTEXT_RESCUE_DECISION:END -->

<!-- STAGE42_DC_CONTEXT_SWITCHABILITY_GATE:START -->
## Stage42-DC Context Switchability / Gain-Harm Gate

- source: `fresh_run`
- role: change context supervision from waypoint residual to gain/harm switchability after Stage42-DB no-go.
- gate: `15 / 15`; verdict `stage42_dc_context_switchability_gate_pass`.
- selected candidate: `baseline_plus_knn_graph`; decision `context_switchability_not_supported`.
- delta vs baseline-family all/t50/hard/easy: `0.0004` / `-0.0001` / `0.0004` / `-0.0024`.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.
<!-- STAGE42_DC_CONTEXT_SWITCHABILITY_GATE:END -->

<!-- STAGE42_DD_SOURCE_SUPPORT_CLOSURE_AUDIT:START -->
## Stage42-DD Source Support Closure Audit

- source: `fresh_stage42_dd_source_support_closure_audit`
- role: close or explicitly block DA-1 legal/source/time-calibration support for ETH_UCY, TrajNet, and UCY.
- gate: `15 / 15`; verdict `stage42_dd_source_support_closure_audit_pass_open_blockers`.
- domains_not_closed: `['ETH_UCY', 'TrajNet', 'UCY']`.
- restricted ETH/UCY source-specific metric/time candidates exist, but global metric/seconds and global t100 deployable claims remain blocked.
- User/external action remains required before official converted/evaluated metric-time or t100 source-CV claims.
- Stage5C remains false; SMC remains false.
<!-- STAGE42_DD_SOURCE_SUPPORT_CLOSURE_AUDIT:END -->

<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:START -->
## Stage42-DE Full-Waypoint Deployment Gap Audit

- source: `fresh_stage42_de_full_waypoint_deployment_gap_audit`
- role: decide whether full-waypoint can be promoted from auxiliary/composer evidence to primary deployable world dynamics.
- gate: `17 / 17`; verdict `stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked`.
- decision: `protected_full_waypoint_composer_supported_deployment_promotion_blocked`.
- horizon_auxiliary_supported: `True`; guarded_composer_supported: `True`.
- primary deployable full-waypoint promotion: `False`.
- blockers: `['protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard', 'ungated_full_waypoint_easy_degradation_unsafe', 'source_legal_time_t100_closure_open', 'graph_group_interaction_has_proximity_caveat']`.
- Conclusion: keep Stage37/teacher or endpoint-linear safety floor; use guarded full-waypoint composer only as protected horizon/shape component until all/hard/proximity/source-support gaps are closed.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:END -->

<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:START -->
## Stage42-DF All-Hard / Proximity Full-Waypoint Repair

- source: `fresh_stage42_df_all_hard_proximity_full_waypoint_repair`
- role: validation-only repair search for the Stage42-DE all/hard/proximity full-waypoint deployment blocker.
- gate: `12 / 14`; verdict `stage42_df_all_hard_proximity_repair_partial`.
- test vs endpoint-linear: all `-0.67%`, t50 `-1.40%`, t100 raw `-0.66%`, hard `-0.72%`, easy `0.19%`.
- delta vs Stage42-CQ: all `-2.44%`, t50 `-2.46%`, t100 raw `-4.14%`, hard `-2.65%`, near@0.05 `-0.05%`.
- decision: `all_hard_proximity_repair_no_primary_promotion_keep_cq_guarded_composer`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:END -->

<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:START -->
## Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair

- source: `fresh_stage42_dg_full_waypoint_all_hard_loss_repair`
- role: actual retraining probe for all/hard/long-horizon weighted full-waypoint dynamics, following Stage42-DE/DF blockers.
- selected loss variant: `balanced` with lambda `100.0`.
- gate: `13 / 15`; verdict `stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `24.58%`, t50 `22.02%`, t100 raw `14.37%`, hard `23.75%`, easy `-25.66%`.
- delta vs Stage42-AM: all `0.00%`, t50 `0.00%`, t100 raw `0.00%`, hard `0.00%`, easy `0.00%`.
- decision: `weighted_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:END -->

<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:START -->
## Stage42-DH Full-Waypoint Proximity / Occupancy-Proxy Loss Repair

- source: `fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair`
- role: actual retraining probe for proximity/density/occupancy-proxy weighted full-waypoint dynamics after Stage42-DE/DF/DG blockers.
- selected candidate: `proximity_close_weighted` with `stage42_am_features` and lambda `100.0`.
- gate: `15 / 16`; verdict `stage42_dh_proximity_occupancy_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `25.51%`, t50 `22.14%`, t100 raw `14.34%`, hard `23.74%`, easy `-29.23%`.
- delta vs Stage42-AM: all `0.93%`, t50 `0.12%`, t100 raw `-0.03%`, hard `-0.01%`, easy `-3.57%`.
- decision: `proximity_occupancy_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:END -->
