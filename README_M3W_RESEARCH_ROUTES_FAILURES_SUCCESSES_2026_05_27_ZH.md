# M3W 目标内研究路线、失败原因、成功证据与当前质量总账

更新时间：2026-05-27

结果来源：`cached_verified_summary`，汇总既有 Stage18 到 Stage42-HV 报告、gate、model/data card、README 与 `research_state.json`。
本文件不是新训练、不是新评估、不是新数据转换；它是给当前长期目标使用的单文件中文总账。

## 0. 必须先说清楚的边界

当前 M3W 仍然不能被包装成下面这些东西：

- 不是 true 3D world model。
- 不是 large-scale foundation world model。
- 不是 global metric prediction model。
- 不是 seconds-level long-horizon predictor。
- 不是 ungated neural dynamics deployment。
- 不是 Stage5C latent generative rollout。
- 不是 SMC。

当前真实定位是：

```text
M3W = protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

更具体地说：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external 数据是 dataset-local / unverified weak-metric diagnostic，不可混成统一物理坐标。
- t+50 / t+100 是 raw-frame horizon；effective seconds、FPS/stride、homography、scale 未全局验证。
- future endpoint / future waypoint 只可作为 supervised label 或 evaluation label，不能作为 inference input。
- central velocity 不作为 official input。
- test endpoints 不用于构建 goals。
- self-audited / visual-prior / auto-silver labels 不是 human gold。
- 当前部署仍依赖 conservative safety floor / fallback；无保护 neural 仍不安全。

## 1. 一句话总结

这个目标内真正完成的是：从一个 SDD pixel-space selector scaffold，推进到一个 **受保护的外部 dataset-local raw-frame 2.5D 多智能体 world-state 候选系统**。  

成功主线不是 JEPA-only，也不是无保护 Transformer，而是：

```text
causal feature store
+ expected-FDE / gain-harm selector
+ conservative fallback
+ external causal history windows
+ scene-agnostic goal prototypes
+ protected full-waypoint / group-consistency policy
+ strict no-leakage / claim guard
```

当前最强可用结论是：

- SDD 内部：Stage26 cost-aware selector 是最强 deployable baseline。
- external t+50：Stage37 safe selector 是第一条真正可部署正迁移路线。
- protected neural / world-state：M3W-Neural v1 与 Stage42 full-waypoint / group-consistency 分支有可写论文的 protected 2.5D evidence。
- metric/time：仍未 ready；Stage42-HM/HN 已经把未来 restricted metric/time conversion path 做成 guarded queue，但当前 ready candidates = 0。
- t100 raw-frame：Stage42-HR/HS 增加了 validation-only t100 easy guard 与冻结复放证据，修复了 HQ 暴露的 t100 easy harm；Stage42-HT 又把该 frozen policy 做成可调用 runtime API；Stage42-HU 审计出真实 row-level batch replay 缺 `candidate/floor/selected` rollout arrays；Stage42-HV 随后从 HR rebuild path 生成本地 row-level replay cache，并用 HT runtime API 对 47,458 test rows 完成真实 batch replay，exact replay gate 28/28。t100 仍只能写 raw-frame diagnostic，不能写 seconds-level；HV cache 是 derived local artifact，不提交 GitHub。

## 2. 阶段路线总览

| 阶段 | 做了什么 | 结果 | 结论 |
| --- | --- | --- | --- |
| Stage18 | SAM-JEPA-2.5D representation pretraining | non-collapse，但 selector / failure / correction / official t+50 无 downstream lift | JEPA 不能作为主贡献 |
| Stage19 | WAM-style data registry、simulation / video / top-down 数据策略 | 建立数据路线，但不把 simulation/video 包装成 official trajectory success | 正确方向是补真实 top-down 数据 |
| Stage20-21 | Web acquisition + SDD/OpenTraj 数据准备；SDD 转 world-state shards | SDD 8 scenes / 60 videos / 10,300 tracks / 10.6M rows；no-leakage pass | SDD 成为 pixel raw-frame official benchmark |
| Stage22 | SDD scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、strong baselines | strongest causal baseline = damped_velocity；existing transfer 无 learned improvement | 需要 SDD-specific selector，而不是旧模型迁移 |
| Stage23 | 试图跑 medium，但因旧 NPZ I/O 慢只完成 quick-plus | quick-plus selector +2.66%，failure AUROC 0.6498，JEPA 无 lift | quick-plus 不能当 medium |
| Stage24 | 修 I/O，构建 fast cache 和 true medium index | I/O 加速约 12.66x；600k medium index；oracle headroom 46.2%；hard-class selector 失败 -43.3%；failure AUROC 0.8715 | 问题不在数据量，而在 selector 任务定义 |
| Stage25 | selector failure forensics、regret-aware/fallback 策略 | 定位 hard label / low-margin / easy over-switch 问题 | 必须从 hard classification 改成 cost-aware expected-FDE |
| Stage26 | feature-complete cost-aware selector | t+50 +14.58%，hard/failure +11.23%，easy degradation 1.81% | SDD best deployable 成立；不训练普通 residual |
| Stage31 | external feature store + zero-shot transfer | SDD->external all -92.67%，t50 -278.57%；adapted 0 | 外部坐标/scene/agent/horizon/domain gap 很大 |
| Stage32 | domain normalization / latent alignment | adapted selector 仍 0；mixed-domain 会伤 SDD easy | 普通 normalization 和 latent distance alignment 不够 |
| Stage33 | coordinate-invariant features / relative targets | 仍主要 safe fallback，external positive transfer 未成立 | 坐标不变还不够，缺逐行几何和 goals |
| Stage34 | external row geometry + train-only goals | t50 / hard 有局部正信号，但 all 负、easy 高 | 不能部署；需要 selective transfer |
| Stage35 | external data expansion + hard/easy/failure labels + selective transfer | all +12.13%，hard +13.98%，easy 0.041%，但 t50 = 0 | all/hard 过了，长时程 t50 是 blocker |
| Stage36 | t50-specific policy/forensics/curriculum | t50 仍 0；oracle headroom 22.98% | 特征和目标原型不足，不能只调 threshold |
| Stage37 | past-only history windows + scene-agnostic goal prototypes + conformal safe t50 selector | all +13.48%，t50 +8.46%，t50 CI [+7.69%, +9.15%]，hard +15.54%，easy 0.041%，gates 16/16 | external t50 deployable selector 修复成功 |
| Stage38 | freeze Stage37，尝试 bounded correction / multi-domain audit | correction 不超过 Stage37；UCY 正，ETH/TrajNet blocker | Stage37 保持 external best |
| Stage39 | Transformer / JEPA / Hybrid neural dynamics under Stage37 floor | 神经模型未超过 Stage37；JEPA non-collapse 但无 lift | 不部署 neural |
| Stage40 | 5-10 trial neural optimization | best neural 等于 Stage37 subset；without fallback 灾难性失败 | Stage37 仍 best deployable |
| Stage41 | M3W-Neural v1 package / protected composite-tail candidate | protected neural candidate 形成，bootstrap/multiseed/pure-UCY/full-waypoint bridge 证据 | 是 protected neural world-state candidate，不是 ungated dynamics |
| Stage42 | long research mode：source/domain/full-waypoint/group-consistency/claim guards/metric-time guards | 多个 protected full-waypoint / source-level / group-consistency 分支通过；sequence/graph context 主 claim 被关闭；metric/time conversion queue ready=0 | 形成 protected 2.5D paper package，但仍非 foundation / true 3D |
| Stage42-HR/HS/HT/HU/HV | t100 easy guard + frozen replay + runtime API + row-cache replay | HR：all +27.72%，t50 +26.99%，t100 raw +6.79%，hard +25.93%，t100 easy degradation -0.31%，gate 23/23；HS：policy hash/replay 固化，gate 27/27；HT：runtime API gate 19/19；HU：定位 blocker；HV：47,458 test rows row-cache replay，selected XY/ADE/switch/metric exact，gate 28/28 | 修复 t100 easy harm，并完成本地真实 row-level batch replay；但 t100 仍是 raw-frame diagnostic，不是 seconds-level，cache 不提交 |

## 3. 成功路线与关键证据

### 3.1 Stage26：SDD cost-aware selector 成功

Stage26 是 SDD 内部第一个真正过 gate 的学习组件。它的关键不是“更大模型”，而是把 selector 从 hard classification 改成 cost-aware / expected-FDE / failure-assisted selector。

关键结果：

| 指标 | 数值 |
| --- | ---: |
| feature store | built |
| selected model | `stage26_failure_assisted_selector` |
| t+50 improvement | +14.58% |
| hard/failure improvement | +11.23% |
| easy degradation | 1.81% |
| correction specialist | not trained |
| Stage5C | false |
| SMC | false |

成功原因：

- 输入特征来自过去：speed、acceleration、heading change、curvature、density、nearest neighbor、TTC、agent type、horizon、split type、goal distance、baseline rollout diagnostics。
- 不再预测单一 best baseline class，而是预测 expected FDE / risk。
- 有 conservative fallback：低置信、低收益、easy risk 高时回退 strongest baseline。
- Stage24 failure predictor 已证明可用，Stage26 用它辅助 selector。

### 3.2 Stage37：external t+50 safe selector 成功

Stage35 已经 all/hard 正，但 t+50 = 0；Stage36 证明 t+50 有 16,263 rows 和 22.98% oracle headroom，不是没空间，而是缺可学 past-only context。Stage37 用历史窗口和场景无关目标原型修复了这个 blocker。

关键结果：

| 指标 | 数值 |
| --- | ---: |
| external rows | 66,303 |
| all improvement | +13.48% |
| t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

成功原因：

- 构建了 K=8/16/32/64 的 past-only history windows。
- 使用 scene-agnostic goal prototypes：straight_continue、slow_stop、left_turn、right_turn、group_follow、density_avoid 等。
- t50 专用 switchability：failure predictor / gain predictor / harm predictor。
- conformal safety rule 用 validation 校准 easy degradation 和 harm_over_fallback。
- 不再把一个 selector 同时粗暴管 t10/t25/t50/t100。

### 3.3 M3W-Neural v1：protected neural candidate 成功，但不是 ungated neural

Stage39/40 的无保护 neural 失败后，Stage41/M3W-Neural v1 走了正确路线：不替代 Stage37 floor，而是在 Stage37/teacher safety floor 下做 safe-switch bounded neural dynamics。

关键 package 指标：

| 指标 | 数值 |
| --- | ---: |
| gates | 41 / 41 |
| all improvement vs Stage37 floor | +21.03% |
| t+50 improvement vs Stage37 floor | +13.65% |
| t+100 raw-frame diagnostic improvement | +14.69% |
| hard/failure improvement | +20.38% |
| easy degradation | 0.00% |
| positive external domains | 3 |
| bootstrap evidence | pass |
| multiseed replication | pass |
| pure UCY neural gate | pass |
| endpoint-to-full bridge | pass on ETH_UCY and TrajNet |

必须强调：

- 这是 protected neural dynamics candidate。
- 不是无保护 neural replacement。
- 不是 latent generative rollout。
- 不等于 Stage5C。
- 不等于 SMC。

### 3.4 Stage42 full-waypoint / group-consistency 分支成功

Stage42 中，最有价值的不是又跑一个大模型，而是把 endpoint selector 成功推进到 full-waypoint / source-level / group-consistency protected evidence。

代表性结果：

#### Stage42-W unified external full-waypoint package

| 指标 | weighted mean |
| --- | ---: |
| ADE all | +9.93% |
| ADE t50 | +9.40% |
| ADE t100 raw-frame diagnostic | +8.48% |
| ADE hard/failure | +10.49% |
| easy degradation | 0.24% |
| FDE t50 | +16.88% |

解释：

- ETH_UCY / TrajNet 来自 Stage42-S row-cache combo。
- UCY 来自 Stage42-V strict pure-UCY full-waypoint candidate。
- 不是单一 merged row-cache；是 policy package。
- 不能写成 metric/seconds/true-3D。

#### Stage42-DJ frozen group-consistency full-waypoint policy

| 指标 | 数值 |
| --- | ---: |
| gate | 22 / 22 |
| all | +24.72% |
| t50 | +22.36% |
| t100 raw-frame diagnostic | +14.35% |
| hard/failure | +23.89% |
| easy degradation | -25.63% |
| near@0.05 | 1.94% -> 1.38% |

解释：

- group-consistency 是目前最能写成 world-state / physical consistency repair 的贡献。
- 它是 protected source-level full-waypoint policy。
- 不是 ungated global full-waypoint replacement。

#### Stage42-EC contribution audit

支持的贡献：

- `explicit_group_consistency_full_waypoint`
- `dual_domain_raw_frame_support`
- `baseline_family_rollout_context`

明确 blocked 的贡献：

- scalar loss family primary claim。
- current sequence/graph residual context。
- goal/scene independent main claim。
- neighbor/interaction independent main claim。
- ungated global full-waypoint replacement。

#### Stage42-HR/HS/HT/HU/HV t100 easy guard, runtime policy and row-cache replay

Stage42-HQ 曾经把 group-consistency full-waypoint policy 推到更高 t100 raw-frame diagnostic gain，但暴露出一个关键安全问题：t100 easy slice 有正 easy degradation。Stage42-HR 没有继续追求更高 t100 数字，而是做 validation-only domain|t100 easy guard。

| 指标 | HQ before | HR after |
| --- | ---: | ---: |
| all | +32.89% | +27.72% |
| t50 | +26.99% | +26.99% |
| t100 raw diagnostic | +21.12% | +6.79% |
| hard/failure | +31.89% | +25.93% |
| easy degradation | -32.09% | -32.33% |
| t100 easy degradation | +2.56% | -0.31% |
| switch rate | 71.90% | 68.16% |

HR 的 validation-only 决策：

- `TrajNet|100`：validation easy degradation > 0，因此 t100 slice 回退 floor。
- `UCY|100`：validation all gain 为正且 easy degradation 为负，因此保留 candidate。

Stage42-HS 把 HR policy 冻结为轻量 artifact，Stage42-HT 再把它变成可调用 runtime API：

| 项目 | 结果 |
| --- | --- |
| policy artifact | `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json` |
| policy hash | `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e` |
| decision replay | exact |
| metric replay | exact |
| max metric diff | 0 |
| HS gates | 27 / 27 |
| HS verdict | `stage42_hs_t100_easy_guard_freeze_pass` |
| HT runtime gate | 19 / 19 |
| HT runtime verdict | `stage42_ht_t100_easy_guard_runtime_policy_pass` |

意义：

- 成功点：t100 easy harm 被修复，all/t50/hard 仍保持明显正收益。
- 代价：t100 raw diagnostic 从 +21.12% 降到 +6.79%，因为安全优先。
- 工程化进展：HT runtime policy 的规则是 `TrajNet|100` 回退 floor、`UCY|100` 保留 candidate、未知 t100 domain 默认回退 floor、非 t100 行不 guard。
- Stage42-HU blocker 已被 Stage42-HV 本机修复：HV 从 HR rebuild path 生成 `data/stage42_t100_runtime_replay_cache/stage42hv_t100_runtime_replay_test_cache.npz`，包含 row_id、domain、source_file、scene_id、frame_id、agent_id、horizon、candidate rollout、floor rollout、selected rollout、future labels eval-only、hard/failure/easy labels 等字段。
- Stage42-HV replay 结果：47,458 test rows；TrajNet 37,918、UCY 9,540；t100 rows 7,048；t100 easy rows 975；selected XY max diff 0、selected ADE max diff 0、switch mismatch 0、metric diff vs HR 全 0；gate 28/28。
- HV cache 是 derived row-level rollout data，不能提交 GitHub；提交的是代码、报告、cache hash 与轻量指标。
- 边界：这不是 seconds-level t100，也不是 metric long-horizon；只是 dataset-local/raw-frame t100 diagnostic safety guard。

## 4. 失败路线、失败原因、后续处置

### 4.1 Hard-class selector 失败

代表阶段：Stage24 / Stage25。  

表现：

- oracle headroom 很大：46.2%。
- 但 validation-selected hard-class selector t+50 improvement = -43.3%。
- easy degradation = 11.33%。

原因：

- oracle best baseline label 低 margin / 高噪声。
- best 和 second-best FDE 差距小，强制 one-hot 学习会过度切换。
- all/easy 样本被错误切换，伤害 easy。
- horizon / split / agent-type 混合后标签分布不稳定。

处置：

- Stage26 改成 expected-FDE / gain-harm / fallback-safe selector 后成功。

### 4.2 JEPA 主线失败

代表阶段：Stage18 / Stage22 / Stage23 / Stage24 / Stage39。  

表现：

- JEPA 多次 non-collapse。
- 但 selector / failure / goal / correction / t+50 downstream lift 没有稳定提升。
- Stage39 JEPA downstream failure AUROC lift 为负。

原因：

- non-collapse 只说明 representation 不塌缩，不说明对 downstream 有用。
- 当前 scene/image/goal/interaction signal 与 protected selector objective 没对齐。
- JEPA latent 可能加入噪声，不能直接提高 safe switching。

处置：

- JEPA 只能作为 auxiliary representation / diagnostic。
- 不能作为 M3W 主贡献。
- 不能说成 latent generative world model。

### 4.3 SDD -> external zero-shot 失败

代表阶段：Stage31。  

表现：

| 指标 | 数值 |
| --- | ---: |
| zero-shot all improvement | -92.67% |
| zero-shot t50 improvement | -278.57% |
| adapted all / t50 | 0.0 / 0.0 |

原因：

- SDD 是 pixel-space，external 是 dataset-local / weak-metric diagnostic。
- coordinate scale / horizon / scene / agent type 都不兼容。
- external scene/goal/interaction 信息缺失。
- latent cache 没有 scale calibration。

处置：

- Stage34 补 row geometry 和 train-only goals。
- Stage35 扩容 external 数据。
- Stage37 用 past-only history + goal prototypes 修复 t50。

### 4.4 普通 domain normalization / latent alignment 失败

代表阶段：Stage32 / Stage33。  

表现：

- per-scene zscore、velocity-scale、path-length、quantile normalization 都不能直接带来 deployable transfer。
- latent adapter 缩小 distribution distance，但不带来 predictive lift。

原因：

- distribution alignment 不等于任务增益。
- selector 需要知道“什么时候切换安全”，不是仅仅让 latent 均值方差接近。
- scene/goal/horizon/track length 的结构差异仍存在。

处置：

- 后续转向 row geometry、relative-error targets、hard/easy/failure labels、history windows、goal prototypes。

### 4.5 普通 residual / correction 失败

代表阶段：Stage22 / Stage24 / Stage38。  

表现：

- correction 在 hard 或 t50 上有时有局部 lift。
- 但 all/easy 不稳，不能部署。
- Stage38 bounded correction 仍不超过 Stage37。

原因：

- residual 直接改轨迹容易伤害 easy cases。
- selected baseline 已经很强，普通 residual 需要非常精准的 gain/harm 判断。
- correction 如果没有 Stage37-style safety gate，会扩大错误。

处置：

- correction specialist 只允许在 selector/failure/hard gates 已可靠时做。
- 当前部署仍保留 Stage37 / teacher floor。

### 4.6 Unprotected Transformer / Hybrid 失败

代表阶段：Stage39 / Stage40。  

表现：

- Stage39 Transformer / JEPA / Hybrid 没有超过 Stage37。
- Stage40 best protected neural 在同 subset 上等于 Stage37。
- neural without fallback 灾难性失败：all -126.36%，t50 -292.10%，easy degradation 612.31%。

原因：

- raw endpoint/FDE loss 不会自动学会 Stage37 的 safe-switch / gain / harm 机制。
- fallback gate 把不可靠 neural switch 吃掉，所以 with-fallback 不超过 Stage37。
- without-fallback 缺少 easy preservation。

处置：

- 神经网络只能在 safety floor 下报告。
- 后续 M3W-Neural v1 走 composite-tail safe-switch / bounded dynamics，而非无保护替代。

### 4.7 Scene/goal 或 neighbor/interaction 独立主 claim 失败

代表阶段：Stage42-AR / AS / EC。  

表现：

- sequence residual variants 全局 all/t50/hard 低于 baseline-family first stage。
- graph/context variants 全局也不超过 baseline-family。
- Stage42-EC 明确关闭 current sequence/graph residual context protocol。

原因：

- 目前主要有效信号来自 baseline-family rollout context + safety floor。
- scene/goal/neighbor context 在当前 residual target 下不能稳定转化成部署增益。
- 静态/图上下文容易过拟合或引入错误切换。

处置：

- 可作为 diagnostic / auxiliary。
- 不能写成独立主贡献。
- 当前 paper-ready claim 应放在 protected group-consistency full-waypoint dynamics。

### 4.8 Metric/time claim 被阻塞

代表阶段：Stage42-HL / HM / HN。  

表现：

- Stage42-HL claim guard：扫描 14 个 paper/README 文件，0 overclaim violation。
- Stage42-HM terms intake v2：source-level candidates 11，after-terms t50/t100 windows 14,457 / 7,129，但 ready now = 0。
- Stage42-HN guarded conversion queue v2：conversion queue count = 0，blocked candidates = 11。

原因：

- local parseability 不等于 legal permission。
- official terms / source identity / local path / allowed use / redistribution / derived data 都需要用户确认。
- restricted metric/time 必须先 guarded conversion、no-leakage、source-CV、final test。

处置：

- 当前只保留 future action queue。
- 不能写 metric/seconds claim。

## 5. 当前 best deployable 分层

| 场景 | 当前 best deployable / candidate | 是否可部署 | 注意 |
| --- | --- | ---: | --- |
| SDD pixel raw-frame | Stage26 cost-aware failure-assisted selector | 是 | SDD-only，不是 metric |
| External t+50 dataset-local raw-frame | Stage37 causal-history + goal-prototype safe selector | 是 | external deployable selector candidate |
| Protected neural world-state | M3W-Neural v1 composite-tail safe-switch bounded dynamics | 候选 | 必须在 Stage37/teacher floor 下 |
| Full-waypoint / group consistency | Stage42-DJ / W protected full-waypoint group-consistency family | 候选 | source-level protected evidence |
| t100 raw-frame diagnostic safety | Stage42-HR/HS/HT/HU/HV group-consistency t100 easy guard runtime policy + row-cache replay | 候选 | 修 t100 easy harm、提供 runtime API，并完成本机 row-level batch replay；不能写 seconds-level，cache 不提交 |
| JEPA-only | 无 | 否 | non-collapse 但无 downstream lift |
| Transformer/Hybrid ungated | 无 | 否 | 不安全，easy 会被破坏 |
| Metric/time restricted benchmark | 无 | 否 | ready candidates = 0 |
| Stage5C latent generative | 无 | 否 | 未执行 |
| SMC | 无 | 否 | 未启用 |

## 6. 当前质量判断

我会把当前质量分成三层说：

### 工程/实验质量

较强。已经有：

- SDD fast cache / medium index。
- external feature store。
- no-leakage audit。
- train/val/test split。
- bootstrap / multi-seed evidence。
- policy freeze / hash / replay。
- README / model card / data card / failure analysis。
- source/legal/claim guard。
- `.venv-pytorch` arm64 runtime 路线。

### 模型贡献质量

中上，但必须加限定词。  

可写的贡献不是“foundation world model”，而是：

```text
protected 2.5D external multi-agent world-state dynamics
with causal safe switching, full-waypoint group consistency,
and strict no-leakage raw-frame evaluation
```

强点：

- Stage26 / Stage37 / M3W-Neural v1 / Stage42 group-consistency 都有正证据。
- hard/failure 与 t50 不再是完全失败。
- easy preservation 被真正纳入 gate。

弱点：

- 仍依赖 safety floor。
- JEPA / Transformer 不能作为独立主贡献。
- scene/goal/interaction 独立 lift 不稳定。
- metric/time 未验证。
- external source diversity 仍有限。

### 论文候选质量

适合写成 **受保护 2.5D 多智能体 world-state dynamics paper package**，但还不能写成：

- true 3D world model paper。
- foundation world model paper。
- broad cross-dataset metric trajectory prediction paper。
- ungated neural dynamics paper。

如果投高水平 venue，当前最稳的 framing 是：

```text
M3W: Leakage-Safe Protected 2.5D Agent-Scene World-State Dynamics
for Top-Down Multi-Agent Trajectory Benchmarks
```

而不是：

```text
Large-Scale 3D Foundation World Model
```

## 7. 最有意义的结果清单

### SDD

- Stage22 建立 SDD pixel raw-frame official benchmark。
- strongest causal baseline：damped_velocity。
- SDD FDE raw-frame pixel-space：t10 5.7843，t25 12.9896，t50 29.4944，t100 60.5580。
- Stage24 建立 true medium index：cross_scene 200k/50k/50k，within_scene 200k/50k/50k，总 600k。
- Stage26 selector：t50 +14.58%，hard/failure +11.23%，easy degradation 1.81%。

### External

- Stage31 zero-shot 失败，明确 domain gap。
- Stage35 external 扩容：test 66,303 rows，t50 16,263，t100 10,008。
- Stage37 external deployable selector：all +13.48%，t50 +8.46%，hard +15.54%，easy 0.041%，CI 正。

### Neural / World-State

- Stage39/40 证明无保护 neural 不能部署。
- M3W-Neural v1 protected candidate：all +21.03%，t50 +13.65%，t100 raw +14.69%，hard +20.38%，easy 0。
- Stage42-DJ group-consistency full-waypoint：all +24.72%，t50 +22.36%，t100 raw +14.35%，hard +23.89%，near@0.05 改善。
- Stage42-W unified full-waypoint package：ETH_UCY、TrajNet、UCY 三域 policy package。
- Stage42-HR/HS/HT/HU/HV t100 easy guard：all +27.72%，t50 +26.99%，t100 raw diagnostic +6.79%，hard +25.93%，t100 easy degradation -0.31%，frozen replay exact，runtime policy gate 19/19；HU gate 17/17 定位缺 row arrays；HV 重建本地 row-level replay cache，47,458 rows exact replay，gate 28/28。

### Claim / Safety / Legal

- Stage42-HL：post-HK metric/time claim guard 0 violations。
- Stage42-HM：restricted metric/time terms intake v2，候选 11，但 ready=0。
- Stage42-HN：guarded conversion queue v2，明确拒绝 conversion while ready candidates=0。

## 8. 下一步最短路径

1. **不要再盲目加大 JEPA/Transformer。** 当前失败说明主 blocker 是 safe-switch / source diversity / metric-time legality，而不是纯模型容量。
2. **继续 group-consistency full-waypoint 路线。** 这是目前最接近 world-state dynamics 的正证据。
3. **补独立合法 external source。** 现在 broad source-level generalization 仍有限，Stage42-HM/HN 已列出 UCY/ETH_UCY source candidates，但需要用户确认 terms/source identity/path。
4. **做 restricted metric/time conversion only after terms。** 不能把 candidate windows 写成 metric result。
5. **如果继续神经动力学，必须保留 Stage37/teacher floor。** 目标是逐步减少 floor 依赖，而不是直接移除。
6. **论文主 claim 要保守。** 写 protected 2.5D world-state dynamics，不写 true 3D/foundation。

## 9. 最终结论

项目是否跑通：是，作为 protected 2.5D raw-frame/dataset-local multi-agent world-state candidate 跑通。  

是否 true 3D：否。  

是否 foundation：否。  

是否 metric/seconds：否。  

SDD best deployable：Stage26 cost-aware selector。  

external best selector：Stage37 causal-history / goal-prototype safe selector。  

当前最强 protected neural/world-state candidate：M3W-Neural v1 + Stage42 protected full-waypoint / group-consistency family。  

JEPA 是否主贡献：否。  

Transformer/Hybrid 是否可无保护部署：否。  

Stage5C 是否 ready：否。  

SMC 是否 ready：否。  

当前最诚实 verdict：

```text
M3W has become a strong protected 2.5D external multi-agent world-state candidate
with validated SDD and external raw-frame gains, but it is not yet a true 3D,
metric-time calibrated, foundation-scale, or ungated neural world model.
```
