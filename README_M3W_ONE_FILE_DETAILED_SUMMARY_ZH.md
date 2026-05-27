# M3W 单文件详细总结：做过什么、失败什么、成功什么、当前质量如何

更新时间：2026-05-27

工作目录：`/Users/yangyue/Downloads/World`

结果来源：

- `fresh_run`：本轮或对应 Stage 实际重新运行、重新训练、重新评估得到。
- `cached_verified`：读取已有结果，但已有 hash / schema / no-leakage / gate / replay / pytest 记录支撑。
- `diagnostic_only`：有分析价值，但不能当 deployable 或主 claim。

本文用途：把 M3W 长期目标内的关键尝试、路线、失败原因、成功证据、当前 best deployable、论文 claim 边界和下一步 blocker 集中写到一个 README。本文是总结文件，不是新训练结果，不把 cached 写成 fresh，不把失败包装成成功。

最新核验范围：本文已纳入 Stage42-FU module contribution ledger、Stage42-FV claim-boundary linter、Stage42-FW source-action consolidator、Stage42-DM reviewer replay package、Stage42-FX objective coverage audit、Stage42-FY horizon retry decision map、Stage42-GA live source/calibration recheck、Stage42-GB source terms prefill、Stage42-GC prefill-to-intake bridge、Stage42-GD calibration-hint-to-intake bridge、Stage42-GE conversion-capability-to-intake bridge。Stage42-DM 当前 gate 为 27/27；Stage42-FX gate 为 15/15；Stage42-FY gate 为 14/14；Stage42-GA/GB/GC/GD/GE 分别为 15/15、15/15、16/16、18/18、20/20。reviewer replay commands 已覆盖 runtime replay、module ledger、claim linter、source-action consolidator、provenance verifier 和 paper-freeze manifest。它不重新训练、不下载、不转换、不调 threshold。

## 本次请求版总览：做了什么、试了什么、什么失败、什么成功

你要的是一个能直接读的总账。下面是按长期目标归纳后的版本。

### A. 我实际推进过的路线

1. **数据与 benchmark 路线。**
   从早期 EWAP/ETH-UCY/OpenTraj 到 SDD，再到 UCY/TrajNet/ETH_UCY external source-level row cache，逐步建立了 world-state rows、scene packs、lazy episode/index、HardBench、BaselineFailureBench、GoalBench、source-level split、no-leakage audit、bootstrap/replay package。SDD 被固化为 pixel raw-frame official benchmark；external 仍是 dataset-local / unverified weak-metric diagnostic。

2. **强因果 baseline 与 fallback 路线。**
   系统实现并比较 constant position、causal constant velocity、damped velocity、constant acceleration、turn-rate、scene-clamped、goal/prototype-directed 等 baseline。后续所有 learned selector / neural / correction 都必须和 strongest causal baseline、Stage26、Stage37 floor 比。这条路线最终成为最稳定的安全骨架。

3. **Selector 路线。**
   先试 hard-class selector，失败后改为 expected-FDE / regret-aware / confidence-gated / easy-safe / fallback-safe selector。Stage26 在 SDD 上成功，Stage37 在 external t+50 上成功，Stage42-FH/FI 在 source/domain protected policy 上进一步成功。

4. **JEPA 表征路线。**
   多轮训练 trajectory-only、scene/trajectory、interaction-aware JEPA。多数阶段 latent non-collapse，但 selector/failure/goal/t50/correction 没有稳定 downstream lift。因此 JEPA 被降级为 auxiliary/diagnostic，不作为主 claim，不写成 latent generative world model。

5. **Transformer / Hybrid neural dynamics 路线。**
   训练过 Transformer-only、JEPA+Transformer hybrid、causal temporal Transformer、protected neural dynamics、full-waypoint sequence dynamics。无保护 neural 不安全；protected neural/full-waypoint 在 Stage37/teacher floor 下有证据，但不能说成独立 ungated neural world model。

6. **外部跨域迁移路线。**
   SDD->external zero-shot 大失败后，逐步引入 coordinate-invariant features、relative targets、external row geometry、train-only goals、scene-agnostic goal prototypes、past-only history windows、hard/easy/failure labels、selective transfer、conformal safety。Stage37 修复 external t+50，Stage42 则继续推进 source/domain/full-waypoint/group-consistency。

7. **安全/物理有效性路线。**
   持续跟踪 easy degradation、harm over fallback、near-collision@0.05、jagged-rate、group consistency。bounded residual/correction 和 unprotected neural 只要伤 easy 就不部署；Stage42-CQ/CR/FE 证明 proximity/safety guard 是必要的。

8. **论文证据与 claim 边界路线。**
   建立 exact replay、policy hash、schema hash、bootstrap CI、claim-boundary linter、module contribution ledger、paper-freeze manifest、reviewer replay package、source action consolidator。Stage42-FU/FV/FW/DM/FX/FY 和 GA-GD 的作用是防止把 diagnostic/cache/not-run 写成成功。

### B. 明确失败的路线和原因

| 路线 | 失败表现 | 主要原因 | 当前处理 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 t50 improvement 约 -43.3%，easy degradation 约 11.33% | low-margin oracle label 被硬分类放大；easy cases 被错误切换 | 改为 expected-FDE/regret-aware/fallback-safe selector |
| JEPA 主线 | 多次 non-collapse，但 downstream heads 无稳定 lift | representation objective 和部署目标错位；latent variance 不等于 gain/harm/easy-safety 信号 | 只保留 auxiliary/diagnostic |
| SDD->external zero-shot | all 约 -92.67%，t50 约 -278.57% | 坐标、scale、horizon、agent type、scene/goal/context 不兼容 | 改为 external-specific row geometry + relative target + history/prototype |
| latent adapter / CORAL | latent gap 变小但预测无提升 | 分布对齐不等于任务/收益/风险对齐 | 不作为成功 claim |
| bounded residual/correction | 未稳定超过 Stage37，容易伤 easy | 直接改轨迹风险高，selected baseline 已强 | 不部署 correction |
| unprotected Transformer/Hybrid | neural without fallback 灾难性或不安全 | 数据仍是 dataset-local/raw-frame，metric/scene grounding 不足 | 只允许 protected neural evidence |
| scalar proximity/occupancy | 有局部提升但不稳定超过 group-consistency / safety floor | scalar loss 表达不了完整群体时空约束 | 采用 explicit source/frame/horizon group-consistency |
| temporal/waypoint repel / Pareto repair | accuracy 和 proximity 常互相牺牲 | post-hoc 几何修复容易牺牲 ADE/hard | 用 constrained safety fallback |
| uniform horizon robustness | TrajNet|100、UCY|100 仍 weak | low-margin ambiguity、source support 稀疏、h100 long-horizon context 不足 | Stage42-FY 决定暂停同特征重试，转 source/legal/guarded conversion |
| source/legal/calibration | conversion_ready 仍为 0 | terms/path/source identity/calibration 仍需用户确认 | Stage42-GB/GC/GD 只做 prefill/hints，不声称 permission |

### C. 明确成功的路线和证据

| 成功点 | 关键证据 | 结论 |
| --- | --- | --- |
| Stage26 SDD cost-aware selector | t+50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 +1.81% | SDD pixel raw-frame best deployable |
| Stage37 external t50 repair | all +13.48%，t50 +8.46%，t50 CI [+7.69%, +9.15%]，hard/failure +15.54%，easy 0.041%，gate 16/16 | external selector-level deployable success |
| Stage42-CO/CP bridge-shape composer | all +3.02%，t50 +1.50%，t100 raw +6.12%，hard +3.28%，2000-bootstrap positive | protected full-waypoint auxiliary evidence |
| Stage42-CQ proximity guard | all +1.77%，t50 +1.07%，t100 raw +3.48%，hard +1.93%，near@0.05 不劣于 endpoint-linear | safety-sensitive protected composer |
| Stage42-DL/DM reviewer replay | rows 47,458，switch exact match true，all/t50/t100raw/hard 约 +24.72% / +22.36% / +14.35% / +23.89%，near@0.05 1.94% -> 1.38% | runtime replay 证明 policy 可复放 |
| Stage42-FE constrained FC/safety composer | all/t50/hard 26.41% / 23.15% / 24.81%，near@0.05 1.32%，gate 19/19 | 把 FC 精度和 DI safety floor 组合成功 |
| Stage42-FH UCY-supported FE composer | all/t50/t100raw/hard 34.98% / 28.97% / 20.57% / 33.10%，TrajNet 与 UCY 都 positive-safe，gate 20/20 | 从 TrajNet-only robust 推进到 dual-domain positive-safe |
| Stage42-FI frozen replay | policy hash 固化，exact replay diff 0，2000-bootstrap CI low all/t50/t100raw/hard 34.62% / 28.46% / 19.96% / 32.73%，gate 25/25 | 不是 test-tuned 偶然输出 |
| Stage42-FU module ledger | main claim 只允许 history、domain expert、safe switch、teacher floor、group-consistency full-waypoint | 防止把 JEPA/Transformer/scene/neighbor 过度写成主贡献 |
| Stage42-FV/FW/FX/FY | claim linter、source action consolidator、objective coverage、horizon retry map 均通过 | 论文/项目边界更稳，不继续盲目重试弱 horizon |
| Stage42-GA-GD | source/calibration recheck、source terms prefill、intake bridge、calibration hint bridge 均通过；conversion_ready 仍为 0 | 把下一步 user-confirmed legal/source/calibration intake 准备好，但不越权转换 |

### D. 当前质量判断

当前最诚实定位：

```text
M3W 是 protected dataset-local / raw-frame 2.5D multi-agent world-state candidate。
它有 SDD、external t50、source/domain protected policy、runtime replay、bootstrap、no-leakage 和 claim-boundary 证据。
它不是 true 3D，不是 foundation，不是 metric predictor，不是 seconds-level long-horizon predictor，也不是 ungated neural dynamics deployable model。
```

当前 best deployable 分层：

| 用途 | 当前最好结果 | 是否可部署 |
| --- | --- | --- |
| SDD pixel raw-frame | Stage26 cost-aware selector | 是，但仅限 SDD pixel/raw-frame claim |
| external t50 selector | Stage37 causal-history + goal-prototype safe selector | 是，external dataset-local/raw-frame claim |
| source/domain protected full-waypoint | Stage42-FH/FI frozen protected policy family | 是，protected source/domain evidence；不允许 uniform horizon overclaim |
| reviewer/runtime replay | Stage42-DM / DL runtime policy | 是，复放/审稿证据 |
| neural dynamics | M3W-Neural v1 protected family | 只在 Stage37/teacher floor 保护下报告；不部署 ungated neural |
| h100 / uniform horizon | Stage42-FY 后仍 blocked | 不可作为成功 claim |
| legal/source conversion | Stage42-GA-GD prefill/hints ready | 不是 permission，不是 converted data |

### E. 下一步最短路径

1. **先关掉 source/legal/calibration blocker。**
   用 Stage42-GB/GC/GD 生成的 intake prefill/hints，让用户逐项确认 terms、allowed use、local path、source identity、calibration evidence。没有这个确认，conversion_ready 必须继续是 0。

2. **只对 legally ready 的 source 做 guarded conversion。**
   不能把 registry、prefill、hint 当 converted/evaluated data。转换后必须重新跑 no-leakage、source-CV、baseline、policy replay。

3. **再重启 h100 / uniform horizon 修复。**
   TrajNet|100 和 UCY|100 的问题不是继续调同一套 threshold，而是需要 source support、long-horizon context、terms-confirmed source-CV 和更强 h100 features。

4. **如果继续 neural world dynamics，必须学 safety-relevant targets。**
   不要训练普通 residual；应训练 gain/harm、group-consistency、source/horizon-aware switchability、full-waypoint consistency，并保留 Stage37/teacher floor。

## 给用户的直接总结

你问“在这个目标内我做了什么、尝试了什么路线、哪些失败了、原因是什么、哪些成功了”。最短回答如下：

1. **真正成功的主线不是无保护大模型，而是 protected selector / safe-switch world-state policy。**
   最早的强因果基线和 fallback 体系最后变成 Stage26、Stage37、Stage42 policy family 的核心。它解决的是“什么时候允许切换、什么时候必须回退”的部署问题。

2. **SDD 内部成功：Stage26 cost-aware selector。**
   SDD 是 pixel-space raw-frame benchmark。Stage26 在 SDD 上达到 t+50 约 +14.58%、hard/failure 约 +11.23%、easy degradation 约 +1.81%。这证明 cost-aware / regret-aware / fallback-safe selector 比 hard classification selector 更靠谱。

3. **External t+50 成功：Stage37 causal history + scene-agnostic goal prototype。**
   Stage31-36 的外部迁移多次失败后，Stage37 用 past-only history window、scene-agnostic goal prototypes、t50 switchability/gain/harm 和 conformal safety 修复了 external t50：all +13.48%、t50 +8.46%、t50 bootstrap CI [+7.69%, +9.15%]、hard/failure +15.54%、easy degradation 0.041%、gate 16/16。

4. **更强的 source/domain protected evidence：Stage42-FH/FI。**
   Stage42-FH/FI 的 frozen protected policy family 在 source/domain 层面更强：all/t50/t100 raw/hard 约 +34.98% / +28.97% / +20.57% / +33.10%；FI exact replay diff 为 0；2000-bootstrap CI low all/t50/t100raw/hard 为 34.62% / 28.46% / 19.96% / 32.73%。这说明 frozen policy 不是一次性偶然输出。

5. **神经网络路线有贡献但不能夸大。**
   M3W-Neural v1 / Stage41/42 证明 protected neural / full-waypoint / group-consistency 组件可以在 teacher/Stage37 floor 保护下提供 evidence；但 JEPA、Transformer、scene/goal、neighbor/interaction 不能写成独立主贡献。Stage42-FU ledger 只允许 history、domain expert、safe switch、teacher floor、group-consistency full-waypoint 作为主 claim。

6. **主要失败路线很清楚：**
   hard-class selector 伤 easy；JEPA non-collapse 但无稳定 downstream lift；unprotected Transformer/Hybrid 不可部署；external zero-shot 因坐标/尺度/horizon/scene context mismatch 崩；latent adapter 缩小分布距离但不带来 predictive lift；bounded residual/correction 不安全；uniform horizon robustness 仍被 TrajNet|100 和 UCY|100 卡住。

7. **当前质量判断：**
   这是一个有强 replay、bootstrap、no-leakage、claim-linter 支撑的 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。它有投稿候选证据链，但不是 true 3D、不是 foundation、不是 metric predictor、不是 seconds-level long-horizon predictor，也没有执行 Stage5C 或 SMC。

8. **下一步不是继续重复同特征模型重试。**
   Stage42-FY 已明确：TrajNet|100 与 UCY|100 的弱 horizon 不能继续靠相同特征/相同 threshold 反复训练修复。下一步优先是合法 source support / UCY terms / guarded conversion / source-CV，然后再重启 h100/horizon robustness。

## 0. 最短结论

M3W 已经从早期的 SDD-only 2.5D trajectory scaffold，推进为一个有 SDD 与 external top-down pedestrian 数据证据的 **protected dataset-local / raw-frame 2.5D multi-agent world-state candidate**。

但它仍然不是：

- true 3D world model
- large-scale foundation world model
- global metric / meter-level predictor
- not seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative rollout
- not SMC-ready model

当前最强部署分层：

| 场景 | 当前最好结果 | 证据状态 |
| --- | --- | --- |
| SDD pixel raw-frame | Stage26 cost-aware selector | t+50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 +1.81%。 |
| External t+50 transfer | Stage37 causal-history + goal-prototype safe selector | all +13.48%，t50 +8.46%，t50 bootstrap CI [+7.69%, +9.15%]，hard/failure +15.54%，easy degradation 0.041%，gate 16/16。 |
| Protected neural / world-state candidate | M3W-Neural v1 composite-tail safe-switch under Stage37/teacher floor | all +21.03%，t50 +13.65%，t100 raw diagnostic +14.69%，hard/failure +20.38%，easy degradation 0；仍是 protected，不是 ungated neural deployment。 |
| Source/domain robust protected policy | Stage42-FH/FI frozen policy family | all/t50/t100raw/hard 约 34.98% / 28.97% / 20.57% / 33.10%；FI exact replay diff 0；CI low 34.62% / 28.46% / 19.96% / 32.73%；TrajNet 和 UCY domain positive-safe。 |
| Module claim ledger | Stage42-FU | gate 14/14；允许主 claim: history、domain expert、safe switch、teacher floor、group-consistency full-waypoint；阻止主 claim: JEPA、Transformer、scene/goal、neighbor/interaction。 |
| Reviewer replay package | Stage42-DM | gate 27/27；runtime rows 47,458；switch exact match true；group runtime all/t50/t100raw/hard improvement 约 24.72% / 22.36% / 14.35% / 23.89%；near@0.05 从 1.94% 降到 1.38%。 |

一句话：**当前最好成果是“受安全 floor 保护的 dataset-local/raw-frame 2.5D 多智能体 world-state candidate”，不是 true 3D 或 foundation world model。**

## 1. 已尝试的主路线

### 1.1 强因果基线与 fallback 路线

做了什么：

- 构建 constant position、constant velocity causal finite difference、damped velocity、constant acceleration、turn-rate、scene-clamped、goal-directed 等 baseline。
- 每个 learned selector / neural model 都必须和 strongest causal baseline、Stage26、Stage37 floor 比。
- 建立 oracle headroom、selector regret、easy/hard/failure 切片、harm over fallback。

结果：

- 成功，是当前最可靠的骨架。
- Stage26、Stage37、Stage42 protected policies 都源自这条路线。

原因：

- 真实轨迹预测里强 causal baseline 非常强。
- 学习模型如果在 easy cases 上乱切换，整体就不可部署。
- fallback floor 把“模型偶尔有用”变成“只在有把握时切换”。

### 1.2 SDD official pixel-space benchmark 路线

做了什么：

- 下载并转换 Stanford Drone Dataset。
- 构建 SDD world-state shards、scene packs、lazy episodes。
- 建立 GoalBench、HardBench、BaselineFailureBench、strongest causal baseline。
- 训练 selector、failure predictor、JEPA heads。

成功：

- Stage26 cost-aware selector 成为 SDD best deployable。
- 关键指标：t+50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 +1.81%。

边界：

- SDD 是 pixel-space，不是 metric。
- t+50/t+100 是 raw annotation-frame horizon，不是 seconds-level。
- homography、scale、effective seconds 没有全局验证。

### 1.3 hard-class selector 路线

做了什么：

- 早期 selector 直接预测“哪个 baseline 最好”。

失败：

- Stage24 hard-class selector t+50 improvement 约 -43.3%。
- easy degradation 约 11.33%。

原因：

- oracle best baseline label 经常 low-margin。
- best 和 second-best 差距很小，hard label 噪声大。
- selector 过度切换 easy cases。

修复：

- 改成 expected-FDE / regret-aware / cost-aware selector。
- 加 confidence gate、predicted gain margin、easy guard、fallback strongest baseline。

### 1.4 JEPA 表征路线

做了什么：

- Stage18/19/22/23/24 与 M3W 期间多轮训练 JEPA-only、trajectory JEPA、scene/trajectory JEPA、interaction-aware JEPA。
- 检查 latent variance / non-collapse。
- 训练 selector/failure/goal/hard probe。

失败：

- JEPA 多次 non-collapse，但没有稳定 downstream lift。
- failure predictor、selector、t50、correction 没有稳定改善。

原因：

- non-collapse 不等于对下游决策有用。
- 表征目标和部署目标错位。
- scene/video/raw frame grounding 与 metric/time calibration 不足。
- latent distribution alignment 不等于 gain/harm target alignment。

当前结论：

- JEPA 只能作为 auxiliary / diagnostic。
- 不能写成 latent generative world model。
- 不能作为主贡献 claim。

### 1.5 Transformer / Hybrid neural dynamics 路线

做了什么：

- 训练 Transformer-only、JEPA+Transformer hybrid、Causal Transformer with history/neighbor tokens、protected neural dynamics、full-waypoint sequence dynamics。
- 输出 trajectory、failure/gain/harm、selector、interaction、occupancy、physical validity proxy。

失败/部分成功：

- Stage39/40 神经模型没有超过 Stage37，部署仍保持 Stage37 selector。
- neural without fallback 往往灾难性失败或伤 easy。
- Stage41/42 的 protected neural/full-waypoint evidence 有明显贡献，但仍依赖 Stage37/teacher safety floor。

原因：

- Stage37 floor 已经强。
- 无保护 residual 或 endpoint dynamics 容易伤 easy。
- 数据是 dataset-local/raw-frame，不是统一物理世界坐标。
- t100 长时程仍受 source support、horizon ambiguity、track length 限制。

当前结论：

- 可以写 protected neural/world-state candidate。
- 不能写 ungated neural dynamics 已可部署。
- Transformer 独立主贡献尚未被 Stage42-FU ledger 允许。

### 1.6 External zero-shot / domain alignment 路线

做了什么：

- 将 SDD / M3W-LAS 直接迁移到 OpenTraj、ETH-UCY、TrajNet、UCY。
- 尝试 zscore、velocity/path normalization、relative target、CORAL、linear latent adapter、domain-conditioned selector。

失败：

- Stage31 zero-shot all improvement 约 -92.67%，t50 约 -278.57%。
- adapted selector 初期约 0 improvement。

原因：

- SDD pixel 与 external dataset-local 坐标不兼容。
- 外部 scene/goal/interaction 缺失。
- agent type 与 horizon 分布不匹配。
- latent adapter 缩小分布距离但没有带来 predictive lift。

修复：

- 补 external row geometry、train-only goals、relative-error targets、coordinate-invariant features。
- 后续转向 selective transfer 和 t50-specific history/prototype features。

### 1.7 External selective transfer 路线

做了什么：

- 构建 external hard/easy/failure labels。
- 训练 hard detector、failure predictor、gain predictor、harm predictor。
- 只在 hard/failure probability 高、predicted gain 高、harm 低时切换。

中间成功：

- Stage35 all +12.13%，hard/failure +13.98%，easy degradation 0.041%。

失败：

- Stage35 t+50 improvement = 0.0，不可部署。

原因：

- all-test objective 淹没 t50。
- 长时程缺完整 past-only history window。
- held-out scene 缺 train-scene goals。
- policy 对 t50 切换太保守。

### 1.8 Stage37 causal history + scene-agnostic goal prototypes

做了什么：

- 构建 K=8/16/32/64 past-only history window。
- 构建 scene-agnostic goal prototypes：straight、slow-stop、left/right turn、group-follow、density-avoid、exit-like direction。
- 训练 t50 switchability/gain/harm predictors。
- 用 conformal / validation safety 控制 easy degradation。

成功：

```text
all improvement: +13.48%
t+50 improvement: +8.46%
t+50 bootstrap CI: [+7.69%, +9.15%]
hard/failure improvement: +15.54%
easy degradation: 0.041%
gates: 16 / 16
verdict: stage37_t50_transfer_repaired_deployable
```

意义：

- 修复 external t50 gate。
- 是当前 external selector-level best deployable。

边界：

- 仍是 dataset-local raw-frame。
- 不是 metric/seconds-level。
- 不是 true 3D 或 foundation。

### 1.9 Bounded correction / residual 路线

做了什么：

- 训练 `prediction = selected_baseline + alpha * bounded_delta`。
- 尝试 linear/ridge/small MLP/horizon-specific/hard-only/t50-only correction。

失败：

- Stage38 correction 没有安全超过 Stage37。
- residual without enough gating 容易伤 easy。

原因：

- correction 直接改轨迹，风险比 selector 更高。
- Stage37 floor 已经捕获大部分安全收益。
- 如果 all/t50/hard 没同时提升并 easy<=2%，不能部署。

当前结论：

- correction 不部署。
- 继续保留 Stage37 / teacher floor。

### 1.10 Full-waypoint / source-level / group-consistency 路线

做了什么：

- 从 endpoint bridge 走向 full-waypoint sequence、source-level full-waypoint evaluation、group-consistency、proximity guard、paper evidence package。
- 训练/评估 static-gated full-waypoint、horizon static gate、row-gain static gate、gain/harm selector、t50 selector、source-level group consistency、proximity/occupancy objectives。

关键成功：

- Stage42-CO common-validation composer：all +3.02%，t50 +1.50%，t100 raw +6.12%，hard +3.28%。
- Stage42-CQ proximity-aware guard：all +1.77%，t50 +1.07%，t100 raw +3.48%，hard +1.93%，near@0.05 不劣于 endpoint-linear。
- Stage42-DL group-consistency runtime：all/t50/t100raw/hard 约 +24.72% / +22.36% / +14.35% / +23.89%，near@0.05 从 1.94% 降到 1.38%。
- Stage42-FE constrained FC/safety composer：all/t50/hard 26.41% / 23.15% / 24.81%，near@0.05 1.32%。
- Stage42-FH UCY-supported composer：all/t50/t100raw/hard 34.98% / 28.97% / 20.57% / 33.10%，TrajNet 和 UCY 都 positive-safe。
- Stage42-FI frozen replay：exact replay diff 0，2000-bootstrap CI low all/t50/t100raw/hard 34.62% / 28.46% / 19.96% / 32.73%，gate 25/25。
- Stage42-DM reviewer replay package：把 runtime replay、module ledger、claim linter、source-action consolidator、provenance 和 paper-freeze manifest 串成最小复核路径；gate 27/27，runtime rows 47,458，switch exact match true。

剩余失败：

- Stage42-FJ/FK/FL/FM/FN/FO/FP 表明 uniform horizon robustness 仍未解决。
- TrajNet|100 与 UCY|100 weak slices 仍存在。

原因：

- h100 source support 不足。
- oracle label low-margin ambiguity 很高。
- validation-to-test source-family shift。
- 单/稀疏 validation source support。
- current history/prototype/rollout gain-harm features 对剩余 h100 weak slices 不够。

## 2. 当前允许的 claim 与禁止的 claim

允许写：

- M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
- SDD 上 Stage26 cost-aware selector 是 best deployable。
- External t50 transfer 被 Stage37 修复。
- Stage42-FH/FI policy family 有 dual-domain/source positive-safe evidence。
- History、domain expert、safe switch、teacher floor、group-consistency full-waypoint 是当前可写主贡献。
- Full-waypoint shape 和 endpoint bridge 是支持组件。

不允许写：

- true 3D world model。
- foundation world model。
- global metric predictor。
- seconds-level t50/t100 predictor。
- JEPA downstream main contribution。
- Transformer independent main contribution。
- scene/goal independent main contribution。
- neighbor/interaction independent main contribution。
- ungated neural dynamics deployable。
- Stage5C 已执行。
- SMC 已启用。

## 3. 模块贡献 ledger 结论

Stage42-FU 生成了模块贡献总账，gate 14/14。

主 claim 允许：

- `history`
- `domain_expert`
- `safe_switch`
- `teacher_floor`
- `group_consistency_full_waypoint`

支持/边界组件：

- `full_waypoint_shape`
- `endpoint_bridge`

暂不允许主 claim：

- `scene_goal`
- `neighbor_interaction`
- `JEPA`
- `Transformer`

解释：

- history 和 domain expert 在 retrained/source-level evidence 中有正贡献。
- safe switch 和 teacher floor 是 deployability 必要条件。
- group-consistency full-waypoint 有 source-level bootstrap-backed positive-safe evidence。
- scene/goal、neighbor/interaction 有混合或弱证据，不能作为独立主贡献。
- JEPA non-collapse 但 downstream lift 不稳。
- Transformer 当前 proxy evidence 不足以写独立主贡献。

## 4. 为什么很多路线失败

### 4.1 为什么 JEPA 没成为主线

JEPA 学到了非塌缩 latent，但 latent 没有稳定改善部署目标。世界模型不是只要 latent variance 非零就成功；必须能改善 selector/failure/t50/hard/failure，并且不能伤 easy。当前 JEPA 没做到。

### 4.2 为什么普通 neural dynamics 没替代 Stage37

无保护 neural 会在 easy cases 上乱改轨迹；有保护 neural 又经常被 Stage37 floor 吃掉收益。Stage41/42 的有效形式更像 protected composition，而不是一个可以直接裸部署的 neural dynamics head。

### 4.3 为什么 external zero-shot 崩

坐标、尺度、horizon、agent type、scene/goal context 都不一致。把 SDD pixel-space 学到的规则直接迁移到 external dataset-local 坐标，会导致预测尺度和切换策略失真。

### 4.4 为什么 t50 曾经修不好

t50 需要长历史、目标方向、停走/转向/密度规避信息。早期模型只有浅层 metadata / all-objective，导致 t50 policy 选择 fallback，不敢切换。Stage37 加 past-only history + goal prototypes 后才修复。

### 4.5 为什么 t100 / h100 仍是 blocker

t100 raw-frame/h100 切片存在低 margin、高歧义、source support 稀疏和 validation-to-test source-family shift。不是简单加 threshold 或全局模型容量能修复。

## 5. 当前项目质量判断

如果按“工程可部署候选”看：

- SDD 和 external selector-level protected policy 已有强证据。
- Stage42-FH/FI 的 dual-domain/source-level frozen policy 证据更强，且有 replay/CI。

如果按“论文候选”看：

- 可以形成一篇 protected 2.5D world-state / safe-switch / source-level group-consistency 方向的论文候选。
- 但不能写 foundation 或 true 3D。
- 不能把 JEPA/Transformer 当主贡献。
- 需要非常明确地写 limitations。

如果按“真正强的世界模型”看：

- 还差 metric/time calibration。
- 还差更多合法、可转换、带 scene/time/scale 支持的 external datasets。
- 还差能脱离 teacher floor 仍安全的 neural dynamics。
- 还差 t100/h100 robust horizon。
- 还差 scene/goal/interaction 的独立强贡献。

## 6. 最短下一步

1. **补 source support / legal terms / raw source conversion。**
   TrajNet|100、UCY|100 的 blocker 很大一部分是 source support，不是模型小修能解决。

2. **针对 h100 做 source-specific row-level long-horizon model。**
   不要再全局 threshold；要用更强 history、neighbor、goal、source-family features。

3. **继续把 claim 写窄。**
   当前主 claim 应该是 protected dataset-local/raw-frame 2.5D world-state candidate + safe-switch + group-consistency/full-waypoint source-level evidence。

4. **不要启用 Stage5C / SMC。**
   当前 gates 还不支持 latent generative execution 或 SMC。

## 7. 验证记录

最近相关验证：

- Stage42-FU module contribution ledger runner：14/14。
- Stage42-FV claim boundary linter：15/15，violations = 0。
- Stage42-FW source action consolidator：16/16，conversion_ready_now = 0；没有下载、转换或评估。
- Stage42-DM reviewer replay package：27/27；最小 replay path 已覆盖 FU/FV/FW/CX/CZ/DK/DL/CV。
- Full pytest：861 passed。

本文没有新增训练；它是针对用户要求的详细总结 README。

<!-- STAGE42_FV_CLAIM_BOUNDARY_LINTER:START -->
## Stage42-FV Claim Boundary / No-Overclaim Linter

- source: `fresh_stage42_claim_boundary_linter_from_paper_package_and_fu`
- gate: `15 / 15`; verdict `stage42_fv_claim_boundary_linter_pass`.
- scanned files: `15`; violations: `0`.
- role: paper-package claim hygiene guard; no training, no threshold tuning, no conversion.
- boundary: M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
- blocked as independent main claims: JEPA, Transformer, scene/goal, neighbor/interaction.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_claim_boundary_linter.py -> 15/15', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_claim_boundary_linter.py tests/test_stage42_module_contribution_ledger.py -> 9 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 857 passed'}`.
<!-- STAGE42_FV_CLAIM_BOUNDARY_LINTER:END -->

<!-- STAGE42_FX_OBJECTIVE_COVERAGE_AUDIT:START -->
## Stage42-FX Objective Coverage Audit

- source: `fresh_stage42_objective_coverage_audit_from_current_evidence`
- gate: `15 / 15`; verdict `stage42_fx_objective_coverage_audit_pass`.
- objectives covered: `6`; blocked objectives `['A']`; partial objectives `['B', 'C', 'D']`; passed objectives `['E']`.
- current best status: `protected_dataset_local_raw_frame_2_5d_candidate`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- role: requirement coverage audit for the active Stage42 A-F long objective; no training, no download, no conversion, no threshold tuning.
- boundary: goal remains active and incomplete; M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_FX_OBJECTIVE_COVERAGE_AUDIT:END -->

<!-- STAGE42_FY_HORIZON_RETRY_DECISION_MAP:START -->
## Stage42-FY Horizon Retry Decision Map

- source: `fresh_stage42_horizon_retry_decision_map_from_fl_fq`
- gate: `14 / 14`; verdict `stage42_fy_horizon_retry_decision_pass`.
- weak horizons: `['TrajNet|100', 'UCY|100']`.
- model retry attempts considered: `5`; promoted policy count `0`.
- decision: stop repeating same-feature weak-horizon model retries now = `True`.
- highest-priority unblocker: `FW-TERMS-ucy_crowd_original`.
- role: retry decision map for h100 weak slices; no training, no download, no conversion, no threshold tuning.
- boundary: uniform horizon robustness remains blocked; protected dataset-local/raw-frame 2.5D only; no metric/seconds, true 3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_FY_HORIZON_RETRY_DECISION_MAP:END -->

<!-- STAGE42_FZ_PAPER_PACKAGE_FXFY_REFRESH:START -->
## Stage42-FZ Paper Package FX/FY Refresh

- source: `fresh_stage42_paper_package_fxfy_refresh`
- gate: `20 / 20`; verdict `stage42_fz_paper_package_fxfy_refresh_pass`.
- role: paper-package refresh over Stage42-FX objective coverage and Stage42-FY horizon retry decision map; no training, no download, no conversion, no test-threshold tuning.
- supported core claims: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint']`.
- blocked main claims: `['JEPA_downstream_lift', 'ungated_neural_dynamics', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'global_metric_seconds_claim']`.
- objective status: blocked `['A']`, partial `['B', 'C', 'D']`, passed `['E']`, goal_complete `False`.
- weak horizons: `['TrajNet|100', 'UCY|100']`; stop_repeat_modeling_now `True`; uniform_horizon_claim_allowed `False`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- boundary: protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_FZ_PAPER_PACKAGE_FXFY_REFRESH:END -->

<!-- STAGE42_GA_LIVE_SOURCE_CALIBRATION_RECHECK:START -->
## Stage42-GA Live Source / Calibration Recheck

- source: `fresh_stage42_live_source_calibration_recheck`
- gate: `15 / 15`; verdict `stage42_ga_live_source_calibration_recheck_pass`.
- role: fresh local path scan plus cached legal/calibration readiness recheck; no download, no conversion, no training, no evaluation.
- targets audited: `7`; local-path-found targets `7`; existing converted/cache targets `1`.
- new conversion-ready targets: `0`; source_action conversion_ready_now `0`; unified queue `0`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- boundary: local file presence is not legal conversion readiness; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GA_LIVE_SOURCE_CALIBRATION_RECHECK:END -->

<!-- STAGE42_GB_SOURCE_TERMS_PREFILL:START -->
## Stage42-GB Source Terms Prefill

- source: `fresh_stage42_gb_source_terms_prefill`
- gate: `15 / 15`; verdict `stage42_gb_source_terms_prefill_pass`.
- role: converts Stage42-GA local path evidence into a user-facing source-terms prefill draft; no download, conversion, training, evaluation, or permission claim.
- datasets prefilled: `5`; with suggested local path `5`; raw-source candidates `5`.
- conversion_ready_now: `0`; highest-priority next action `FW-TERMS-ucy_crowd_original`.
- prefill draft: `outputs/stage42_long_research/source_terms_confirmation_prefill_stage42.json`.
- boundary: prefill is not legal permission; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GB_SOURCE_TERMS_PREFILL:END -->

<!-- STAGE42_GC_PREFILL_INTAKE_BRIDGE:START -->
## Stage42-GC Prefill -> Intake Bridge

- source: `fresh_stage42_gc_prefill_intake_bridge`
- gate: `16 / 16`; verdict `stage42_gc_prefill_intake_bridge_pass`.
- role: adds GB local path/source identity suggestions into the EH intake template as non-permission `prefill_suggestion` hints.
- intake rows: `5`; suggestions added `5`; user-confirmed rows `0`.
- conversion_ready_now: `0`; updated intake template `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- boundary: user_confirmation is still blank; no download/conversion/training/evaluation; protected dataset-local/raw-frame 2.5D only.
<!-- STAGE42_GC_PREFILL_INTAKE_BRIDGE:END -->

<!-- STAGE42_GD_CALIBRATION_HINT_INTAKE_BRIDGE:START -->
## Stage42-GD Calibration Hint -> Intake Bridge

- source: `fresh_stage42_gd_calibration_hint_intake_bridge`
- gate: `18 / 18`; verdict `stage42_gd_calibration_hint_intake_bridge_pass`.
- role: adds DU metadata-only H/FPS/stride hints into the intake template as non-claim `calibration_prefill` leads.
- rows with hints: `3`; metric/time subset hint rows `2`.
- conversion_ready_now: `0`; metric/seconds claim allowed now `False` / `False`.
- boundary: hints are not permission, not conversion readiness, and not global metric/seconds evidence; Stage5C/SMC remain false.
<!-- STAGE42_GD_CALIBRATION_HINT_INTAKE_BRIDGE:END -->

<!-- STAGE42_GE_CONVERSION_CAPABILITY_INTAKE_BRIDGE:START -->
## Stage42-GE Conversion Capability -> Intake Bridge

- source: `fresh_stage42_ge_conversion_capability_intake_bridge`
- gate: `20 / 20`; verdict `stage42_ge_conversion_capability_intake_bridge_pass`.
- role: adds DW source-specific dry-run capability into the intake template as non-permission `conversion_capability_prefill`.
- source-specific rows available for `2` dataset rows; source-CV feasible after terms for `1` row.
- t50/t100 windows after terms: `10060` / `5696`; conversion_ready_now `0`.
- boundary: dry-run capability is not permission or conversion readiness; no download/conversion/training/evaluation; no metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GE_CONVERSION_CAPABILITY_INTAKE_BRIDGE:END -->

<!-- STAGE42_GF_POST_CONFIRMATION_CONVERSION_PLAN:START -->
## Stage42-GF Post-Confirmation Conversion Plan

- source: `fresh_stage42_gf_post_confirmation_conversion_plan`
- gate: `16 / 16`; verdict `stage42_gf_post_confirmation_conversion_plan_pass`.
- role: ranks GE source-specific conversion capability rows into a post-confirmation execution plan.
- planned source rows: `6`; technical-ready-after-terms sources `5`; source-CV-capable datasets `1`.
- t50/t100 after-terms windows: `10060` / `5696`; source_ready_now `0`; manifest ready targets `0`.
- EI validator recheck: `10 / 10`; FT unified guarded queue recheck: `12 / 12`, queue count `0`.
- verification: focused GF/GE/FT tests `11 passed`; full test suite `893 passed`.
- boundary: plan is not permission, not conversion, not evaluation; no metric/seconds/true-3D/foundation/Stage5C/SMC claim.
<!-- STAGE42_GF_POST_CONFIRMATION_CONVERSION_PLAN:END -->
