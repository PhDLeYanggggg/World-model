# M3W 长期目标工作总账：尝试路线、失败原因、成功证据与当前结论

更新时间：2026-05-27
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README、`research_state.json`，并纳入最近 `fresh_run` 的 Stage42-ES 到 Stage42-FO 结果。
本文件用途：把“在 M3W 这个长期目标里做了什么、试过哪些路线、哪些失败、为什么失败、哪些成功、当前大概是什么质量”集中写到一个 README。它不是新训练结果；不会把 cached 结果写成 fresh；不会把 diagnostic 结果写成 deployable success。

## 0. 一句话结论

M3W 已经从早期 SDD-only selector scaffold，推进到一个有 SDD 与 external top-down dataset-local raw-frame 证据的 **protected 2.5D multi-agent world-state candidate**。

最新补充结论：

```text
Stage42-EU/EV/EW/EX/EY 都没有提升到超过 Stage42-DI 的新 deployable policy。
Stage42-EZ 进一步测试 temporal group-repel shape，all/t50/hard 有极小正增量，但 near@0.05 比 Stage42-DI 差，因此不 promoted。
Stage42-FA waypoint-wise repel 修复了 proximity，但 all/hard 低于 Stage42-DI，因此同样不 promoted。
Stage42-FB 在 DI/FA 之间做 validation-only Pareto composer，near@0.05 进一步下降到 1.10%，但 all/hard 各损失约 0.07pp，因此是 safety-sensitive diagnostic，不是新 best deployable。
Stage42-FC 把 proximity / group-interaction signal 放进 supervised training objective 后，all/t50/hard 分别高于 Stage42-DI/FB，但 near@0.05 比 Stage42-DI 差约 0.48pp，因此不 promoted。
Stage42-FD 进一步把 FA waypoint-wise safety teacher 放进 train-only objective regularization，但 validation 选择回 teacher_alpha=0 的 FC-like 控制项；all/t50/hard 仍为正但略低于 FC，near@0.05 仍比 Stage42-DI 差约 0.48pp，因此不 promoted。
Stage42-FE 用 validation-only constrained FC→DI safety fallback，把 FC 高精度和 DI proximity safety 组合起来：all/t50/hard 为 26.41% / 23.15% / 24.81%，near@0.05 为 1.32%，比 FC 低 0.54pp 且不劣于 DI，因此 promotable。
Stage42-FF 已冻结 FE policy，并做 exact replay + 2000-bootstrap：all/t50/t100raw/hard 的 CI low 分别为 26.08% / 22.71% / 13.46% / 24.46%，replay max diff = 0。
Stage42-FG 随后做 source/domain/horizon 鲁棒性审计，结果是 partial：TrajNet robust，但 UCY 仍是 weak domain，TrajNet|100 也有 easy-safety 弱切片；因此不能把 FE/FF 的 global positive 包装成“每个 external source 都 positive”。
Stage42-FH 用 UCY train-only internal validation 重新选择 FE composer family，修复 FG 暴露的 UCY fallback-only 弱域：all/t50/t100raw/hard 为 34.98% / 28.97% / 20.57% / 33.10%，TrajNet 和 UCY 都 positive-safe，gate 20/20。
Stage42-FI 冻结 FH policy，并做 exact replay + 2000-bootstrap：policy hash f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6，replay diff 为 0，bootstrap CI low all/t50/t100raw/hard 为 34.62% / 28.46% / 19.96% / 32.73%，gate 25/25。
Stage42-FJ 对 frozen FH/FI policy 做 source/domain/horizon/scene 鲁棒性审计：TrajNet 与 UCY 两个 domain 均 robust positive-safe，所有 powered sources 都 robust，但 TrajNet|100、UCY|50、UCY|100 仍是 horizon weak slices；因此 dual-domain 与 broad source claim 可以写，uniform horizon claim 不能写。
Stage42-FK 针对这些 weak horizon 做 validation-only repair：全局 all/t50/t100raw/hard 变为 35.18% / 28.97% / 21.13% / 33.33%，但弱 horizon 数没有减少，仍是 TrajNet|100、UCY|50、UCY|100；因此 verdict 是 pass_with_horizon_limit，不允许 uniform horizon claim。
Stage42-FL 对 FK/FJ 剩余 weak horizon 做 fresh 取证：三个弱切片共同根因是 oracle label low-margin ambiguous；TrajNet|100 的 diagnostic oracle vs FH 只有 1.06%，UCY|50 为 6.75%，UCY|100 为 2.74%，且 0.05 relative-margin 内的低 margin 比例分别约 99.18%、92.52%、90.28%。因此下一步不是继续整片候选替换，而是训练 horizon-specific row-level switch model，且必须用更强 history/neighbor/goal features 和保守 safety gate。
Stage42-FM 按 FL 的诊断训练 validation-only row-level weak-horizon switch specialist：全局 all/t50/t100raw/hard 变为 35.20% / 29.03% / 21.14% / 33.35%，easy degradation 为 -37.10%，near@0.05 为 1.25%；弱切片从 3 个降到 2 个，UCY|50 被修复，但 TrajNet|100 与 UCY|100 仍因 easy-safety / low-margin ambiguity 没过 robust horizon gate。因此 FM 是有价值的 row-level repair，但 verdict 仍是 pass_with_horizon_limit，不允许 uniform horizon claim。
Stage42-FN 在 FM 后增加 validation-only conservative easy guard：全局 all/t50/t100raw/hard 为 34.86% / 29.03% / 20.19% / 32.96%，easy degradation 为 -37.14%，near@0.05 为 1.24%；但弱切片仍是 TrajNet|100 和 UCY|100，没有新增修复。FN 因此是有价值的负结果：更保守的 easy guard 可以保持全局安全，但会牺牲 all/t100/hard，仍不能解除 uniform horizon blocker。
Stage42-FO 进一步训练 validation-only row-level gain/harm specialist，输入包括 Stage37/past history/prototype/rollout diagnostics，future labels 只用于 validation training target。它在 TrajNet|100 上切换 1962 行、UCY|100 上选择 keep_fm；全局 all/t50/t100raw/hard 回到 35.20% / 29.03% / 21.14% / 33.35%，但 weak horizons 仍是 TrajNet|100 与 UCY|100。因此 FO 证明“更像模型的 gain/harm specialist”也还没有足够信号解除 low-margin horizon blocker。
这些结果的价值是负结果定位加正向修复：post-hoc repair 接近 Pareto 边界；objective-level training 能突破 all/hard；简单 safety-teacher target blend 不足；显式 constrained safety fallback 能修复 FC 的 proximity blocker；source/domain/horizon 审计发现 UCY weak；UCY internal-val support 进一步把 weak domain 修成 dual-domain positive-safe；FI 冻结和复放证明这个 policy 不是临时 test-tuned 结果；FJ/FK/FL/FM/FN/FO 则把允许 claim 精确收窄到 dual-domain/source robust，但不允许 uniform horizon overclaim，并解释 uniform horizon blocker 来自低 margin/高歧义 weak horizon；FM 证明 row-level switch 能修复一部分弱切片，FN 证明单纯更保守 easy guard 不能修复剩余 TrajNet|100 / UCY|100，FO 证明当前 past/prototype/rollout gain-harm features 仍不足以可靠预测剩余 h100 weak-slice safety。 但这仍是 dataset-local raw-frame 2.5D evidence，不能写 metric/seconds/true-3D/foundation。
```

## 0.1 本次给你的详细总结

### 我在这个目标里实际尝试过的主路线

1. **强因果基线与安全 fallback 路线。**
   从 constant velocity / damped velocity / scene-clamped / goal-directed 等强因果 baseline 出发，建立 selector、failure predictor、hard/failure bench、GoalBench、no-leakage audit。这个路线最终发展成 Stage26、Stage37 和 Stage42 的 protected policy 家族，是当前最可靠路线。

2. **JEPA 表征路线。**
   多轮训练 JEPA-only / scene-trajectory JEPA / interaction-aware JEPA，检查 non-collapse、probe、downstream heads。结论是多次 non-collapse，但没有稳定 downstream lift，因此不能作为主贡献，也不能说成 latent generative world model。

3. **Transformer / Hybrid neural dynamics 路线。**
   训练 Transformer-only、JEPA+Transformer hybrid、protected neural dynamics、full-waypoint sequence dynamics。无保护 neural 不安全；受 Stage37/teacher floor 保护的 neural/full-waypoint 变体有证据，但它仍是 protected world-state candidate，不是可独立部署的神经世界模型。

4. **SDD official pixel-space benchmark 路线。**
   把 SDD 做成 pixel raw-frame official benchmark，建立 scene packs、episodes、baselines、HardBench/FailureBench/GoalBench。Stage26 cost-aware selector 在 SDD 上成为 best deployable。

5. **External transfer / cross-domain 路线。**
   从 OpenTraj / UCY / ETH-UCY / TrajNet 等外部 top-down pedestrian 数据出发，经历 zero-shot 失败、normalization 失败、latent adapter 失败、row geometry 修复、history window 修复、goal prototype 修复、selective transfer 修复，最终 Stage37 修复 external t50，Stage42 继续推进到 source-level/full-waypoint/proximity/group-consistency。

6. **安全与物理有效性路线。**
   系统评估 easy degradation、harm over fallback、near-collision@0.05、jagged-rate、group consistency。这个路线证明普通 residual/correction 很容易伤 easy，必须用 safe-switch / proximity guard / fallback floor。

7. **统计与复现路线。**
   对关键 policy 做 bootstrap、exact replay、policy hash、schema hash、frozen policy、no-leakage report。最近 Stage42-FI 已把 FH policy freeze，并用 exact replay 和 2000-bootstrap 固化。

### 失败路线和失败原因

| 失败路线 | 具体表现 | 主要原因 | 处理方式 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 selector t50 约 -43.3%，easy degradation 约 11.33% | oracle label low-margin、class ambiguity、过度切换 easy cases | 改成 expected-FDE / regret-aware / fallback-safe selector |
| JEPA 主线 | non-collapse，但 selector/failure/t50/correction 无稳定 lift | 表征目标和部署损失错位；latent 没变成 gain/harm 信号 | 保留为 auxiliary/diagnostic，不做主 claim |
| SDD->external zero-shot | all 约 -92.67%，t50 约 -278.57% | 坐标、scale、horizon、agent type、scene/goal 缺失不兼容 | 做 coordinate-invariant、row geometry、relative targets |
| latent adapter | 分布距离缩小但预测不提升 | latent alignment 不等于 target alignment | 不把 adapter 当成功，只保留诊断 |
| external early selective transfer | all/hard 有正信号但 t50=0 | all objective 淹没 t50；缺 long-horizon history/goal prototype | Stage37 构建 past-only history + scene-agnostic goal prototypes |
| bounded residual / correction | 不稳定超过 Stage37，容易伤 easy | residual 直接改轨迹风险高，strong baseline 已很强 | 不部署 correction，保留 protected selector |
| unprotected Transformer/Hybrid | neural without fallback 不安全 | 数据仍是 dataset-local/raw-frame，scene/metric grounding 不足 | 只允许 protected neural/world-state candidate |
| scalar proximity/occupancy | all 有时提升，但 hard 或 safety 不够 | scalar loss 无法完整表达 group dynamics | 转向 explicit source/frame/horizon group-consistency |
| temporal/waypoint repel repair | proximity 或 accuracy 单边改善，但不 Pareto dominate | post-hoc 几何修复会牺牲 ADE 或 hard | 用 constrained composer / fallback 组合 |
| broad source robustness | FE/FF global positive，但 UCY weak | UCY 缺 train-only internal validation support | Stage42-FH 增加 UCY internal-val support 并重新冻结 |

### 成功路线和成功证据

| 成功点 | 证据 | 结论 |
| --- | --- | --- |
| SDD Stage26 cost-aware selector | t50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 +1.81% | SDD pixel raw-frame best deployable |
| External Stage37 selector | all +13.48%，t50 +8.46%，t50 CI +7.69% 到 +9.15%，hard +15.54%，easy 0.041%，gate 16/16 | external t50 transfer repaired deployable |
| Stage42-CO/CP bridge-shape composer | all +3.02%，t50 +1.50%，t100 raw +6.12%，hard +3.28%，2000-bootstrap positive | full-waypoint auxiliary bridge evidence |
| Stage42-CQ proximity guard | all +1.77%，t50 +1.07%，near@0.05 修复到不劣于 endpoint-linear | safety-sensitive composer |
| Stage42-DL/DQ/ES/ET group-consistency | all 约 +24.72%，t50 +22.36%，hard +23.89%，near@0.05 1.94% -> 1.38% | source/frame/horizon group-consistency 有真实价值 |
| Stage42-FE constrained FC/safety composer | all/t50/hard 26.41% / 23.15% / 24.81%，near@0.05 1.32%，gate 19/19 | 修复 FC 的 proximity blocker |
| Stage42-FH UCY-supported composer | all/t50/t100raw/hard 34.98% / 28.97% / 20.57% / 33.10%，UCY 与 TrajNet 都 positive-safe，gate 20/20 | 从 TrajNet robust 推进到 dual-domain positive-safe |
| Stage42-FI frozen replay | replay diff 0；2000-bootstrap CI low all/t50/t100raw/hard 34.62% / 28.46% / 19.96% / 32.73%；gate 25/25 | FH policy 已冻结，可复放，不是 test-tuned 偶然结果 |
| Stage42-FJ robustness audit | TrajNet 与 UCY domain robust；powered sources robust；TrajNet|100、UCY|50、UCY|100 仍 weak；gate 14/14 | 允许 dual-domain/source claim，但禁止 uniform horizon claim |
| Stage42-FK horizon repair attempt | all/t50/t100raw/hard 35.18% / 28.97% / 21.13% / 33.33%；weak horizons 仍为 TrajNet|100、UCY|50、UCY|100；gate 15/15 | 全局小幅提升，但 uniform horizon claim 仍 blocked |
| Stage42-FL weak-horizon forensics | TrajNet|100、UCY|50、UCY|100 的 root cause 都是 oracle label low-margin ambiguous；gate 15/15 | 解释 FK 为什么修不掉 uniform horizon：整片替换不够，需要 row-level horizon specialist |
| Stage42-FM row-level weak-horizon specialist | all/t50/t100raw/hard 35.20% / 29.03% / 21.14% / 33.35%；UCY|50 repaired；weak horizons reduced from 3 to 2；gate 15/15 | row-level switch 有效但不充分；TrajNet|100 和 UCY|100 仍 blocked，因此 uniform horizon claim 仍禁止 |
| Stage42-FN conservative easy guard | all/t50/t100raw/hard 34.86% / 29.03% / 20.19% / 32.96%；weak horizons 仍为 TrajNet|100、UCY|100；gate 15/15 | 更保守 easy guard 保持全局安全但牺牲 all/t100/hard，不能修复 uniform horizon blocker |
| Stage42-FO gain/harm specialist | all/t50/t100raw/hard 35.20% / 29.03% / 21.14% / 33.35%；TrajNet|100 切 1962 行，UCY|100 keep_fm；gate 16/16 | 模型化 gain/harm specialist 仍不能修复剩余 h100 weak horizons；需要更强 source/horizon-specific data 或更真实 long-horizon context |

但是当前仍然不是：

- true 3D world model
- large-scale foundation world model
- metric / meter-level predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative execution
- SMC-ready model

当前最诚实定位：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

当前 best deployable 分层：

| 用途 | 当前最强结果 | 状态 |
| --- | --- | --- |
| SDD pixel raw-frame official benchmark | Stage26 cost-aware selector | SDD t+50 与 hard/failure 正提升；仍是 pixel/raw-frame，不是 metric。 |
| External t+50 selector | Stage37 history + goal-prototype safe selector | external all/t50/hard/easy 同时过 gate，是 external selector best deployable。 |
| Protected neural/world-state candidate | M3W-Neural v1 / Stage41-42 protected policy family | 有 protected neural/full-waypoint/runtime evidence，但仍依赖 Stage37 / teacher safety floor。 |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware composer guard | 用一部分 ADE 增益换 near-collision 安全修复。 |
| Source-level full-waypoint policy | Stage42-DL/DQ/ES/ET group-consistency full-waypoint family | source/frame/horizon group-consistency 目标得到 fresh 支持；仍是 protected raw-frame 2.5D evidence。 |
| Group-risk/adaptive/temporal/waypoint/Pareto/objective follow-up | Stage42-EU/EV/EW/EX/EY/EZ/FA/FB/FC/FD/FE | 证明 risk bucket、temporal/waypoint repel、DI/FA Pareto composer、teacher blend 都不足；FE constrained FC→DI safety fallback 首次同时保留 FC all/t50/hard 并修复 proximity 到不劣于 DI。 |
| Paper claim | 受限 claim | 可以写 protected dataset-local raw-frame 2.5D world-state candidate；不能写 true 3D / foundation / metric / seconds-level / Stage5C / SMC。 |

## 1. 永久边界

所有阶段和所有报告都必须保留这些边界：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实物理米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 标签不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- future endpoint / future waypoint 只能作为 supervised label 或 evaluation label，不能作为 inference input。
- 不使用 central velocity official input。
- 不用 test endpoints 构建 goals。
- 不用 test metrics 调 threshold。
- 无保护 neural dynamics 不部署。

## 2. 路线总览

| 路线 | 做了什么 | 结果 | 核心原因 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | per-agent multi-agent 2.5D world-state scaffold、baseline fallback、diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 true 3D / foundation。 |
| JEPA representation | Stage18/19/后续多轮 JEPA non-collapse、probe、downstream lift 检查。 | 失败为主。 | non-collapse 不等于 downstream lift；selector/failure/correction/t50 没有稳定改善。 |
| SDD official benchmark | SDD world-state shards、scene packs、episodes、baselines、HardBench/GoalBench。 | 成功。 | SDD 成为 official pixel raw-frame benchmark；仍无 verified scale/homography。 |
| SDD hard-class selector | 预测 best baseline class。 | 失败。 | low-margin label、class ambiguity、easy over-switch，导致 Stage24 t+50 为负。 |
| SDD expected-FDE selector | 预测每个 baseline expected FDE/risk + fallback。 | 成功，Stage26。 | cost/regret/easy-safety 约束修复过度切换。 |
| External zero-shot | SDD selector / latent 直接迁移到 OpenTraj/ETH/UCY/TrajNet。 | 大失败。 | 坐标、scale、horizon、scene/goal、agent type 不兼容。 |
| Domain normalization / latent adapter | zscore、velocity/path normalization、CORAL、linear adapter。 | 不足。 | 缩小 latent distribution gap 不等于目标对齐。 |
| External row geometry / train-only goals | 补逐行几何、relative target、train-only candidate goals。 | 局部正信号。 | t50/hard 有空间，但 all/easy 不稳。 |
| Selective transfer | hard/easy/failure labels + gain/harm/fallback policy。 | 部分成功。 | all/hard/easy 可过，但 t50 初期仍 fallback 0。 |
| Stage37 causal history + goal prototypes | past-only history window、scene-agnostic goal prototypes、switchability/conformal safety。 | 成功。 | t+50 终于可安全切换并正迁移。 |
| Bounded correction / residual | Stage37 保护下做 bounded delta correction。 | 不部署。 | 未稳定超过 Stage37，且 residual 容易伤 easy。 |
| Transformer / JEPA / Hybrid neural | Stage37 保护下训练 neural dynamics。 | 诊断为主。 | 无保护 neural 不安全；受保护 neural 没稳定超过 Stage37。 |
| Full-waypoint / source-level | row cache、full-waypoint dynamics、source-level full-waypoint evaluation。 | protected 成功。 | 直接 full-waypoint 训练/评估比 endpoint bridge 更可信。 |
| Interaction / occupancy target | scalar proximity/occupancy、explicit group-consistency repair、group-schema ablation。 | Stage42-ES/ET 支持 explicit group-consistency。 | source/frame/horizon group target 比 isolated control 有小但正的增量，并修复 near-collision。 |

## 3. 关键成功结果

### 3.1 SDD：Stage26 cost-aware selector

结果来源：`cached_verified`

```text
Stage26 selector:
  t+50 improvement: about +14.58%
  hard/failure improvement: about +11.23%
  easy degradation: about +1.81%
```

意义：

- Stage26 是 SDD pixel raw-frame 上的 best deployable selector。
- 它修复了 Stage24 hard-class selector 的 easy over-switch。
- 它不是 metric predictor，不是 true 3D，也不是 foundation。

### 3.2 External：Stage37 t+50 transfer repaired

结果来源：`cached_verified`

```text
Stage37:
  all improvement: +13.48%
  t+50 improvement: +8.46%
  t+50 bootstrap CI: [+7.69%, +9.15%]
  hard/failure improvement: +15.54%
  easy degradation: 0.041%
  gates: 16 / 16
  verdict: stage37_t50_transfer_repaired_deployable
```

意义：

- Stage35/36 的问题是 all/hard 正但 t50 仍为 0。
- Stage37 用 past-only history window + scene-agnostic goal prototypes + gain/harm/safety gate 修复了 t50。
- 这是 external selector-level deployable success，但仍是 dataset-local/raw-frame，不是 metric/seconds-level。

### 3.3 M3W-Neural v1 protected package

结果来源：`cached_verified`

```text
M3W-Neural v1 protected package:
  all ADE improvement: about +21.03%
  t50 improvement: about +13.65%
  t100 raw-frame diagnostic improvement: about +14.69%
  hard/failure improvement: about +20.38%
  easy degradation: 0
```

意义：

- 它是 protected neural/world-state candidate，不是 ungated neural dynamics。
- 有意义的部分来自 Stage37 / teacher safety floor 下的 protected composition。
- 不应写成“神经网络已经独立超过全部 baseline”。

### 3.4 Stage42 common-validation composer / proximity guard

结果来源：`cached_verified`

Stage42-CO common-validation composer：

```text
test vs endpoint-linear ADE:
  all: +3.02%
  t50: +1.50%
  t100 raw diagnostic: +6.12%
  hard/failure: +3.28%
```

Stage42-CP bootstrap：

```text
bootstrap_n = 2000
all CI: [+2.64%, +3.37%]
t50 CI: [+0.90%, +2.09%]
t100 raw CI: [+5.39%, +6.94%]
hard/failure CI: [+2.90%, +3.68%]
```

Stage42-CQ proximity-aware guard：

```text
test vs endpoint-linear ADE:
  all: +1.77%
  t50: +1.07%
  t100 raw diagnostic: +3.48%
  hard/failure: +1.93%
  easy degradation: +0.25%
near_collision@0.05 delta vs endpoint-linear: -0.06%
```

意义：

- CO/CP 给 accuracy evidence。
- CQ 用一部分 ADE 增益换 near-collision 安全修复。
- CR 进一步证明 no-guard accuracy 更高但 proximity risk 更差；guard 是安全/准确率 Pareto tradeoff。

### 3.5 Stage42 source-level full-waypoint / group-consistency

结果来源：`cached_verified` + Stage42-ES/ET `fresh_run`

Stage42-AM source-level full-waypoint：

```text
rows: 47458
all improvement: about +24.58%
t50 improvement: about +22.02%
t100 raw diagnostic improvement: about +14.37%
hard/failure improvement: about +23.75%
easy degradation: about -25.66%
```

Stage42-DL/DQ runtime group-consistency policy：

```text
rows: 47458
all improvement: about +24.72%
t50 improvement: about +22.36%
t100 raw diagnostic improvement: about +14.35%
hard/failure improvement: about +23.89%
near@0.05: 1.94% -> 1.38%
switch exact match: true
```

Stage42-ES interaction / occupancy target selection：

```text
selected target family: explicit_group_consistency_repair
gate: 17 / 17
verdict: stage42_es_interaction_occupancy_target_selection_pass

scalar proximity/occupancy:
  all: +25.51%
  t50: +22.14%
  t100 raw: +14.34%
  hard: +23.74%
  easy: -29.23%
  delta vs Stage42-AM all: +0.93%
  delta vs Stage42-AM hard: -0.01%
  status: diagnostic, not selected

explicit group-consistency:
  all: +24.72%
  t50: +22.36%
  t100 raw: +14.35%
  hard: +23.89%
  easy: -25.63%
  delta vs Stage42-AM all: +0.14%
  delta vs Stage42-AM hard: +0.14%
  near@0.05: 1.94% -> 1.38%
  status: selected
```

Stage42-ET group-consistency target ablation：

```text
selected target: source_frame_horizon
gate: 16 / 16
verdict: stage42_et_group_consistency_target_ablation_pass

source_frame_horizon:
  all: +24.72%
  t50: +22.36%
  t100 raw: +14.35%
  hard: +23.89%
  easy: -25.63%
  near@0.05: 1.38%

agent_isolated_no_interaction control:
  all: +24.58%
  t50: +22.02%
  t100 raw: +14.37%
  hard: +23.75%
  easy: -25.66%

source_frame_horizon increment vs isolated:
  all: +0.14%
  t50: +0.35%
  hard: +0.14%
  easy degradation increment: +0.03%
  own-base near@0.05 reduction: +0.55%
  p05 min-distance gain vs isolated: +7.77%
```

意义：

- scalar proximity/occupancy 不是完全无用，但没有被选作 deployable interaction target，因为 hard improvement 不超过 baseline-family control。
- explicit group-consistency 被选择，因为它在 all/hard 上小幅超过 Stage42-AM，同时 near-collision 更安全。
- Stage42-ET 说明 source/frame/horizon group target 的增量不是单纯 scalar loss artifact；它比 no-interaction isolated control 有小但正的 t50/hard/all 增益。
- 这仍然是 protected source-level raw-frame 2.5D evidence，不是 metric/seconds-level，也不是 floor-free neural claim。

## 4. 关键失败路线与原因

### 4.1 JEPA non-collapse 但 downstream 无 lift

表现：

- Stage18 / Stage19 / Stage22 / Stage23 / later JEPA 多轮 non-collapse。
- 但 selector、failure predictor、goal predictor、hard/failure correction、official t+50 没有稳定改善。

原因：

- latent variance 正常只说明没有 collapse。
- JEPA target 与 cost-aware decision / failure / correction 的部署目标错位。
- scene/video/trajectory latent 没有转化为可部署的 gain/harm/easy-safety 信号。

结论：

```text
JEPA 当前只能作为 auxiliary / diagnostic，不是主贡献，也不是生成式 world model。
```

### 4.2 Stage24 hard-class selector 大失败

表现：

```text
oracle headroom: about 46.2%
trained hard-class selector t+50 improvement: about -43.3%
easy degradation: about 11.33%
```

原因：

- 直接预测 best baseline class 会强迫 low-margin / ambiguous 样本切换。
- easy cases 的错误切换代价很高。
- 没有 cost/regret/fallback/easy guard。

修复：

- Stage25/26 改成 expected-FDE / regret-aware / confidence-gated / fallback-safe policy。

### 4.3 SDD -> external zero-shot 崩溃

表现：

```text
Stage31 SDD -> external zero-shot:
  all improvement: about -92.67%
  t50 improvement: about -278.57%
external adapted selector:
  about 0 improvement
```

原因：

- SDD 是 pixel-space，external 是 dataset-local / weak metric diagnostic。
- scale、frame step、horizon、agent type、scene/goal availability 不一致。
- latent adapter 缩小分布距离，但没有对齐预测目标。

修复方向：

- Stage33-37 做 coordinate-invariant features、row geometry、train-only goals、relative targets、history windows、goal prototypes。

### 4.4 Stage34/35/36：external 局部正信号但 t+50 不过

表现：

- Stage34：t50/hard 局部正，但 all/easy 不稳。
- Stage35：all +12.13%，hard/failure +13.98%，easy 0.041%，但 t50 = 0。
- Stage36：只调 horizon threshold 仍不能修好。

原因：

- t50 有 oracle headroom，但现有特征不足以判断“何时安全切换”。
- all-test objective 会淹没 t50。
- held-out scene 缺 test goals，不能用 test endpoints。

修复：

- Stage37 构建 past-only history window 和 scene-agnostic goal prototypes，训练 t50-specific switchability / gain / harm / conformal safety。

### 4.5 Bounded residual / correction 不部署

表现：

- Stage38 bounded correction、later residual/correction variants 没有稳定超过 Stage37。
- 容易伤 easy cases。

原因：

- residual 直接改轨迹风险高。
- selected baseline 已经很强，错误 residual 会放大 harm。
- 没有足够强的 physical/scene constraints 时，不应部署。

结论：

```text
correction head 只保留 diagnostic；当前 best deployable 仍是 protected selector / safe-switch policy。
```

### 4.6 Transformer / Hybrid neural 没有成为无保护主模型

表现：

- Stage39/40 训练 Transformer / JEPA / Hybrid。
- neural without fallback 不安全。
- neural with fallback 没有稳定超过 Stage37。

原因：

- Stage37 的 hand-engineered causal history + goal prototype + conservative gate 已经很强。
- neural model 容易学习到 selector imitation，但不能稳定提供额外 dynamics lift。
- 数据仍是 dataset-local/raw-frame，scene/image/metric grounding 不足。

结论：

```text
不能部署 ungated neural dynamics；当前 neural evidence 必须写成 protected / teacher-floor candidate。
```

### 4.7 t+100 仍是 blocker

表现：

- 多次 t100 repair / source-CV / shadow holdout 发现 t100 positive gain 不稳。
- source-CV guard 可保 all/t50/hard/easy，但经常把 t100 gain 回退为 0。

原因：

- t100-capable independent sources 不足。
- horizon/scale/source differences 更严重。
- raw-frame t100 不能写成 seconds-level long horizon。

结论：

```text
t100 只能写 diagnostic；不能作为 stable deployable main claim。
```

## 5. 当前模型质量

最准确说法：

```text
M3W 当前是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
它已经有 SDD 与 external 的 selector-level / protected full-waypoint evidence。
它还不是 true 3D、不是 foundation、不是 global metric/seconds-level predictor。
```

从论文候选角度看：

- 可以写的贡献：strict no-leakage raw-frame multi-agent benchmark pipeline；cost-aware fallback-safe selector；external t50 transfer repair；source-level full-waypoint protected evaluation；group-consistency interaction/occupancy safety target。
- 不能写的贡献：true 3D、foundation、metric long-horizon、ungated neural dynamics、JEPA generative world model、SMC、Stage5C execution。

从部署角度看：

- SDD：Stage26 selector 是 best deployable。
- External：Stage37 selector 是 external selector best deployable。
- Source-level full-waypoint：Stage42 group-consistency protected policy family 是最强 evidence family，但仍应在 protected floor 下报告。
- Neural：只能 protected，不应 floor-free 部署。

## 6. 为什么当前还不是“真正强的多模态世界模型”

主要差距：

1. **缺 metric / time geometry。**
   没有全局 verified homography、meter-per-pixel、annotation stride、effective seconds。

2. **外部数据仍不够完整。**
   ETH / TrajNet / UCY 的 legal/source/time/t100 support 还没有完全闭合。

3. **神经 dynamics 独立贡献不足。**
   Transformer/Hybrid 仍没有在无保护情况下稳定超过 Stage37。

4. **JEPA 没有证明 downstream lift。**
   non-collapse 不足以成为主贡献。

5. **t100 稳定性不足。**
   t100 raw-frame diagnostic 不能写成稳定 seconds-level long-horizon success。

6. **scene/goal/interaction 独立贡献有限。**
   当前最强机制更多来自 baseline-family rollout context + safe-switch + group-consistency repair，而不是完整 scene/video/graph world representation。

## 7. 下一步最值得做

1. **继续 Stage42 的 interaction/occupancy 方向，但不要只调 scalar loss。**
   Stage42-ES/ET 已经说明 explicit source/frame/horizon group-consistency 更值得推进。下一步应该做 group-consistency constraint training，而不是把 scalar proximity loss 当主线。

2. **补 external source / legal / time / t100 support。**
   若想把 external 证据从 UCY/limited source-level 推成更强跨域 claim，必须补 ETH_UCY / TrajNet / UCY 的 source terms、time geometry、t100-capable split，而不是继续在同一批 cached rows 上榨指标。

3. **如果继续做神经世界模型，必须让 neural 学 group-consistency / gain-harm / full-waypoint constraint。**
   不是继续训练普通 residual；应让 neural 学会什么时候安全切换、怎样保持群体一致性、怎样不破坏 easy cases。

## 8. 本文件相关最新校验

最近已完成并记录的关键校验：

```text
Stage42-ES run: 17 / 17 gates
Stage42-ET run: 16 / 16 gates
Stage42-EU run: 15 / 18 gates
Stage42-EV run: 12 / 14 gates
Stage42-EW run: 14 / 16 gates
Stage42-EX run: 15 / 17 gates
Stage42-EY run: 16 / 18 gates
Stage42-EZ run: 17 / 18 gates
Stage42-FA run: 15 / 17 gates
Stage42-FB run: 14 / 16 gates
Stage42-FC run: 22 / 23 gates
latest focused tests for Stage42-ES/ET/EU/EV/EW/EX/EY/EZ/FA/FB/FC: passed
latest full pytest after Stage42-FC refresh: 786 passed in 36.07s
```

本次 README 更新本身是总结与索引更新，不是新训练，不改变模型 gate。

<!-- STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING:START -->
## Stage42-EU Group-Consistency Constraint Training

- source: `fresh_stage42_group_consistency_constraint_training`
- role: trains source/frame/horizon group-risk weighted full-waypoint dynamics, then applies validation-selected group repair.
- gate: `15 / 18`; verdict `stage42_eu_group_consistency_constraint_training_positive_not_promoted`.
- selected training variant: `group_unsafe_weighted` with lambda `10.0`.
- test all/t50/t100raw/hard/easy: `22.81%` / `22.35%` / `12.68%` / `21.97%` / `-23.91%`.
- delta vs Stage42-DI all/hard/easy: `-1.90%` / `-1.91%` / `1.72%`.
- near@0.05 base/final: `1.88%` / `1.33%`.
- decision: `group_constraint_training_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING:END -->

<!-- STAGE42_EV_CONSTRAINT_AWARE_COMPOSER:START -->
## Stage42-EV Constraint-Aware Composer

- source: `fresh_stage42_constraint_aware_composer`
- role: validation-only composer over floor / Stage42-AM / Stage42-DI / Stage42-EU by domain, horizon, and group-risk buckets.
- gate: `12 / 14`; verdict `stage42_ev_constraint_aware_composer_positive_not_promoted`.
- selected composer mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.71%` / `22.35%` / `14.35%` / `23.88%` / `-25.10%`.
- delta vs Stage42-DI all/hard/easy: `-0.00%` / `-0.00%` / `0.53%`.
- near@0.05 base/final: `1.94%` / `1.37%`.
- decision: `constraint_aware_composer_positive_but_keep_stage42_di`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EV_CONSTRAINT_AWARE_COMPOSER:END -->

<!-- STAGE42_EW_ADAPTIVE_GROUP_REPAIR:START -->
## Stage42-EW Adaptive Group Repair

- source: `fresh_stage42_adaptive_group_repair`
- role: validation-only adaptive repair over Stage42-DI candidate grid by global / domain+horizon / domain+horizon+risk slices.
- gate: `14 / 16`; verdict `stage42_ew_adaptive_group_repair_positive_not_promoted`.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ew_adaptive_group_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EW_ADAPTIVE_GROUP_REPAIR:END -->

<!-- STAGE42_EX_GROUP_LEVEL_RISK_REPAIR:START -->
## Stage42-EX Group-Level Risk Repair

- source: `fresh_stage42_group_level_risk_repair`
- role: validation-only adaptive repair with risk aggregated to source/frame/horizon groups before candidate selection.
- gate: `15 / 17`; verdict `stage42_ex_group_level_risk_repair_positive_not_promoted`.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ex_group_level_risk_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EX_GROUP_LEVEL_RISK_REPAIR:END -->

<!-- STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR:START -->
## Stage42-EY Continuous Group-Risk Repair

- source: `fresh_stage42_continuous_group_risk_repair`
- role: validation-only continuous group-risk bucket repair over Stage42-DI repair candidates.
- gate: `16 / 18`; verdict `stage42_ey_continuous_group_risk_repair_positive_not_promoted`.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ey_continuous_group_risk_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR:END -->

<!-- STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR:START -->
## Stage42-EZ Temporal Group-Repel Repair

- source: `fresh_stage42_temporal_group_repel_repair`
- role: tests temporal weighting for group-repel offsets after Stage42-EW/EX/EY risk-bucket repairs failed to beat Stage42-DI.
- selected candidate: `{'mode': 'temporal_repel', 'temporal_kind': 'tail', 'gamma': 1.0, 'direction_mode': 'nearest_current', 'min_sep': 0.12, 'margin': 0.0, 'strength': 0.25}`.
- gate: `17 / 18`; verdict `stage42_ez_temporal_group_repel_repair_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.73%` / `22.40%` / `14.35%` / `23.89%` / `-25.64%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `0.01%` / `0.04%` / `0.00%` / `0.00%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.51%`.
- decision: `temporal_group_repel_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR:END -->

<!-- STAGE42_FA_WAYPOINTWISE_GROUP_REPEL_REPAIR:START -->
## Stage42-FA Waypoint-Wise Group-Repel Repair

- source: `fresh_stage42_waypointwise_group_repel_repair`
- role: tests per-waypoint group-consistency offsets after Stage42-EZ temporal single-direction repair failed proximity promotion.
- selected candidate: `{'mode': 'waypointwise_repel', 'min_sep': 0.12, 'strength': 0.2, 'temporal_kind': 'sqrt_tail', 'gamma': 1.0, 'smooth': True, 'cap_scale': 0.75}`.
- gate: `15 / 17`; verdict `stage42_fa_waypointwise_group_repel_repair_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.61%` / `22.05%` / `14.36%` / `23.77%` / `-25.67%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `-0.11%` / `-0.31%` / `0.02%` / `-0.11%` / `-0.03%`.
- near@0.05 base/final: `1.94%` / `1.21%`.
- decision: `waypointwise_group_repel_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FA_WAYPOINTWISE_GROUP_REPEL_REPAIR:END -->

<!-- STAGE42_FB_PROXIMITY_PARETO_COMPOSER:START -->
## Stage42-FB Proximity Pareto Composer

- source: `fresh_stage42_proximity_pareto_composer`
- role: validation-only composer between Stage42-DI accuracy policy and Stage42-FA proximity-safety policy.
- selected candidate: `{'mode': 'group_di_near_fa_safer', 'threshold': 0.05, 'margin': 0.0}`.
- gate: `14 / 16`; verdict `stage42_fb_proximity_pareto_composer_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.65%` / `22.19%` / `14.35%` / `23.82%` / `-25.64%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `-0.07%` / `-0.18%` / `0.00%` / `-0.07%` / `-0.01%`.
- near@0.05 final/use_fa_rate: `1.10%` / `9.34%`.
- decision: `proximity_pareto_composer_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FB_PROXIMITY_PARETO_COMPOSER:END -->

<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:START -->
## Stage42-FC Objective-Level Proximity Training

- source: `fresh_stage42_objective_level_proximity_training`
- role: moves proximity/group-interaction signal from post-hoc repair into supervised full-waypoint training objective.
- selected objective: `label_proximity_objective`; feature mode `stage42_am_features`; lambda `10.0`.
- gate: `22 / 23`; verdict `stage42_fc_objective_level_proximity_training_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `26.37%` / `23.01%` / `14.02%` / `24.76%` / `-31.10%`.
- delta vs Stage42-DI all/hard/near005: `1.66%` / `0.87%` / `0.48%`.
- decision: `objective_level_training_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:END -->

<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:START -->
## Stage42-FD Safety-Aware Joint Objective Training

- source: `fresh_stage42_safety_aware_joint_objective_training`
- role: tests whether FA safety-teacher regularization inside the training objective can break the FC accuracy/proximity tradeoff.
- selected objective: `fc_label_proximity_control`; feature mode `stage42_am_features`; lambda `100.0`; teacher alpha `0.0`.
- gate: `22 / 26`; verdict `stage42_fd_safety_aware_joint_objective_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `26.33%` / `22.70%` / `14.02%` / `24.69%` / `-31.11%`.
- delta vs Stage42-FC all/hard/near005: `-0.04%` / `-0.07%` / `0.01%`.
- delta vs Stage42-DI all/hard/near005: `1.62%` / `0.80%` / `0.48%`.
- decision: `safety_aware_objective_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:END -->

<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:START -->
## Stage42-FE Constrained FC/Safety Composer

- source: `fresh_stage42_constrained_fc_safety_composer`
- role: validation-only constrained composer from high-accuracy Stage42-FC to DI/FA/FB safety fallbacks.
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.0025}`.
- gate: `19 / 19`; verdict `stage42_fe_constrained_fc_safety_composer_pass_promotable`.
- test all/t50/t100raw/hard/easy: `26.41%` / `23.15%` / `14.01%` / `24.81%` / `-31.06%`.
- delta vs FC all/hard/near005: `0.04%` / `0.05%` / `-0.54%`.
- delta vs DI all/hard/near005: `1.69%` / `0.92%` / `-0.06%`.
- decision: `promote_stage42_fe_constrained_fc_safety_composer`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:END -->

<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:START -->
## Stage42-FF FE Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fe_policy_freeze_replay`
- role: freeze Stage42-FE constrained FC/safety composer and add 2000-bootstrap plus exact replay evidence.
- gate: `23 / 23`; verdict `stage42_ff_fe_policy_freeze_replay_pass`.
- frozen policy hash: `a78db26aa155b38799f5b866f32a2d205018adf2054d9409a016da3163328dff`.
- replay all/t50/t100raw/hard/easy: `26.41%` / `23.15%` / `14.01%` / `24.81%` / `-31.06%`.
- bootstrap lows all/t50/t100raw/hard: `26.08%` / `22.71%` / `13.46%` / `24.46%`.
- exact replay max metric/diagnostic diff: `0.0` / `0.0`.
- Boundary: frozen protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:END -->

<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:START -->
## Stage42-FG FE Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fe_source_robustness_audit`
- role: audit frozen Stage42-FE/FF across domain/source/horizon/scene slices without retraining or threshold reselection.
- gate: `11 / 12`; verdict `stage42_fg_fe_source_robustness_partial`.
- robust domains: `['TrajNet']`.
- weak domain-horizon slices: `['TrajNet|100', 'UCY|10', 'UCY|25', 'UCY|50', 'UCY|100']`.
- weak sources: `['TrajNet/Train/crowds/crowds_zara03.txt']`.
- broad uniform source claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:START -->
## Stage42-FH UCY-Supported FE Composer

- source: `fresh_stage42_ucy_supported_fe_composer`
- role: repair Stage42-FG UCY fallback-only weakness by adding train-only UCY internal validation before FE composer selection.
- gate: `20 / 20`; verdict `stage42_fh_ucy_supported_fe_composer_pass`.
- positive safe domains: `['TrajNet', 'UCY']`; weak domains: `[]`.
- all/t50/t100raw/hard/easy: `34.98%` / `28.97%` / `20.57%` / `33.10%` / `-36.91%`.
- decision: `promote_stage42_fh_ucy_supported_fe_composer`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:END -->

<!-- STAGE42_FI_FH_POLICY_FREEZE_REPLAY:START -->
## Stage42-FI FH Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fh_policy_freeze_replay`
- role: freeze Stage42-FH UCY-supported FE composer and add 2000-bootstrap plus exact replay evidence.
- gate: `25 / 25`; verdict `stage42_fi_fh_policy_freeze_replay_pass`.
- frozen policy hash: `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`.
- replay all/t50/t100raw/hard/easy: `34.98%` / `28.97%` / `20.57%` / `33.10%` / `-36.91%`.
- bootstrap lows all/t50/t100raw/hard: `34.62%` / `28.46%` / `19.96%` / `32.73%`.
- exact replay max metric/diagnostic diff: `0.0` / `0.0`.
- dual-domain support: UCY `True`, TrajNet `True`.
- Boundary: frozen protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FI_FH_POLICY_FREEZE_REPLAY:END -->

<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:START -->
## Stage42-FJ FH Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fh_source_robustness_audit`
- role: audit frozen Stage42-FH/FI policy across domain/source/horizon/scene slices without retraining or threshold reselection.
- gate: `14 / 14`; verdict `stage42_fj_fh_source_robustness_pass`.
- robust domains: `['TrajNet', 'UCY']`.
- weak domains: `[]`.
- robust domain-horizon slices: `['TrajNet|10', 'TrajNet|25', 'TrajNet|50', 'UCY|10', 'UCY|25']`.
- weak domain-horizon slices: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- robust sources: `['TrajNet/Test/crowds/students002.txt', 'TrajNet/Train/crowds/crowds_zara03.txt', 'TrajNet/Train/crowds/students003.txt']`.
- weak sources: `[]`.
- dual-domain positive-safe claim allowed: `True`.
- broad uniform source claim allowed: `True`.
- broad uniform horizon claim allowed: `False`.
- Boundary: frozen protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:START -->
## Stage42-FK FH Horizon Weak-Slice Validation Repair

- source: `fresh_stage42_fh_horizon_weak_slice_repair`
- role: validation-only repair attempt for FJ weak horizon slices; no retraining and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fk_fh_horizon_weak_slice_repair_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.18%` / `28.97%` / `21.13%` / `33.33%` / `-36.88%`.
- weak horizons before: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- applied overrides: `{'TrajNet|100': {'candidate': 'fb', 'rows': 5608, 'reason': 'validation_safe_best_score'}, 'UCY|50': {'candidate': 'fh', 'rows': 2340, 'reason': 'validation_safe_best_score'}, 'UCY|100': {'candidate': 'fa', 'rows': 1440, 'reason': 'validation_safe_best_score'}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:END -->

<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:START -->
## Stage42-FL FH Weak-Horizon Forensics

- source: `fresh_stage42_fh_horizon_weak_slice_forensics`
- role: fresh diagnostic for FK/FJ weak horizons; no policy promotion and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fl_horizon_weak_slice_forensics_pass`.
- analyzed weak horizons: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- root cause counts: `{'oracle_label_low_margin_ambiguous': 3}`.
- next action: `train_horizon_specific_row_level_switch_model_with_stronger_history_neighbor_goal_features`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC; uniform horizon claim still blocked.
<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:END -->

<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:START -->
## Stage42-FM FH Weak-Horizon Row-Level Switch Specialist

- source: `fresh_stage42_fh_horizon_row_switch_specialist`
- role: validation-only row-level specialist attempt for FK/FJ/FL weak horizon slices; no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.20%` / `29.03%` / `21.14%` / `33.35%` / `-37.10%`.
- weak horizons before: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied policies: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'feature_threshold', 'candidate': 'fb', 'feature': 'path_length', 'direction': 'ge', 'threshold': 0.3749999749633932, 'rows': 5608, 'switch_rows': 3008}, 'UCY|50': {'key': 'UCY|50', 'mode': 'feature_threshold', 'candidate': 'di', 'feature': 'endpoint_delta_fh', 'direction': 'le', 'threshold': 0.026976035023941254, 'rows': 2340, 'switch_rows': 1170}, 'UCY|100': {'key': 'UCY|100', 'mode': 'feature_threshold', 'candidate': 'fb', 'feature': 'endpoint_delta_floor', 'direction': 'ge', 'threshold': 0.02336742544527692, 'rows': 1440, 'switch_rows': 936}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:END -->

<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:START -->
## Stage42-FN FH Horizon Conservative Easy Guard

- source: `fresh_stage42_fh_horizon_conservative_easy_guard`
- role: validation-only conservative easy-safety guard for FM remaining weak horizon slices; no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fn_conservative_easy_guard_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `34.86%` / `29.03%` / `20.19%` / `32.96%` / `-37.14%`.
- weak horizons before: `['TrajNet|100', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied guards: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'feature_guard', 'replacement': 'floor', 'feature': 'path_length', 'direction': 'le', 'threshold': 0.3749999749633932, 'rows': 5608, 'guard_rows': 2593}, 'UCY|100': {'key': 'UCY|100', 'mode': 'feature_guard', 'replacement': 'fa', 'feature': 'min_distance', 'direction': 'le', 'threshold': 0.12583341276755197, 'rows': 1440, 'guard_rows': 288}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:END -->

<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:START -->
## Stage42-FO FH Horizon Gain/Harm Specialist

- source: `fresh_stage42_fh_horizon_gain_harm_specialist`
- role: validation-only row-level gain/harm specialist for remaining weak horizon slices; no test threshold tuning.
- gate: `16 / 16`; verdict `stage42_fo_gain_harm_specialist_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.20%` / `29.03%` / `21.14%` / `33.35%` / `-37.10%`.
- weak horizons before: `['TrajNet|100', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied policies: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'gain_harm_model', 'gain_min': 0.0, 'harm_max': 0.35, 'max_switch': 0.35, 'rows': 5608, 'switch_rows': 1962}, 'UCY|100': {'key': 'UCY|100', 'mode': 'keep_fm', 'rows': 1440, 'switch_rows': 0}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:END -->
