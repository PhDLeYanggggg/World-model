# M3W 长期目标工作总账：尝试路线、失败原因、成功证据与当前结论

更新时间：2026-05-27
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README、`research_state.json`，并纳入最近 `fresh_run` 的 Stage42-ES 到 Stage42-GY 结果。
本文件用途：把“在 M3W 这个长期目标里做了什么、试过哪些路线、哪些失败、为什么失败、哪些成功、当前大概是什么质量”集中写到一个 README。它不是新训练结果；不会把 cached 结果写成 fresh；不会把 diagnostic 结果写成 deployable success。

当前更便于阅读的单文件总账已同步到：`README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md`。该文件是本轮面向用户的主 summary；本文件保留为更长的历史总账。

## 本次交付版摘要

你要的总结已经集中写在本文件中。最短结论如下：

```text
当前 M3W 的真实质量：
  protected dataset-local / raw-frame 2.5D multi-agent world-state candidate

当前不是：
  true 3D world model
  large-scale foundation world model
  metric / meter-level predictor
  seconds-level long-horizon predictor
  ungated neural dynamics deployable model
  Stage5C latent generative execution
  SMC-ready system

当前 best deployable 分层：
  SDD: Stage26 cost-aware selector
  External t+50: Stage37 causal-history + goal-prototype safe selector
  Protected source/domain/full-waypoint: Stage42-FH/FI frozen policy family
  Paper/evidence boundary: Stage42-FU/FV/FW/FX/FY/GH/GI claim guards
  Source/legal blocker handling: Stage42-GW/GX/GY h100 blocker closure + UCY integrity/terms prefill
```

最重要的进展链：

1. **Stage26 在 SDD pixel/raw-frame 上成功。**
   Cost-aware selector 达到 t+50 约 `+14.58%`、hard/failure 约 `+11.23%`、easy degradation 约 `+1.81%`。这证明 hard-class selector 失败后，expected-FDE / regret-aware / fallback-safe selector 是正确方向。

2. **Stage37 修复 external t+50 transfer。**
   Past-only history window、scene-agnostic goal prototypes、gain/harm/safety gate 让 external 迁移从 t50=0 推进到 all `+13.48%`、t+50 `+8.46%`、t50 bootstrap CI `[+7.69%, +9.15%]`、hard/failure `+15.54%`、easy degradation `0.041%`、gate `16 / 16`。

3. **Stage42-FH/FI 把 source/domain protected policy 固化。**
   FH 通过 UCY train-only internal validation 修复 UCY weak-domain：all/t50/t100raw/hard 为 `34.98% / 28.97% / 20.57% / 33.10%`，TrajNet 与 UCY 都 positive-safe。FI 冻结并 exact replay：policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`，replay diff `0`，2000-bootstrap CI low all/t50/t100raw/hard 为 `34.62% / 28.46% / 19.96% / 32.73%`，gate `25 / 25`。

4. **Stage42-FU/GI 约束论文 claim。**
   Stage42-FU module ledger 允许主 claim 的模块只有：`history`、`domain_expert`、`safe_switch`、`teacher_floor`、`group_consistency_full_waypoint`、`full_waypoint_shape`、`endpoint_bridge`。被阻止作为主 claim 的模块是：`scene_goal`、`neighbor_interaction`、`JEPA`、`Transformer`。Stage42-GI 刷新 paper claim evidence audit，gate `25 / 25`，明确 post-confirmation calibrated subset 只是候选计划，不是 permission、conversion 或 evaluation。

5. **Stage42-GH 给出下一步可校准数据路线，但不能写成已完成。**
   GH 识别 post-confirmation calibrated ETH/UCY subset candidates：restricted candidates after terms = `5`，ready now = `0`，after-terms calibrated t50/t100 windows = `10060 / 5696`，domains = `ETH_UCY, UCY`。这只是用户确认 terms/path/source identity 后的候选图，不是下载、转换、评估或 metric/seconds claim。

6. **Stage42-GW/GX/GY 把 h100 / UCY legal blocker 从“模糊阻塞”变成可执行清单。**
   GW 明确：`TrajNet|100` 是 hard blocker，原因是缺少 official long raw TrajNet source；`UCY|100` 有技术候选但 legal conversion not ready，因此 `can_run_repair_now_count = 0`。GX 对 UCY candidate files 做 integrity manifest：`6 / 6` 文件存在，target-family candidates = `2`，parsed rows = `98,032`，parsed t100 windows = `11,848`，但 conversion_ready_now 仍为 `0`。GY 基于 GX 生成 terms prefill：prefill rows = `6`，hash/source identity suggestion 均已填入，但 `terms_accepted_by_user=false`、`allowed_use=""`、`confirmed_by_user=""`，agent 不能自动填写 legal acceptance。结论：这三步成功关闭了 h100 blocker 的证据链，但没有下载、转换、评估，也没有解除 legal blocker。

## 0. 最新当前版：路线、失败、成功、质量判断

### 0.1 我在这个长期目标内真正尝试过的路线

| 路线 | 做了什么 | 当前结论 |
| --- | --- | --- |
| 数据采集与 registry | 搜索/登记 SDD、OpenTraj、ETH-UCY、UCY、TrajNet、egocentric/video、simulation/traffic diagnostic；建立 license/action/user-required reports。 | 成功建立数据采集框架；但 registry-only 不算 converted，legal/terms 未确认的数据不能用作 official success。 |
| SDD official pixel benchmark | SDD 解压、转换 world-state shards、scene packs、lazy episodes、HardBench/FailureBench/GoalBench、no-leakage audit、strong causal baselines。 | 成功；但 SDD 是 pixel-space raw-frame，不是 metric/seconds-level。 |
| SDD selector | 从 hard-class selector 到 expected-FDE / regret-aware / fallback-safe selector。 | hard-class 失败，Stage26 cost-aware selector 成功，成为 SDD best deployable。 |
| 外部跨域迁移 | OpenTraj/UCY/ETH-UCY/TrajNet feature store、row geometry、normalization、relative target、external baselines、selective transfer。 | zero-shot 大失败；Stage37 通过 history/prototype/safety 修复 external t50。 |
| 神经网络世界动力学 | JEPA-only、Transformer-only、Hybrid、bounded correction、full-waypoint sequence、protected neural candidate。 | 无保护 neural 不部署；protected neural / full-waypoint 有证据，但仍依赖 Stage37/teacher safety floor。 |
| 安全/物理有效性 | easy degradation、harm over fallback、near@0.05、jagged-rate、proximity guard、group-consistency。 | 成功建立 safety gate；多条高精度路线因 proximity/easy 失败而不 promoted。 |
| Source/domain/full-waypoint policy | Stage42-DL/DM runtime replay、CO/CP/CQ bridge/shape、FE/FH/FI source-domain protected policies。 | 当前最强 source/domain protected evidence，已 freeze/replay/bootstrap，但仍不允许 uniform horizon claim。 |
| Claim/paper guard | module ledger、claim linter、paper evidence audit、paper freeze manifest、source-action consolidator、horizon retry map。 | 成功防止过度 claim：JEPA/Transformer/scene-goal/neighbor-interaction 不能写独立主贡献。 |
| Legal/source blocker closure | Stage42-GW/GX/GY 对 h100 与 UCY candidate 做 blocker decision、integrity manifest、terms prefill。 | 成功把下一步用户动作具体化；但 legal 未确认前 conversion/eval 必须是 `not_run`。 |

### 0.2 失败路线与失败原因

| 失败/受阻路线 | 失败表现 | 根因 | 现在的处理 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 t50 improvement 约 `-43.3%`，easy degradation 约 `11.33%`。 | oracle best-baseline label 低 margin、高歧义；hard label 迫使 easy case 过度切换。 | 改成 expected-FDE / regret-aware / fallback-safe selector。 |
| Stage18/19/22/23 JEPA 主线 | 多次 non-collapse，但 selector/failure/goal/t50/correction 无稳定 downstream lift。 | 表征目标和部署收益/风险目标不对齐；latent variance 不等于可用 gain/harm 信号。 | 只能写 auxiliary/diagnostic，不能当主贡献或生成式 world model。 |
| SDD->external zero-shot | Stage31 外部 all improvement 约 `-92.67%`，t50 约 `-278.57%`。 | SDD pixel 与 external dataset-local 坐标、scale、horizon、agent type、scene/goal context 不兼容。 | 做 external row geometry、relative target、history window、goal prototype。 |
| 普通 normalization / latent adapter | latent gap 缩小但 selector 无正提升。 | 分布距离变小不代表任务损失、gain/harm、easy-safety 对齐。 | 不再把 latent distance reduction 写成 predictive success。 |
| Stage34/35 early selective transfer | t50/hard 局部正，但 all/easy 不稳，或 t50=0。 | all objective 淹没 long-horizon；缺 t50 专用 history/goal/switchability。 | Stage37 专门修 t50。 |
| bounded residual / correction | 未稳定超过 Stage37，普通 residual 容易伤 easy。 | 直接改轨迹比选择/回退更危险，strong baseline floor 已很强。 | correction 不部署，除非先过 selector/failure/safety gate。 |
| 无保护 Transformer/Hybrid | neural without fallback 不安全或不超过 Stage37。 | 当前数据仍是 dataset-local/raw-frame，metric/scene grounding 不足；模型学会复制或错误切换。 | 只允许 Stage37/teacher floor protected neural evidence。 |
| scene/goal 独立主 claim | 多轮 gate 后贡献不稳定或被 baseline/context 吸收。 | train-only goal/scene proxy 对 held-out/domain shift 支持有限。 | Stage42-FU/GJ 明确不能作为独立主贡献。 |
| neighbor/interaction 独立主 claim | scalar neighbor/interaction 有时局部正，但无法稳定独立提升。 | 原始 neighbor scalar 不足以表达群体时空约束。 | 只允许 group-consistency full-waypoint 作为受限贡献。 |
| uniform h100/horizon claim | TrajNet|100、UCY|100 持续 weak；UCY|50 后被 FM 修复但 h100 仍阻塞。 | low-margin ambiguity、source support 稀疏、h100 long-horizon context 不足、legal conversion 未 ready。 | GW/GX/GY 先建立 blocker/integrity/terms prefill，不强行跑。 |

### 0.3 成功路线与核心证据

| 成功点 | 关键数字 | 质量边界 |
| --- | --- | --- |
| Stage26 SDD cost-aware selector | t50 `+14.58%`；hard/failure `+11.23%`；easy degradation `+1.81%`。 | SDD pixel/raw-frame best deployable；不是 metric。 |
| Stage37 external t50 repair | all `+13.48%`；t50 `+8.46%`；t50 CI `[+7.69%, +9.15%]`; hard/failure `+15.54%`; easy `0.041%`; gates `16/16`。 | external dataset-local/raw-frame deployable selector。 |
| M3W-Neural v1 protected candidate | all `+21.03%`; t50 `+13.65%`; t100 raw `+14.69%`; hard/failure `+20.38%`; easy `0.00%`; gates `41/41`。 | protected neural candidate under Stage37/teacher floor；不是 ungated neural deployment。 |
| Stage42-DL/DM runtime replay | runtime rows `47,458`; switch exact match true; all/t50/t100raw/hard `+24.72% / +22.36% / +14.35% / +23.89%`; near@0.05 `1.94% -> 1.38%`。 | reviewer replay / runtime evidence。 |
| Stage42-CQ proximity guard | all/t50/t100raw/hard `+1.77% / +1.07% / +3.48% / +1.93%`; near@0.05 不劣于 endpoint-linear/floor。 | safety-sensitive composer，牺牲部分 ADE 换安全。 |
| Stage42-FE constrained safety composer | all/t50/hard `26.41% / 23.15% / 24.81%`; near@0.05 `1.32%`; gate `19/19`。 | 修复 FC proximity blocker，promotable protected policy。 |
| Stage42-FH UCY-supported composer | all/t50/t100raw/hard `34.98% / 28.97% / 20.57% / 33.10%`; TrajNet/UCY 都 positive-safe; gate `20/20`。 | source/domain protected policy。 |
| Stage42-FI freeze/replay | policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`; replay diff `0`; CI low all/t50/t100raw/hard `34.62% / 28.46% / 19.96% / 32.73%`; gate `25/25`。 | frozen policy 非 test-tuned 偶然结果。 |
| Stage42-FU/GJ module claim lock | allowed main modules = history、domain expert、safe switch、teacher floor、group-consistency full-waypoint、full-waypoint shape、endpoint bridge；blocked = JEPA、Transformer、scene_goal、neighbor_interaction。 | 论文 claim 边界已锁。 |
| Stage42-GW/GX/GY h100 blocker closure | GW gate `17/17`; GX candidate files `6/6`, rows `98,032`, t100 windows `11,848`; GY terms prefill rows `6`, gate `14/14`。 | 只说明 blocker 被结构化；legal 未确认前不可 conversion/eval。 |

### 0.4 当前 best deployable 分层

| 场景 | 当前 best | 是否部署 |
| --- | --- | --- |
| SDD pixel/raw-frame | Stage26 cost-aware selector | 可部署于 SDD pixel/raw-frame benchmark。 |
| External t50 | Stage37 causal-history + goal-prototype safe selector | 可部署于 external dataset-local/raw-frame selector task。 |
| Protected neural/world-state | M3W-Neural v1 composite-tail safe-switch | 仅作为 Stage37/teacher floor protected candidate。 |
| Source/domain/full-waypoint | Stage42-FH/FI frozen protected policy family | 可作为 protected source/domain evidence；不能写 uniform horizon。 |
| h100/uniform horizon | 仍 blocked | TrajNet|100 缺 raw source；UCY|100 需 legal confirmation/guarded conversion。 |

### 0.5 当前一句话质量判断

```text
M3W 当前是 protected dataset-local / raw-frame 2.5D multi-agent world-state candidate。
它已经有 SDD、external t50、source/domain protected policy、runtime replay、bootstrap、no-leakage、claim guard 证据。
它还不是 true 3D、不是 foundation、不是 global metric/seconds-level、不是 ungated neural dynamics deployable。
```

### 0.6 下一步最短路径

1. **先解决 legal/source blocker。** 使用 Stage42-GY prefill，让用户明确确认 UCY/ETH_UCY/TrajNet 的 official source identity、terms accepted、allowed use、local path。agent 不能代填 legal acceptance。
2. **只对 legal-ready source 做 guarded conversion。** conversion 后重新跑 no-leakage、source-CV、baseline、Stage37/Stage42 policy replay。
3. **再修 h100/uniform horizon。** 对 TrajNet|100 / UCY|100 需要真实 long-horizon source support、row-level h100 context 和 stricter easy-safety gate。
4. **神经网络路线继续但不越界。** 只训练 gain/harm、group-consistency、full-waypoint consistency、source/horizon-aware switchability；不训练普通无保护 residual，不执行 Stage5C/SMC。

## 一句话结论

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
Stage42-FP 进一步把 TrajNet|100 / UCY|100 拆到 source、scene、validation support 和 oracle margin 层面，结论是两个 h100 weak slices 都存在 source-family shift、单/稀疏 validation support、low-margin ambiguity、low material headroom 和 source-specific easy-safety CI failure。因此下一步必须补 source support 或更强 h100 long-horizon context，而不是继续盲目调全局 threshold。
这些结果的价值是负结果定位加正向修复：post-hoc repair 接近 Pareto 边界；objective-level training 能突破 all/hard；简单 safety-teacher target blend 不足；显式 constrained safety fallback 能修复 FC 的 proximity blocker；source/domain/horizon 审计发现 UCY weak；UCY internal-val support 进一步把 weak domain 修成 dual-domain positive-safe；FI 冻结和复放证明这个 policy 不是临时 test-tuned 结果；FJ/FK/FL/FM/FN/FO/FP 则把允许 claim 精确收窄到 dual-domain/source robust，但不允许 uniform horizon overclaim，并解释 uniform horizon blocker 来自低 margin/高歧义、source support 和 h100 context 不足；FM 证明 row-level switch 能修复一部分弱切片，FN 证明单纯更保守 easy guard 不能修复剩余 TrajNet|100 / UCY|100，FO 证明当前 past/prototype/rollout gain-harm features 仍不足以可靠预测剩余 h100 weak-slice safety，FP 证明剩余 blocker 还带有明确 source/support 层面的缺口。 但这仍是 dataset-local raw-frame 2.5D evidence，不能写 metric/seconds/true-3D/foundation。
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
- not seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative execution
- not SMC-ready model

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

<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:START -->
## Stage42-FP H100 Weak-Horizon Source / Support Audit

- source: `fresh_stage42_h100_weak_horizon_source_support_audit`
- role: diagnostic source/support decomposition for remaining h100 weak horizons after Stage42-FO; no new training and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fp_h100_source_support_audit_pass`.
- h100 weak horizons: `['TrajNet|100', 'UCY|100']`.
- blocker counts: `{'long_horizon_h100_context_still_insufficient': 2, 'low_material_headroom': 2, 'oracle_low_margin_ambiguous': 2, 'single_or_sparse_validation_source_support': 2, 'source_specific_easy_safety_ci_failure': 2, 'validation_to_test_source_family_shift': 2, 'gain_harm_policy_abstained_due_to_validation_safety': 1}`.
- recommended next action: `source_support_or_long_horizon_context_repair_before_retrying_policy_promotion`.
- conclusion: uniform horizon robustness remains blocked; TrajNet|100 and UCY|100 need source/support or stronger long-horizon context repair before any policy promotion.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `.venv-pytorch/bin/python run_stage42_h100_weak_horizon_source_support_audit.py` -> `15 / 15`; focused pytest `4 passed`; full pytest `832 passed in 30.13s`.
<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:END -->

<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:START -->
## Stage42-FQ H100 Source-Support Repair Queue

- source: `fresh_stage42_h100_source_support_repair_queue`
- role: local source-support repair queue for FP h100 blockers; no conversion, no training, no auto-download.
- gate: `15 / 15`; verdict `stage42_fq_h100_source_support_repair_queue_pass`.
- weak keys: `['TrajNet|100', 'UCY|100']`.
- local gap summary: `{'ETH_UCY': {'files': 18, 't100_files': 7, 'independent_t100_groups': 6, 'short_or_non_t100_files': 11}, 'TrajNet': {'files': 59, 't100_files': 0, 'independent_t100_groups': 0, 'short_or_non_t100_files': 59}, 'UCY': {'files': 24, 't100_files': 6, 'independent_t100_groups': 4, 'short_or_non_t100_files': 18}}`.
- TrajNet|100 status: no local long raw h100 TrajNet source; user must provide or confirm official longer source.
- UCY|100 status: local UCY h100 candidates exist but are terms-unverified and require conversion/no-leakage/source-CV before use.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_h100_source_support_repair_queue.py -> 15/15', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_h100_source_support_repair_queue.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 836 passed'}`.
<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:END -->

<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:START -->
## Stage42-FR UCY H100 Terms-Gated Conversion Preflight

- source: `fresh_stage42_ucy_h100_terms_gated_conversion_preflight`
- role: file-level UCY h100 candidate preflight from FQ; no conversion, no training, no auto-download.
- gate: `14 / 14`; verdict `stage42_fr_ucy_h100_terms_gated_preflight_pass`.
- candidates: `6` total, `2` target-family candidates.
- conversion_preflight_ready_count: `0`; blockers `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'redistribution_policy_unknown', 'derived_data_policy_unknown', 'local_path_confirmation_missing', 'source_identity_missing', 'confirmed_by_user_missing']`.
- recommended first sources after user confirmation: `['UCY_zara02', 'UCY_zara01']`.
- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 840 passed'}`.
<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:END -->

<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:START -->
## Stage42-FS UCY H100 Terms Intake Validator

- source: `fresh_stage42_ucy_h100_terms_intake_validator`
- role: validates candidate-level UCY h100 terms intake and writes a guarded conversion queue; no conversion, training, download, or evaluation.
- gate: `14 / 14`; verdict `stage42_fs_ucy_h100_terms_intake_validator_pass`.
- candidate_rows_validated: `6`; target_family_candidates `2`.
- terms_ready_candidates: `0`; guarded_conversion_queue_count `0`.
- top blockers: `{'allowed_use_missing': 6, 'confirmed_by_user_missing': 6, 'derived_data_policy_unknown': 6, 'local_path_confirmation_missing': 6, 'redistribution_policy_unknown': 6, 'source_identity_missing': 6, 'terms_acceptance_date_missing': 6, 'terms_not_accepted': 6}`.
- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_ucy_h100_terms_intake_validator.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_intake_validator.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 844 passed'}`.
<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:END -->

<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:START -->
## Stage42-FT Unified Guarded Conversion Queue

- source: `fresh_stage42_unified_guarded_conversion_queue`
- role: unifies global source readiness and UCY H100 candidate readiness into one non-executing guarded conversion queue.
- gate: `12 / 12`; verdict `stage42_ft_unified_guarded_conversion_queue_pass`.
- source_ready_targets: `0`; h100_ready_candidates `0`; unified_queue_count `0`.
- blocked_action_count: `11`; downloaded/converted/evaluated now `0` / `0` / `0`.
- Boundary: queue only; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py -> 12/12', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_unified_guarded_conversion_queue.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 848 passed'}`.
<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:END -->

<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:START -->
## Stage42-FU Module Contribution Ledger

- source: `fresh_stage42_module_contribution_ledger_from_aa_y_bw_ec_dp_de`
- role: machine-readable claim ledger over AA/Y/BW/EC/DP/DE evidence; no new training or threshold tuning.
- gate: `14 / 14`; verdict `stage42_fu_module_contribution_ledger_pass`.
- main claim modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`.
- blocked/auxiliary modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`.
- Core supported claims: history, domain expert, safe-switch/teacher floor, and source-level group-consistency full-waypoint.
- Blocked as main independent claims under current evidence: JEPA downstream lift, Transformer-only contribution, scene/goal, neighbor/interaction, ungated neural/global metric/seconds.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_module_contribution_ledger.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_module_contribution_ledger.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 852 passed'}`.
<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:END -->

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

<!-- STAGE42_FW_SOURCE_ACTION_CONSOLIDATOR:START -->
## Stage42-FW Source Action Consolidator

- source: `fresh_stage42_source_action_consolidator_from_existing_blockers`
- gate: `16 / 16`; verdict `stage42_fw_source_action_consolidator_pass`
- consolidated actions: `10`; categories `{'legal_terms_and_local_path': 5, 'h100_weak_horizon_source_support': 2, 'domain_closure': 3}`
- top actions: `['FW-TERMS-ucy_crowd_original', 'FW-H100-TrajNet|100', 'FW-DOMAIN-TrajNet', 'FW-DOMAIN-UCY', 'FW-H100-UCY|100']`
- conversion_ready_now: `0`; blocked_action_count: `11`
- This is a source/legal/horizon action router only: no download, conversion, training, evaluation, metric/seconds claim, Stage5C execution, or SMC.
- Highest-value path remains UCY terms/path confirmation plus guarded conversion/no-leakage/source-CV; TrajNet h100 needs a longer legal source because local snippets are too short.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; not true 3D, not foundation, not metric/seconds-level.
<!-- STAGE42_FW_SOURCE_ACTION_CONSOLIDATOR:END -->

<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:START -->
## Stage42-GJ Module Claim Lock

- source: `fresh_stage42_gj_module_claim_lock_from_fu_z_dp_dq_gh`
- gate: `19 / 19`; verdict `stage42_gj_module_claim_lock_pass`.
- locked supported modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`.
- locked blocked modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`.
- protected full-waypoint runtime supported: `True`; ungated full-waypoint deployable: `False`.
- calibrated post-confirmation candidates: `5`; ready now: `0`; after-terms t50/t100: `10060` / `5696`.
- next admissible experiments are restricted to terms-confirmed guarded conversion, changed-target gain/harm or full-sequence context, protected full-waypoint runtime replay, and source/horizon-specific h100 support repair.
- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or post-confirmation-candidate-as-data claim.
<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:END -->

<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:START -->
## Stage42-GK Context Switchability Family Audit

- source: `fresh_stage42_gk_context_switchability_family_audit`
- gate: `14 / 14`; verdict `stage42_gk_context_switchability_family_audit_pass`.
- decision: `context_switchability_family_not_supported`; material context families: `[]`.
- best family `baseline_plus_history_goal_neighbor` vs baseline-family control: all/t50/t100raw/hard/easy = `-0.000003` / `0.000000` / `0.000000` / `0.000006` / `0.000093`.
- Target changed from residual trajectory deltas to gain/harm/switchability. Future labels are train/val/eval labels only, never inference inputs.
- If no material family is supported, scene/goal/neighbor context remains blocked as an independent main claim under this changed-target audit.
- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or test-endpoint claim.
<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:END -->

<!-- STAGE42_GZ_FULL_WAYPOINT_CLAIM_GUARD:START -->
## Stage42-GZ Full-Waypoint Claim Guard

- source: `fresh_stage42_gz_full_waypoint_claim_guard`
- gate: `18 / 18`
- verdict: `stage42_gz_full_waypoint_claim_guard_pass`
- Protected full-waypoint evidence can be cited only as dataset-local/raw-frame 2.5D evidence.
- Endpoint-only or endpoint-linear bridge success must not be counted as learned full-waypoint dynamics.
- Ungated full-waypoint neural deployment remains rejected.
- Group-consistency full-waypoint is supported under protected policy; neighbor/interaction alone remains blocked as an independent main claim.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_GZ_FULL_WAYPOINT_CLAIM_GUARD:END -->

<!-- STAGE42_HA_FULL_WAYPOINT_OVERCLAIM_LINTER:START -->
## Stage42-HA Full-Waypoint Overclaim Linter

- source: `fresh_stage42_ha_full_waypoint_overclaim_linter`
- gate: `14 / 14`
- verdict: `stage42_ha_full_waypoint_overclaim_linter_pass`
- files_scanned: `15`
- violations_total: `0`
- Endpoint/full-waypoint, ungated full-waypoint, group/neighbor independent-main, metric/seconds, Stage5C and SMC overclaims were scanned.
- No unsupported full-waypoint overclaim lines were found.
<!-- STAGE42_HA_FULL_WAYPOINT_OVERCLAIM_LINTER:END -->

<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:START -->
## Stage42-HB Teacher-Floor Necessity Meta-Audit

- source: `fresh_stage42_hb_teacher_floor_necessity_meta_audit`
- gate: `16 / 16`
- verdict: `stage42_hb_teacher_floor_necessity_meta_audit_pass`
- Direct conclusion: Stage37 / teacher floor is the current safety mechanism and rollout-context floor, not merely a disposable crutch.
- Protected current all/t50/t100raw/hard/easy: `21.03%` / `13.65%` / `14.69%` / `20.38%` / `0.00%`.
- Ungated endpoint/full-waypoint easy degradation remains unsafe: `124.59%` / `124.59%`.
- Narrow t50 floor relaxation is supported only on selected slices: rows `11538`, t50 `28.97%`, hard `28.97%`, easy `-21.41%`.
- Global floor removal and floor-free neural deployment remain false.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:END -->

<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:START -->
## Stage42-HC Floor-Alternative Gate Stress Matrix

- source: `fresh_stage42_hc_floor_alternative_gate_stress`
- gate: `14 / 14`
- verdict: `stage42_hc_floor_alternative_gate_stress_pass`
- Tested Stage42-E internal self-gate, uncertainty gate, conformal risk gate, harm predictor, teacher-dependent gates, and bounded residual families as floor alternatives.
- floor-free deployable count: `0`; teacher-dependent deployable count: `6`.
- best floor-free candidate `harm_predictor_gate` reaches all/t50/hard `35.95%` / `25.20%` / `35.86%` but is not deployable because `['near_collision_delta_over_1pp']`.
- best deployable teacher-dependent candidate `current_composite_tail_policy` reaches all/t50/hard `21.03%` / `13.65%` / `20.38%` with easy `0.00%`.
- Deployment decision remains: keep Stage37/teacher floor globally; allow only validation-backed partial t50 relaxation on selected slices.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:END -->

<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:START -->
## Stage42-HD Floor-Free Proximity-Guard Repair

- source: `fresh_stage42_hd_floor_free_proximity_guard_repair`
- gate: `13 / 13`
- verdict: `stage42_hd_floor_free_proximity_guard_repair_pass`
- Tested floor-free internal/harm/uncertainty/conformal gates with a validation-selected proximity guard.
- pre-guard deployable count: `0`; post-guard deployable count: `4`.
- best post-guard family `harm_predictor_gate` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%` with easy `0.00%` and collision delta `-0.47%`.
- The teacher gate is not used in this repair, but causal floor fallback remains required; this is not global floor removal.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:END -->

<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:START -->
## Stage42-HE Floor-Free Proximity-Guard Robustness Audit

- source: `fresh_stage42_he_floor_free_proximity_guard_robustness`
- gate: `21 / 21`
- verdict: `stage42_he_floor_free_proximity_guard_robustness_pass`
- Audits the Stage42-HD teacherless proximity-guard repaired gate with 2000-bootstrap and per-domain/per-horizon checks.
- policy `harm_predictor_gate` with min_sep `0.05` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%`.
- bootstrap CI lows all/t50/t100raw/hard `20.38%` / `13.22%` / `12.94%` / `19.57%`; easy CI high `-16.17%`.
- robust_positive_domains: `ETH_UCY, TrajNet, UCY`; weak_domain_horizon_slices: `none`.
- Teacher gate is not used, but causal floor fallback remains required. This is not global floor removal, not metric/seconds, not true 3D, not Stage5C, and not SMC.
<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:END -->

<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:START -->
## Stage42-HF Teacherless Gate Deployment Contract

- source: `fresh_stage42_hf_teacherless_gate_deployment_contract`
- verdict: `stage42_hf_teacherless_gate_deployment_contract_pass`
- gates: `15 / 15`
- result: Stage42-HE supports a teacherless proximity-guarded switch gate, but only with causal floor fallback.
- metrics: all `20.74%`, t50 `13.82%`, t100 raw diagnostic `13.68%`, hard/failure `19.99%`, easy degradation `0.00%`.
- allowed claim: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked claims: global causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C execution, and SMC.
- deployment default remains protected causal-floor fallback; Stage42-HF is a claim/deployment contract refresh, not new training.
<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:END -->

<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:START -->
## Stage42-HG Teacherless / Floor-Free Claim Linter

- source: `fresh_stage42_hg_teacherless_claim_linter`
- verdict: `stage42_hg_teacherless_claim_linter_pass`
- gates: `15 / 15`
- scanned files: `18`; violations: `0`.
- allowed phrase: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked: global floor-free neural deployment, causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C, and SMC.
- role: applies Stage42-HF contract to the paper/README surface; this is not new training or threshold tuning.
<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:END -->

<!-- STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS:START -->
## Stage42-HI Restricted Metric/Time Readiness

- source: `fresh_stage42_hi_restricted_metric_time_readiness`
- verdict: `stage42_hi_restricted_metric_time_readiness_pass_blocked_by_terms`
- gates: `14 / 14`
- restricted metric/time candidates: `6` across `['ETH_UCY', 'UCY']`.
- technical ready after terms: `6`; ready now: `0`.
- conclusion: ETH/UCY source-level H/FPS/stride evidence exists, but no metric/seconds claim is allowed until user-confirmed source terms plus conversion/no-leakage/source-CV/final-test.
- no training, conversion, download, Stage5C, or SMC occurred.
<!-- STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS:END -->

<!-- STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT:START -->
## Stage42-HJ Restricted Metric/Time Source-CV Preflight

- source: `fresh_stage42_hj_restricted_metric_time_source_cv_preflight`
- verdict: `stage42_hj_restricted_metric_time_source_cv_preflight_pass_with_eth_ucy_source_cv_limit`
- gates: `15 / 15`
- usable after terms sources: `4`; ready now: `0`.
- source-CV feasible after terms: `['UCY']`; robust after terms: `['UCY']`.
- source-CV blocked after terms: `['ETH_UCY']`.
- window potential after terms: t50 `9845`, t100 `5696`.
- conclusion: restricted metric/time source-CV is technically plannable for UCY and blocked for ETH_UCY by current t100 source support; source terms still block all conversion/evaluation claims.
<!-- STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT:END -->

<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:START -->
## Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight

- source: `fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight`
- verdict: `stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked`
- gates: `16 / 16`
- augmented ETH_UCY independent sources after terms: `5`.
- augmented ETH_UCY t50/t100 windows after terms: `4397` / `1433`.
- cached BL technical t100 safe-positive: `True`; ready now: `False`.
- conclusion: ETH_UCY source-CV blocker is technically repairable after terms using ETH-Person XML candidates, but conversion/evaluation and metric/seconds claims remain blocked until user-confirmed terms and guarded rerun.
<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:END -->

<!-- STAGE42_IO_HORIZON_SEQUENCE_GRAPH_CONTEXT_ROUTER:START -->
## Stage42-IO Horizon-Specific Sequence+Graph Context Router

- source: `fresh_stage42_horizon_sequence_graph_context_router`
- role: tests whether splitting t10/t25/t50/t100 fixes the negative Stage42-EQ global sequence+graph context router.
- gate: `13 / 13`; verdict `stage42_io_horizon_sequence_graph_context_router_pass`.
- positive_horizon_sequence_graph_context_routers: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`.
- best_overall_router: `h10_motion_goal_context`.
- best all/t50/t100raw/hard/easy: `0.069270` / `0.000000` / `0.000000` / `0.072655` / `-0.035269`.
- horizon_specific_increment_verdict: `stage42_io_horizon_sequence_graph_context_router_supported`.
- Boundary: fresh horizon-specific router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IO_HORIZON_SEQUENCE_GRAPH_CONTEXT_ROUTER:END -->

<!-- STAGE42_IP_T50_T100_SEQUENCE_GRAPH_BLOCKER_AUDIT:START -->
## Stage42-IP t50/t100 Sequence+Graph Blocker Audit

- source: `fresh_stage42_t50_t100_sequence_graph_blocker_audit`
- role: explains why Stage42-IO sequence+graph context did not become deployable at t50/t100.
- gate: `12 / 12`; verdict `stage42_ip_t50_t100_sequence_graph_blocker_audit_pass`.
- t50_diagnosis: `router_under_switches_despite_headroom`.
- t100_diagnosis: `weak_predictive_signal_or_baseline_family_dominance`.
- blocker_counts: `{'unsafe_or_uncalibrated_switching': 2, 'weak_predictive_signal_or_baseline_family_dominance': 2, 'router_under_switches_despite_headroom': 1, 'low_margin_candidate_ambiguity': 1}`.
- conclusion: blocker audit only; no new deployable model and no t50/t100 context contribution claim.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IP_T50_T100_SEQUENCE_GRAPH_BLOCKER_AUDIT:END -->

<!-- STAGE42_IQ_T50_SWITCHABILITY_CALIBRATION_REPAIR:START -->
## Stage42-IQ t50 Switchability Calibration Repair

- source: `fresh_stage42_t50_switchability_calibration_repair`
- role: formal repair attempt for Stage42-IP t50 under-switching using validation-selected gain/harm calibration.
- gate: `11 / 11`; verdict `stage42_iq_t50_switchability_calibration_repair_pass`.
- repair_supported: `False`; repair_verdict `validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom`.
- best_trial: `baseline_plus_history_goal_neighbor__gain_only`.
- best test t50 / hard / easy: `0.000001` / `0.000001` / `-0.000000`.
- conclusion: if unsupported, do not continue pure threshold tuning; next step needs changed supervision/source support/candidate family.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IQ_T50_SWITCHABILITY_CALIBRATION_REPAIR:END -->

<!-- STAGE42_IR_T50_SOURCE_PATTERN_SWITCHABILITY_REPAIR:START -->
## Stage42-IR t50 Source-Pattern Switchability Repair

- source: `fresh_stage42_t50_source_pattern_switchability_repair`
- role: formal source-support repair attempt for Stage42-IQ t50 switchability failure.
- gate: `11 / 11`; verdict `stage42_ir_t50_source_pattern_switchability_repair_pass`.
- repair_supported: `False`; repair_verdict `t50_source_pattern_switchability_repair_not_supported`.
- best_trial: `history_only__gain_only`.
- best test t50 / hard / easy: `0.000000` / `0.000000` / `-0.000000`.
- conclusion: source-pattern support does not repair the context t50 route under this protocol; future repair needs new candidate policies or source data.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IR_T50_SOURCE_PATTERN_SWITCHABILITY_REPAIR:END -->

<!-- STAGE42_IS_DATA_CALIBRATION_REFRESH:START -->
## Stage42-IS 数据与标定刷新

- source: `fresh_run_on_current_head_after_stage42_ir`
- 做了什么：在 source-pattern t50 修复失败后，重新跑 Stage42-A/BN/DW 数据标定和 source-specific conversion dry-run。
- gates: Stage42-A `7 / 7`, Stage42-BN `13 / 13`, Stage42-DW `15 / 15`。
- 当前可继续外部域：`opentraj, eth_ucy, trajnet, ucy`。
- source-specific calibration candidates: `ETH_seq_eth`, `ETH_seq_hotel`, `UCY_zara01`, `UCY_zara02`, `UCY_zara03`, `UCY_students03`。
- terms 确认后技术可转换：`5 / 6`；估计 t50/t100 windows: `10060 / 5696`。
- source-CV 条件：只有 `UCY` 达到“terms 确认后可做 source-CV”；ETH/BIWI source 数不足，TrajNet 仍是短 snippet diagnostic，AerialMPT raw path 未找到。
- 结论：下一步最可信路线是 legal/confirmed UCY source-specific conversion 或补新 source；仍不能写 global metric/seconds claim。
- 验证：focused pytest `10 passed`；full pytest `1110 passed in 1980.35s`。
<!-- STAGE42_IS_DATA_CALIBRATION_REFRESH:END -->
