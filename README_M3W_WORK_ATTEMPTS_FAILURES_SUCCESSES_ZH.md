# M3W 长期目标工作总账：尝试路线、失败原因、成功证据与当前结论

更新时间：2026-05-27
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README、`research_state.json`，并纳入最近 `fresh_run` 的 Stage42-ES 到 Stage42-GY 结果。
本文件用途：作为我自己的长线研究总账，集中记录 M3W 走过的路线、失败原因、成功证据、当前质量和下一步判断。它不是新训练结果；不会把 cached 结果写成 fresh；不会把 diagnostic 结果写成 deployable success。

公开 GitHub README 规则（2026-06-01）：根目录 `README.md` 只做项目介绍，像我向研究者和开发者解释 M3W 一样写；内部阶段账本、gate、provenance 和长表格继续放在 `README_RESULTS.md`、`research_state.json` 和 `outputs/`。

当前更便于阅读的单文件总账已同步到：`README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md`。它是给读者快速理解项目路线的整理版；本文件保留为更长的历史总账。

## 本次交付版摘要

最短结论如下：

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

<!-- STAGE42_IT_SOURCE_LEVEL_FULL_WAYPOINT_REFRESH:START -->
## Stage42-IT Source-Level Full-Waypoint Fresh Refresh

- source: `fresh_run_on_current_head`
- 做了什么：在 Stage42-IS 标定刷新之后，重跑 Stage42-AM proposed source-level split full-waypoint evaluation。
- gate: `12 / 12`; verdict `stage42_am_source_level_full_waypoint_eval_pass_positive`。
- test rows: `47458`; domains: TrajNet `37918`, UCY `9540`; full-waypoint rows: `32056`。
- protected full-waypoint ADE all/t50/t100raw/hard: `0.245788` / `0.220171` / `0.143652` / `0.237494`。
- protected full-waypoint FDE all/t50/t100raw/hard: `0.221325` / `0.222358` / `0.128623` / `0.213338`。
- bootstrap CI low all/t50/t100raw/hard: `0.242554` / `0.215923` / `0.137653` / `0.233887`。
- domain split: TrajNet positive；UCY 在这个 proposed source-level test 里仍是 fallback-only。
- 结论：full-waypoint source-level evidence 在当前 HEAD 下仍为正，但仍是 protected dataset-local/raw-frame 2.5D，不是 metric/seconds、true 3D、Stage5C 或 SMC。
- 验证：focused pytest `3 passed`；full pytest `.venv-pytorch/bin/python -m pytest tests -> 1110 passed in 4392.72s (1:13:12)`。
<!-- STAGE42_IT_SOURCE_LEVEL_FULL_WAYPOINT_REFRESH:END -->

<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:START -->
## Stage42-IU Source-Level UCY Full-Waypoint Specialist Integration

- source: `fresh_composition_from_current_stage42_it_and_cached_verified_stage42_v`
- role: closes the Stage42-IT UCY fallback-only source-level weakness by retaining Stage42-IT TrajNet and importing the cached-verified Stage42-V UCY specialist slice.
- gate: `17 / 17`; verdict `stage42_iu_source_level_ucy_full_waypoint_integration_pass`.
- rows: `47458`; domains: TrajNet + UCY.
- weighted ADE all/t50/t100raw/hard: `0.305568` / `0.284549` / `0.195280` / `0.302105`.
- weighted easy degradation: `-0.242171`.
- positive domains all/t50/t100raw/hard: `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']`.
- limitation: no single merged row-cache artifact yet; this is source-level policy-package composition evidence.
- boundary: protected dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:END -->
<!-- STAGE42_IV_SOURCE_LEVEL_ROW_CACHE_INTEGRATION:START -->
## Stage42-IV Source-Level Row-Cache Full-Waypoint Integration

- source: `fresh_run_current_source_level_row_cache_and_cached_verified_stage42v_ucy`
- role: turns the Stage42-IU TrajNet+UCY source-level policy package into a single row-level merged cache with bootstrap.
- gate: `20 / 20`; verdict `stage42_iv_source_level_row_cache_integration_pass`.
- rows: `47458`; domains: `{'TrajNet': 37918, 'UCY': 9540}`.
- ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`.
- easy degradation: `0.000000`.
- bootstrap t50 CI: `[0.242930, 0.251388]`; bootstrap_n `2000`.
- limitation: cache is local and not committed; claims remain dataset-local/raw-frame 2.5D.
- 边界：不是 metric/seconds，不是 true 3D，不是 foundation；Stage5C 未执行，SMC 未启用。
<!-- STAGE42_IV_SOURCE_LEVEL_ROW_CACHE_INTEGRATION:END -->

<!-- STAGE42_IW_ROW_CACHE_MECHANISM_AUDIT:START -->
## Stage42-IW Source-Level Row-Cache Mechanism Audit

- source: `fresh_run_row_cache_mechanism_audit_from_cached_verified_stage42iv_cache`
- role: mechanism audit over the Stage42-IV single merged row-cache, not a new metric-only summary.
- gate: `18 / 18`; verdict `stage42_iw_row_cache_mechanism_audit_pass`.
- rows: `47458`; domain rows: `{'TrajNet': 37918, 'UCY': 9540}`.
- ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`.
- easy degradation: `0.000000`; switch rows `33355`; fallback exact floor rate `1.000000`.
- full-waypoint coverage: `0.675460`; bootstrap t50 CI `[0.242612, 0.251123]`.
- interpretation: safe-switch and teacher/floor protection are directly supported by this row-cache; waypoint labels are sequence-capable but not complete for every row; history/neighbor/goal/interaction still require retrained ablation evidence.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IW_ROW_CACHE_MECHANISM_AUDIT:END -->

<!-- STAGE42_IX_SOURCE_LEVEL_CONTEXT_REPAIR:START -->
## Stage42-IX Source-Level Context Repair Trials

- source: `fresh_run_weighted_floor_residual_context_repair`
- role: retrained repair attempt after Stage42-AO showed context was not incremental after baseline-family rollout features.
- gate: `11 / 12`; verdict `stage42_ix_context_repair_completed_context_not_proven`.
- tested: `6` weighted/floor-residual variants.
- best_trial: `baseline_family_absolute_weighted`; best all/t50/t100raw/hard `0.280381` / `0.317359` / `0.143387` / `0.269583`.
- easy degradation: `-0.311860`.
- positive_context_repair_trials: `[]`.
- context_claim_verdict: `stage42_ix_context_repair_negative_context_still_not_incremental`.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IX_SOURCE_LEVEL_CONTEXT_REPAIR:END -->

<!-- STAGE42_IY_SOURCE_LEVEL_NONLINEAR_CONTEXT_REPAIR:START -->
## Stage42-IY Source-Level Nonlinear Context Repair

- source: `fresh_run_sampled_extra_trees_context_capacity_repair`
- role: nonlinear capacity test after Stage42-IX still failed to make context incremental.
- gate: `12 / 13`; verdict `stage42_iy_nonlinear_context_repair_completed_context_not_proven`.
- trials: `4` ExtraTrees residual models; deterministic train cap `120000`.
- best_trial: `tree_baseline_family_residual`; best all/t50/t100raw/hard `0.221602` / `0.246937` / `0.187483` / `0.232718`.
- easy degradation: `-0.125700`.
- positive_nonlinear_context_trials: `[]`.
- capacity_hypothesis_verdict: `stage42_iy_nonlinear_context_capacity_not_sufficient`.
- boundary: sampled train-only nonlinear repair; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IY_SOURCE_LEVEL_NONLINEAR_CONTEXT_REPAIR:END -->

<!-- STAGE42_IZ_SOURCE_LEVEL_NONLINEAR_CONTEXT_SLICE_AUDIT:START -->
## Stage42-IZ Source-Level Nonlinear Context Slice Audit

- source: `fresh_run_retrained_extra_trees_context_slice_audit`
- role: after Stage42-IY, test whether nonlinear context has only local slice-level utility.
- gate: `11 / 11`; verdict `stage42_iz_context_slice_audit_positive`.
- supported_context_slice_count: `14`.
- decision: `context_has_powered_slice_level_support`.
- blocker_counts: `{'no_powered_positive_context_slice': 0, 'context_below_baseline_family': 55, 'easy_or_safety_not_primary_blocker': 2}`.
- boundary: train-only slice thresholds, validation-selected safe policy, test-once audit; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IZ_SOURCE_LEVEL_NONLINEAR_CONTEXT_SLICE_AUDIT:END -->

<!-- STAGE42_JA_CONTEXT_SLICE_POLICY_PROMOTION:START -->
## Stage42-JA Context-Slice Policy Promotion Audit

- source: `fresh_run_validation_selected_context_slice_policy`
- role: promote Stage42-IZ slice-level context evidence into a validation-selected fallback-safe policy, or reject promotion.
- gate: `10 / 12`; verdict `stage42_ja_context_slice_policy_not_promotable`.
- selected_rule_count: `13`; test_context_rule_coverage_rate `0.977327`.
- context policy all/t50/t100raw/hard/easy: `0.203253` / `0.190761` / `0.107057` / `0.195825` / `-0.211871`.
- delta vs baseline-family all/t50/t100raw/hard/easy: `-0.023421` / `-0.070733` / `-0.084708` / `-0.042885` / `-0.069684`.
- decision: `validation_selected_context_slice_policy_not_promoted`.
- boundary: validation-only slice policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JA_CONTEXT_SLICE_POLICY_PROMOTION:END -->

<!-- STAGE42_JB_CONSERVATIVE_CONTEXT_SLICE_POLICY_REPAIR:START -->
## Stage42-JB Conservative Context-Slice Policy Repair

- source: `fresh_run_validation_greedy_conservative_context_slice_repair`
- role: after Stage42-JA failed, try a stricter validation-greedy, inference-safe, core-preserving context slice repair.
- gate: `11 / 13`; verdict `stage42_jb_conservative_context_policy_not_promotable`.
- selected_rule_count: `4`; test_context_rule_coverage_rate `0.526950`.
- conservative policy all/t50/t100raw/hard/easy: `0.231382` / `0.190761` / `0.191765` / `0.227164` / `-0.220374`.
- delta vs baseline-family all/t50/t100raw/hard/easy: `0.004708` / `-0.070733` / `0.000000` / `-0.011546` / `-0.078187`.
- primary_blocker: `context_policy_has_core_metric_regression`.
- boundary: validation-greedy policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JB_CONSERVATIVE_CONTEXT_SLICE_POLICY_REPAIR:END -->

<!-- STAGE42_JC_LATEST_EVIDENCE_TIER_CONSOLIDATION:START -->
## Stage42-JC Latest Evidence Tier Consolidation

- source: `fresh_stage42_jc_latest_evidence_tier_consolidation`
- gate: `20 / 20`; verdict: `stage42_jc_latest_evidence_tier_consolidation_pass`
- main evidence: `T1_source_level_row_cache_full_waypoint` with all `29.15%`, t50 `24.70%`, t100 raw-frame diagnostic `19.63%`, hard/failure `28.73%`, easy degradation `0.00%`.
- context boundary: Stage42-IZ has `14` local supported context slices, but JA/JB failed promotion, so context is not a deployable/global main contribution.
- claim boundary: still protected dataset-local/raw-frame 2.5D; not true 3D, not foundation, not metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_JC_LATEST_EVIDENCE_TIER_CONSOLIDATION:END -->

<!-- STAGE42_JD_CALIBRATION_READINESS_RECONCILIATION:START -->
## Stage42-JD Calibration Readiness Reconciliation

- source: `fresh_stage42_jd_calibration_readiness_reconciliation`
- gate: `21 / 21`; verdict: `stage42_jd_calibration_readiness_reconciliation_pass`
- required datasets covered: `['aerialmpt', 'eth_ucy', 'opentraj', 'sdd', 'tgsim', 'trajnet', 'ucy']`; direct path groups found `9 / 9`.
- source-specific metric/time candidates: `7`; ready now: `False`.
- conclusion: external validation/full-waypoint work can continue in raw-frame/dataset-local mode, but metric/seconds claims remain blocked until user-confirmed terms, guarded conversion, no-leakage, and restricted evaluation.
- Stage5C not executed; SMC not enabled.
<!-- STAGE42_JD_CALIBRATION_READINESS_RECONCILIATION:END -->

<!-- STAGE42_JE_SOURCE_ROTATION_FULL_WAYPOINT_EVAL:START -->
## Stage42-JE Source-Rotation Full-Waypoint Evaluation

- source: `fresh_stage42_je_source_rotation_full_waypoint_eval`
- gate: `14 / 14`; verdict: `stage42_je_source_rotation_full_waypoint_eval_pass`
- held-out domain rotations: ETH_UCY: all 25.23%, t50 21.07%, hard 26.08%, easy 27.83%; TrajNet: all 30.11%, t50 39.29%, hard 29.21%, easy -24.27%; UCY: all 21.86%, t50 23.73%, hard 20.19%, easy -21.09%.
- decision: `source_rotation_positive_but_not_global_deployable`; deployable held-out domains: `['TrajNet', 'UCY']`.
- boundary: this is stricter cross-domain raw-frame evidence; it does not change the no-metric/no-seconds/no-Stage5C/no-SMC boundary.
<!-- STAGE42_JE_SOURCE_ROTATION_FULL_WAYPOINT_EVAL:END -->

<!-- STAGE42_JF_SOURCE_ROTATION_EASY_GUARD_REPAIR:START -->
## Stage42-JF Source-Rotation Easy-Guard Repair

- source: `fresh_stage42_jf_source_rotation_easy_guard_repair`
- gate: `9 / 9`; verdict: `stage42_jf_source_rotation_easy_guard_repair_pass`
- held-out easy-guard rotations: ETH_UCY: cap 1.00, all 25.23%, t50 21.07%, hard 26.08%, easy 27.83%; TrajNet: cap 0.75, all 30.13%, t50 39.29%, hard 29.19%, easy -25.02%; UCY: cap 0.75, all 21.86%, t50 23.73%, hard 20.19%, easy -21.09%.
- decision: `easy_guard_repair_partial_domain_bounded`; deployable domains after easy guard: `['TrajNet', 'UCY']`; still blocked: `['ETH_UCY']`.
- boundary: validation-only switch budget; no test threshold tuning, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JF_SOURCE_ROTATION_EASY_GUARD_REPAIR:END -->

<!-- STAGE42_JG_ETH_UCY_SOURCE_SPECIFIC_EASY_GUARD:START -->
## Stage42-JG ETH_UCY Source-Specific Easy-Guard Feasibility

- source: `fresh_stage42_jg_eth_ucy_source_specific_easy_guard`
- gate: `11 / 11`; verdict: `stage42_jg_eth_ucy_source_specific_easy_guard_pass`
- source-CV folds: ETH/seq_eth/obsmat.txt: all 0.58%, t50 -32.47%, hard 0.63%, easy -11.79%; ETH/seq_hotel/obsmat.txt: all 8.64%, t50 15.05%, hard 8.70%, easy -15.89%; UCY/students03/obsmat.txt: all 8.73%, t50 9.39%, hard 10.24%, easy 19.42%; UCY/zara01/obsmat.txt: all 12.50%, t50 17.97%, hard 11.43%, easy -24.69%; UCY/zara02/obsmat.txt: all 27.54%, t50 36.18%, hard 28.92%, easy 81.62%.
- decision: `eth_ucy_source_specific_policy_partial_source_support`; deployable sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt']`; blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt', 'UCY/zara02/obsmat.txt']`.
- boundary: this is ETH_UCY source-specific support only, not cross-domain zero-shot success; still no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JG_ETH_UCY_SOURCE_SPECIFIC_EASY_GUARD:END -->

<!-- STAGE42_JH_ETH_UCY_HARM_AWARE_SOURCE_GUARD:START -->
## Stage42-JH ETH_UCY Harm-Aware Source Guard

- source: `fresh_stage42_jh_eth_ucy_harm_aware_source_guard`
- gate: `9 / 9`; verdict: `stage42_jh_eth_ucy_harm_aware_source_guard_pass`
- source-CV harm-aware folds: ETH/seq_eth/obsmat.txt: all 0.58%, t50 -32.47%, hard 0.63%, easy -11.82%; ETH/seq_hotel/obsmat.txt: all 8.64%, t50 15.05%, hard 8.70%, easy -15.89%; UCY/students03/obsmat.txt: all 9.09%, t50 9.03%, hard 10.02%, easy 10.78%; UCY/zara01/obsmat.txt: all 12.50%, t50 17.97%, hard 11.43%, easy -24.69%; UCY/zara02/obsmat.txt: all 30.39%, t50 38.99%, hard 30.27%, easy -2.52%.
- decision: `eth_ucy_harm_aware_guard_partial_support`; deployable sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt', 'UCY/zara02/obsmat.txt']`; blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; easy repaired: `['UCY/zara02/obsmat.txt']`.
- boundary: this is ETH_UCY source-specific support only, not global/cross-domain success; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JH_ETH_UCY_HARM_AWARE_SOURCE_GUARD:END -->

<!-- STAGE42_JI_ETH_UCY_SOURCE_ROBUST_BLOCKED_REPAIR:START -->
## Stage42-JI ETH_UCY Source-Robust Blocked-Source Repair

- source: `fresh_stage42_ji_eth_ucy_source_robust_blocked_repair`
- gate: `10 / 10`; verdict: `stage42_ji_eth_ucy_source_robust_blocked_repair_pass`
- targets from JH blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repair folds: ETH/seq_eth/obsmat.txt: all 0.97%, t50 -31.92%, hard 1.05%, easy -14.48%, deployable=False; UCY/students03/obsmat.txt: all 5.42%, t50 3.69%, hard 6.23%, easy 7.24%, deployable=False.
- decision: `eth_ucy_blocked_sources_still_blocked`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; easy improved: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: held-out sources still blocked remain fallback-only; no global ETH_UCY/cross-domain overclaim, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JI_ETH_UCY_SOURCE_ROBUST_BLOCKED_REPAIR:END -->

<!-- STAGE42_JJ_ETH_UCY_BLOCKED_SOURCE_GEOMETRY_SUPPORT:START -->
## Stage42-JJ ETH_UCY Blocked-Source Geometry/Family Support

- source: `fresh_stage42_jj_eth_ucy_blocked_source_geometry_support`
- gate: `11 / 11`; verdict: `stage42_jj_eth_ucy_blocked_source_geometry_support_pass`
- family/geometry support audit: ETH/seq_eth/obsmat.txt: static all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, family-oracle t50 53.80%, deployable=False; UCY/students03/obsmat.txt: static all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, family-oracle t50 39.14%, deployable=False.
- decision: `blocked_sources_not_repaired_family_support_diagnostic`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: static causal family support does not globally repair ETH_UCY; blocked sources stay fallback-only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JJ_ETH_UCY_BLOCKED_SOURCE_GEOMETRY_SUPPORT:END -->

<!-- STAGE42_JK_ETH_UCY_ROW_FAMILY_SELECTOR:START -->
## Stage42-JK ETH_UCY Row-Level Family Selector

- source: `fresh_stage42_jk_eth_ucy_row_family_selector`
- gate: `11 / 11`; verdict: `stage42_jk_eth_ucy_row_family_selector_pass`
- row-family heldout results: ETH/seq_eth/obsmat.txt: all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 53.80%, deployable=False; UCY/students03/obsmat.txt: all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 39.14%, deployable=False.
- decision: `row_family_selector_not_deployable_on_blocked_sources`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: no full ETH_UCY/cross-domain overclaim; still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JK_ETH_UCY_ROW_FAMILY_SELECTOR:END -->

<!-- STAGE42_JL_ETH_UCY_SOURCE_SUPPORT_COVERAGE:START -->
## Stage42-JL ETH_UCY Source Support Coverage

- source: `fresh_stage42_jl_eth_ucy_source_support_coverage`
- gate: `11 / 11`; verdict: `stage42_jl_eth_ucy_source_support_coverage_pass`
- source-support heldout results: ETH/seq_eth/obsmat.txt: support=True, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 53.80%; UCY/students03/obsmat.txt: support=False, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 39.14%.
- decision: `source_support_policy_not_deployable_support_blocker`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; unsupported: `['UCY/students03/obsmat.txt']`.
- boundary: this is a source-support diagnostic/repair attempt, still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JL_ETH_UCY_SOURCE_SUPPORT_COVERAGE:END -->

<!-- STAGE42_JM_ETH_UCY_CALIBRATED_SUPPORT_RECHECK:START -->
## Stage42-JM ETH_UCY Calibrated Support Recheck

- source: `fresh_stage42_jm_eth_ucy_calibrated_support_recheck`
- gate: `11 / 11`; verdict: `stage42_jm_eth_ucy_calibrated_support_recheck_pass`
- calibrated-support heldout results: ETH/seq_eth/obsmat.txt: local_calib=source_specific_annotation_step_meter_coordinate_evidence, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, deployable=False; UCY/students03/obsmat.txt: local_calib=source_specific_annotation_step_meter_coordinate_evidence, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, deployable=False.
- decision: `calibrated_support_recheck_blocked_no_safe_deployment`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: source-specific calibration evidence is recorded, but the main claim remains dataset-local/raw-frame 2.5D; no global metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JM_ETH_UCY_CALIBRATED_SUPPORT_RECHECK:END -->

<!-- STAGE42_JN_LOCAL_CALIBRATED_SOURCE_SUPPORT_INTAKE:START -->
## Stage42-JN Local Calibrated Source Support Intake

- source: `fresh_stage42_jn_local_calibrated_source_support_intake`
- gate: `12 / 12`; verdict: `stage42_jn_local_calibrated_source_support_intake_pass`
- parseable support candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; long-horizon candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- decision: `candidate_sources_found_but_user_terms_required`; auto_convert_allowed: `[]`.
- boundary: candidate-source intake only; no conversion, no deployment claim, no global metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JN_LOCAL_CALIBRATED_SOURCE_SUPPORT_INTAKE:END -->

<!-- STAGE42_JO_LOCAL_CALIBRATED_SOURCE_GUARDED_CONVERSION_PREFLIGHT:START -->
## Stage42-JO Local Calibrated Source Guarded Conversion Preflight

- source: `fresh_stage42_jo_local_calibrated_source_guarded_conversion_preflight`
- gate: `13 / 13`; verdict: `stage42_jo_local_calibrated_source_guarded_preflight_pass`
- technical_ready_after_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; conversion_allowed_now: `[]`.
- decision: `guarded_conversion_preflight_blocked_pending_user_terms`; blocked_by_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- boundary: preflight only; no conversion, no deployable source-support claim, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JO_LOCAL_CALIBRATED_SOURCE_GUARDED_CONVERSION_PREFLIGHT:END -->

<!-- STAGE42_JP_LOCAL_CALIBRATED_SOURCE_TERMS_PREFILL:START -->
## Stage42-JP Local Calibrated Source Terms Prefill

- source: `fresh_stage42_jp_local_calibrated_source_terms_prefill`
- gate: `15 / 15`; verdict: `stage42_jp_local_calibrated_source_terms_prefill_pass`
- official_hint_rows: `3`; license_found_rows: `1`; conversion_ready_now: `0`.
- high_confidence_official_source_rows: `['Wild-Track']`; manual_only_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- boundary: terms prefill only; no permission, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JP_LOCAL_CALIBRATED_SOURCE_TERMS_PREFILL:END -->

<!-- STAGE42_JQ_LOCAL_CALIBRATED_SOURCE_TERMS_VALIDATION:START -->
## Stage42-JQ Local Calibrated Source Terms Validation

- source: `fresh_stage42_jq_local_calibrated_source_terms_validator`
- gate: `14 / 14`; verdict: `stage42_jq_local_calibrated_source_terms_validation_pass`
- datasets_validated: `3`; terms_accepted_rows: `0`; conversion_ready_rows: `0`.
- blocked_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; ready_for_future_guarded_conversion: `[]`.
- boundary: user terms validator only; no download, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JQ_LOCAL_CALIBRATED_SOURCE_TERMS_VALIDATION:END -->

<!-- STAGE42_JR_SOURCE_CONTEXT_FRESH_REPLAY:START -->
## Stage42-JR Source Context Fresh Replay

- source: `fresh_stage42_jr_source_context_fresh_replay`
- gate: `12 / 12`; verdict: `stage42_jr_source_context_negative_evidence_pass`
- baseline-family all/t50/hard remains positive: `0.2878` / `0.3154` / `0.2758`.
- sequence context did not add lift: best all/t50/hard delta `-0.0245` / `-0.0831` / `-0.0284`.
- graph context did not add lift: best all/t50/hard delta `-0.0230` / `-0.0858` / `-0.0262`.
- boundary: negative result preserved; no sequence/graph independent main claim, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JR_SOURCE_CONTEXT_FRESH_REPLAY:END -->

<!-- STAGE42_JS_SOURCE_CONTEXT_GAIN_HARM_CLOSURE:START -->
## Stage42-JS Source Context Gain/Harm Closure

- source: `fresh_stage42_js_source_context_gain_harm_closure`
- gate: `14 / 14`; verdict: `stage42_js_source_context_gain_harm_closure_pass`
- narrow horizon positives: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`; these are not t50/t100 main-claim evidence.
- t50 blocker: `router_under_switches_despite_headroom` with oracle headroom `0.0352`; IQ repair t50 `0.000001`, IR repair t50 `0.000000`.
- t100 blocker: `weak_predictive_signal_or_baseline_family_dominance` with oracle headroom `0.0112`.
- decision: close the current source-level sequence/graph gain-harm candidate family for t50/t100 independent contribution; next work needs new candidate policies or row/source-slice objectives.
- boundary: raw-frame/dataset-local 2.5D only; no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JS_SOURCE_CONTEXT_GAIN_HARM_CLOSURE:END -->

<!-- STAGE42_JT_CURRENT_MODULE_CLAIM_REFRESH:START -->
## Stage42-JT Current Module Claim Refresh

- source: `fresh_stage42_jt_current_module_claim_refresh`
- gate: `15 / 15`; verdict: `stage42_jt_current_module_claim_refresh_pass`
- row-cache ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`; easy `0.000000`.
- AO standalone context variants: `['history_only', 'motion_goal_context']`; incremental after baseline-family: `[]`.
- blocked independent claims: `['incremental_context_after_baseline_family', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'sequence_graph_t50_t100_independent_main_claim', 'JEPA_downstream_main_claim', 'Transformer_independent_main_claim', 'ungated_full_waypoint_deployment', 'metric_seconds_or_true3d_claim']`.
- decision: current paper wording should center protected row-cache/full-waypoint + safe-switch/teacher-floor; keep scene/goal, neighbor/interaction, JEPA, Transformer, and sequence/graph t50/t100 as blocked or auxiliary.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JT_CURRENT_MODULE_CLAIM_REFRESH:END -->

<!-- STAGE42_JU_CURRENT_REVIEWER_REPLAY_PACKAGE:START -->
## Stage42-JU Current Reviewer Replay Package

- source: `fresh_stage42_ju_current_reviewer_replay_package`
- gate: `17 / 17`; verdict: `stage42_ju_current_reviewer_replay_package_pass`.
- replay commands: `outputs/stage42_long_research/current_reviewer_replay_commands_stage42.sh`.
- row-cache ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`.
- current package locks the latest claim boundary: protected source-level full-waypoint row-cache + safe-switch/floor is supported; independent scene/goal, neighbor/interaction, JEPA, Transformer, ungated, metric/time, true-3D and foundation claims remain blocked.
- public README remains a human project introduction; detailed replay/provenance stays in internal result files.
<!-- STAGE42_JU_CURRENT_REVIEWER_REPLAY_PACKAGE:END -->

<!-- STAGE42_JV_SOURCE_SLICE_EVIDENCE_MATRIX:START -->
## Stage42-JV Source Slice Evidence Matrix

- source: `fresh_stage42_jv_source_slice_evidence_matrix_from_cached_verified_row_cache`
- gate: `18 / 18`; verdict: `stage42_jv_source_slice_evidence_matrix_pass`.
- cache rows/domains/source-files: `47458` / `2` / `3`.
- all-slice ADE/FDE improvement: `0.291543` / `0.278634`; easy degradation `0.000000`.
- domain metrics available for: `['TrajNet', 'UCY']`; horizon metrics available for: `['10', '100', '25', '50']`.
- this strengthens the paper evidence table by decomposing protected row-cache/full-waypoint evidence across domain, horizon, source-file, hard/easy, switch/fallback, and waypoint-completeness slices.
- boundary remains dataset-local/raw-frame 2.5D; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_JV_SOURCE_SLICE_EVIDENCE_MATRIX:END -->

<!-- STAGE42_JW_TEACHER_FLOOR_NECESSITY_SLICE_AUDIT:START -->
## Stage42-JW Teacher Floor Necessity Slice Audit

- source: `fresh_stage42_jw_teacher_floor_necessity_slice_audit`
- gate: `14 / 14`; verdict: `stage42_jw_teacher_floor_necessity_slice_audit_pass`.
- switch/fallback rows: `33355` / `14103`; fallback exact floor rate `1.000000`.
- hard/failure switch rate `0.729644` vs easy switch rate `0.412616`.
- guarded t50 relaxation safety: `True` with t50 `0.289698`.
- decision: keep the teacher/floor globally; only guarded t50 relaxation is supported, and floor-free neural deployment remains forbidden.
- boundary remains dataset-local/raw-frame 2.5D; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_JW_TEACHER_FLOOR_NECESSITY_SLICE_AUDIT:END -->

<!-- STAGE42_JX_CURRENT_PAPER_EVIDENCE_REFRESH:START -->
## Stage42-JX Current Paper Evidence Refresh

- source: `fresh_stage42_jx_current_paper_evidence_refresh`
- gate: `15 / 15`; verdict: `stage42_jx_current_paper_evidence_refresh_pass`.
- current evidence rows/domains/horizons: `47458` / `['TrajNet', 'UCY']` / `['10', '25', '50', '100']`.
- ADE all/t50/t100raw/hard: `29.15%` / `24.70%` / `19.63%` / `28.73%`; easy `0.00%`.
- teacher/floor necessity: fallback rows `14103`, exact-floor rate `1.000000`, global floor-free neural deployable `False`.
- README-facing decision: public GitHub README stays project-owner style; detailed staged evidence remains internal.
- paper boundary: protected dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_JX_CURRENT_PAPER_EVIDENCE_REFRESH:END -->

<!-- STAGE42_JY_CONTEXT_MATERIALITY_BY_SOURCE_SLICE:START -->
## Stage42-JY Context Materiality By Source Slice

- source: `fresh_stage42_jy_context_materiality_by_source_slice`
- gate: `14 / 14`; verdict: `stage42_jy_context_materiality_by_source_slice_pass`.
- baseline-family control remains dominant: all/t50/hard `28.78%` / `31.54%` / `27.58%`.
- material global incremental context variants: `[]`.
- best narrow context slice signal: `{'variant': 'motion_goal_context', 'slice': 'horizon=10', 'metric': 'all_improvement', 'delta': 0.02748739379455012}`.
- decision: keep independent scene/goal/neighbor/interaction as blocked main claims; next context attempt must use source/horizon-slice objectives rather than repeating the closed protocol.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_JY_CONTEXT_MATERIALITY_BY_SOURCE_SLICE:END -->

<!-- STAGE42_KA_CONTEXT_SOURCE_HORIZON_OBJECTIVE_CONTRACT:START -->
## Stage42-KA Context Source/Horizon Objective Contract

- source: `fresh_stage42_ka_context_source_horizon_objective_contract`
- gate: `15 / 15`; verdict: `stage42_ka_context_source_horizon_objective_contract_pass`.
- global material context variants over baseline-family control: `[]`.
- narrow auxiliary context slices preserved for future source/horizon training: `[{'variant': 'history_only', 'horizon': 10}, {'variant': 'motion_goal_context', 'horizon': 10}]`.
- diagnostic router conflicts: `[{'horizon': 25, 'candidate': 'baseline_plus_history_goal_neighbor', 'decision': 'diagnostic_router_only_not_baseline_family_positive'}]`.
- t50 blocker: `router_under_switches_despite_headroom`; t50 oracle headroom `3.52%`.
- t100 blocker: `weak_predictive_signal_or_baseline_family_dominance`; t100 raw oracle headroom `1.12%`.
- decision: do not promote scene/goal/neighbor/interaction as independent global main claims; next context attempt must use row-level source/horizon objectives under Stage37/teacher floor.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_KA_CONTEXT_SOURCE_HORIZON_OBJECTIVE_CONTRACT:END -->

<!-- STAGE42_KB_T50_ROW_LEVEL_CONTEXT_OBJECTIVE:START -->
## Stage42-KB t50 Row-Level Context Objective

- source: `fresh_stage42_kb_t50_row_level_context_objective`
- gate: `12 / 12`; verdict: `stage42_kb_t50_row_level_context_objective_pass`.
- best trial: `baseline_plus_history` with `context_only` and margin `0.0`.
- t50/all/hard/easy vs baseline-family: `0.00%` / `0.00%` / `0.00%` / `-0.00%`.
- deployable_increment_supported: `False`; reason: `validation_safe_policy_under_switches`.
- boundary: validation-selected t50 row-level experiment only; raw-frame/dataset-local 2.5D, no metric/seconds, no true-3D/foundation, no Stage5C, no SMC.
<!-- STAGE42_KB_T50_ROW_LEVEL_CONTEXT_OBJECTIVE:END -->

<!-- STAGE43_A_SAFETY_FLOOR_REPLAY:START -->
## STAGE43_A_SAFETY_FLOOR_REPLAY

source = `fresh_stage43_a_safety_floor_replay`
verdict = `stage43_a_safety_floor_replay_pass`
gate = `14 / 14`
latent_state_training_precondition = `True`

Stage43-A freezes the safety floor before any latent-state model training. Historical floors are cached-verified and hashed: Stage26 SDD selector, Stage37 external t50 selector, and M3W-Neural v1 protected composite. The current Stage42 source/domain full-waypoint protected policy is replayed fresh from the row cache.

Fresh Stage42 replay: all `0.291543`, t50 `0.247045`, t100 raw-frame diagnostic `0.196335`, hard/failure `0.287273`, easy degradation `0.000000`, fallback exact floor rate `1.000000`.

No Stage5C execution, no SMC, no metric/seconds/true-3D/foundation claim. Future endpoints/waypoints remain labels only.
<!-- STAGE43_A_SAFETY_FLOOR_REPLAY:END -->

<!-- STAGE43_B_LATENT_STATE_DATASET_CONTRACT:START -->
## STAGE43_B_LATENT_STATE_DATASET_CONTRACT

source = `fresh_stage43_b_latent_state_dataset_contract`
verdict = `stage43_b_latent_state_dataset_contract_pass`
gate = `12 / 12`
endpoint_latent_state_training_ready = `True`
full_waypoint_supervised_training_ready = `True`

Stage43-B builds the latent-state dataset contract from Stage35/36/37 external geometry/history/goal/baseline artifacts and the Stage42 source-level full-waypoint cache. It separates inference tokens from labels: future endpoint/waypoint labels are loss/eval only and are not model inputs.

Endpoint/failure/gain/harm/occupancy and full-waypoint supervised latent-state training are now contract-ready under the frozen local Stage43 source-level supervision cache. No Stage5C/SMC/metric/seconds/true-3D/foundation claim is made.
<!-- STAGE43_B_LATENT_STATE_DATASET_CONTRACT:END -->

<!-- STAGE43_C_PROTECTED_LATENT_STATE_SMALL:START -->
## STAGE43_C_PROTECTED_LATENT_STATE_SMALL

source = `fresh_stage43_c_protected_latent_state_small`
verdict = `stage43_c_protected_latent_state_candidate_pass`
gate = `8 / 8`
deploy_neural = `True`

Stage43-C trains a real PyTorch protected latent-state head on the Stage43 contract. Inputs are causal/current-or-past only; future endpoint/full-waypoint labels remain loss/eval only. The model learns z_t and z_t -> z_{t+h}, plus endpoint/failure/gain/harm/occupancy heads, then evaluates only through a safety-floor fallback policy.

Protected eval vs floor: all `0.177665`, t50 `0.137515`, t100 raw diagnostic `0.018234`, hard/failure `0.181572`, easy degradation `0.000000`.

This is not Stage5C, not SMC, not metric/seconds-level, not true 3D, and not a foundation model.
<!-- STAGE43_C_PROTECTED_LATENT_STATE_SMALL:END -->

<!-- STAGE43_D_LATENT_STATE_ROBUSTNESS_AUDIT:START -->
## STAGE43_D_LATENT_STATE_ROBUSTNESS_AUDIT

source = `fresh_stage43_d_latent_state_robustness_audit`
verdict = `stage43_d_latent_state_robustness_partial`
gate = `8 / 9`
multi_domain_claim_allowed = `False`

Stage43-D re-evaluates the Stage43-C protected latent-state checkpoint on the full held-out UCY test split and adds bootstrap confidence intervals. This is a robustness audit, not a new threshold-tuning run and not a Stage5C/SMC execution.

Full UCY test metrics: all `0.177665`, t50 `0.137515`, t100 raw diagnostic `0.018234`, hard/failure `0.181572`, easy degradation `0.000000`, switch rate `0.176500`.

Bootstrap CI lows: all `0.157468`, t50 `0.104683`, hard/failure `0.164075`, easy CI high `0.000000`.

Scope limitation: this proves UCY held-out dataset-local/raw-frame robustness only; multi-domain robustness remains a next gate.
<!-- STAGE43_D_LATENT_STATE_ROBUSTNESS_AUDIT:END -->

<!-- STAGE43_E_MULTIDOMAIN_LATENT_EVAL:START -->
## STAGE43_E_MULTIDOMAIN_LATENT_EVAL

source = `fresh_stage43_e_multidomain_latent_eval`
verdict = `stage43_e_multidomain_latent_eval_blocker_mapped`
gate = `8 / 8`
multi_domain_latent_candidate = `False`

Stage43-E evaluates the Stage43 protected latent-state checkpoint across the currently available train/val/test domains. It confirms UCY heldout support but refuses a multi-domain claim because ETH_UCY and TrajNet are not present as held-out test domains in the current Stage43 split.

UCY heldout: all `0.163151`, t50 `0.136820`, t100 raw diagnostic `0.009722`, hard/failure `0.164765`, easy degradation `0.000000`.

Missing heldout domains for a real multi-domain latent claim: `['ETH_UCY', 'TrajNet']`. Next required step is a source-level or scene-level split containing ETH_UCY, TrajNet, and UCY as held-out domains without test endpoint goal leakage.
<!-- STAGE43_E_MULTIDOMAIN_LATENT_EVAL:END -->

<!-- STAGE43_F_SOURCE_LEVEL_HELDOUT_SPLIT:START -->
## STAGE43_F_SOURCE_LEVEL_HELDOUT_SPLIT

source = `fresh_stage43_f_source_level_heldout_split`
verdict = `stage43_f_source_level_split_ready`
gate = `11 / 11`

Stage43-F builds the source-file-level heldout split required by Stage43-E. It reuses the existing Stage35/36/37 external artifacts as a data pool, but creates a new split manifest where ETH_UCY, TrajNet, and UCY all appear in test through disjoint source files.

Pool rows `337991`, domains `{'ETH_UCY': 150798, 'TrajNet': 120890, 'UCY': 66303}`, row hash `9c8b4d51e0f7a1618dce410c7dd23fbf7f21da5de587d4ae021257775164c3c5`.

New split rows: train `146809`, val `101446`, test `89736`. Test domains `['ETH_UCY', 'TrajNet', 'UCY']`.

Important boundary: this is not a new model result. The old Stage43-C checkpoint remains UCY-heldout evidence only; a new Stage43 latent model must be trained/evaluated on this split before any multi-domain latent claim is allowed.
<!-- STAGE43_F_SOURCE_LEVEL_HELDOUT_SPLIT:END -->

<!-- STAGE43_G_SOURCE_LEVEL_PROTECTED_LATENT:START -->
## STAGE43_G_SOURCE_LEVEL_PROTECTED_LATENT

source = `fresh_stage43_g_source_level_protected_latent`
verdict = `stage43_g_source_level_latent_candidate_pass`
gate = `10 / 10`
deploy_neural = `True`

Stage43-G trains a fresh protected latent-state model on the Stage43-F source-file-level split, where ETH_UCY, TrajNet, and UCY all appear in held-out test through disjoint source files. This replaces the earlier UCY-only checkpoint for multi-domain evaluation.

Protected test metrics vs floor: all `0.858018`, t50 `0.821362`, t100 raw diagnostic `0.783976`, hard/failure `0.866818`, easy degradation `0.000000`.

Safety note: test switch rate is `1.000000` and fallback rate is `0.000000`. This means the full split result needs bootstrap and safety-stress confirmation before it can replace the frozen floor as a deployment policy.

This remains dataset-local/raw-frame 2.5D evidence. Stage5C and SMC are disabled; no metric/seconds/true-3D/foundation claim is made.
<!-- STAGE43_G_SOURCE_LEVEL_PROTECTED_LATENT:END -->

<!-- STAGE43_H_SOURCE_LEVEL_LATENT_ROBUSTNESS:START -->
## STAGE43_H_SOURCE_LEVEL_LATENT_ROBUSTNESS

source = `fresh_stage43_h_source_level_latent_robustness`
verdict = `stage43_h_unit_consistent_audit_failed_keep_floor`
gate = `9 / 10`
deploy_stage43_g = `False`

Stage43-H audits Stage43-G and finds a unit-consistency issue: the Stage43-G headline compared normalized neural delta error against dataset-local floor FDE. After multiplying neural error by each row's scale, all/t50/hard remain positive but easy degradation becomes unsafe.

Unit-consistent metrics: all `0.351410`, t50 `0.158059`, t100 raw diagnostic `0.004466`, hard/failure `0.377402`, easy degradation `1.597489`.

Conclusion: keep the frozen Stage37/Stage42 safety floor. Stage43-G is a useful neural dynamics signal but not a deployable replacement until a calibrated safe-switch repair passes unit-consistent easy/proximity gates.
<!-- STAGE43_H_SOURCE_LEVEL_LATENT_ROBUSTNESS:END -->

<!-- STAGE43_I_UNIT_CONSISTENT_SAFE_SWITCH:START -->
## STAGE43_I_UNIT_CONSISTENT_SAFE_SWITCH

source = `fresh_stage43_i_unit_consistent_safe_switch`
verdict = `stage43_i_unit_consistent_safe_switch_pass`
gate = `13 / 13`
deploy_stage43_i_candidate = `True`

Stage43-I repairs the Stage43-G unit-consistent easy-harm failure by adding a fixed prior easy-risk guard (`stage35_easy_prob <= 0.03`) and conservative source/domain switch caps before allowing the source-level latent endpoint to replace the frozen floor. The policy is treated as a conservative safety-family repair, not as a test-selected threshold sweep.

Unit-consistent safe-switch metrics: all `0.231071`, t50 `0.113648`, t100 raw diagnostic `0.013513`, hard/failure `0.244058`, easy degradation `0.000000`, switch rate `0.185255`.

Bootstrap CI lows: all `0.228439`, t50 `0.110331`, hard/failure `0.239911`; easy CI high `0.000000`.

Source caveat: worst source all improvement is `-0.001034`, so this is not a uniform per-source claim.

This is still protected dataset-local/raw-frame 2.5D evidence. It is not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.
<!-- STAGE43_I_UNIT_CONSISTENT_SAFE_SWITCH:END -->

<!-- STAGE43_J_SOURCE_LEVEL_CAVEAT_AUDIT:START -->
## STAGE43_J_SOURCE_LEVEL_CAVEAT_AUDIT

source = `fresh_stage43_j_source_level_caveat_audit`
verdict = `stage43_j_source_level_caveat_mapped`
gate = `7 / 7`
source_uniform_candidate = `False`
domain_level_candidate = `True`

Stage43-J audits the Stage43-I source-level slices and blocks a uniform per-source claim. Stage43-I remains a unit-consistent domain-level protected latent candidate, but one small TrajNet source is slightly negative and multiple source t50 slices remain floor-only.

Worst source `83b0417df499ccae`: all `-0.001034`, t50 `0.000000`, easy degradation `0.000000`.

Next allowed repair: source-family gate or source-balanced retraining selected on train/validation only. Forbidden: disabling a test source by source id or tuning thresholds from test source metrics.
<!-- STAGE43_J_SOURCE_LEVEL_CAVEAT_AUDIT:END -->

<!-- STAGE43_K_SOURCE_SLICE_REPAIR:START -->
## STAGE43_K_SOURCE_SLICE_REPAIR

source = `fresh_stage43_k_source_slice_repair`
verdict = `stage43_k_source_slice_negative_repaired`
gate = `12 / 12`
source_safe_candidate = `True`
uniform_positive_source_candidate = `False`

Stage43-K addresses the Stage43-J source-level caveat without test-source threshold tuning. It starts from Stage43-I's unit-consistent safe switch and adds a validation-only source-family guard: source families unsupported or unsafe on validation are floored. This removes negative source-slice harm, but it is not a uniform positive per-source claim.

Metrics: all `0.231096`, t50 `0.113648`, t100 raw diagnostic `0.013513`, hard/failure `0.244058`, easy degradation `0.000000`, switch rate `0.185199`.

Bootstrap CI lows: all `0.228620`, t50 `0.108802`, hard/failure `0.240428`; easy CI high `0.000000`.

Source safety: negative source count `0`, min source all improvement `0.000000`, max source easy degradation `0.009491`.
Blocked test source families under the validation-only guard: `TrajNet_mot`.

Claim boundary remains: protected dataset-local/raw-frame 2.5D evidence only; no metric/seconds claim, no Stage5C execution, no SMC.
<!-- STAGE43_K_SOURCE_SLICE_REPAIR:END -->

<!-- STAGE43_L_FULL_WAYPOINT_SUPERVISION_CACHE:START -->
## STAGE43_L_FULL_WAYPOINT_SUPERVISION_CACHE

source = `fresh_stage43_l_full_waypoint_supervision_cache`
verdict = `stage43_l_full_waypoint_supervision_cache_pass`
gate = `10 / 10`
full_waypoint_supervised_training_ready = `True`

Stage43-L freezes source-level train/val/test full-waypoint supervision labels under the Stage43 source split. The cache is local and intentionally not committed. This closes the Stage43-B blocker for supervised full-waypoint latent-state training while keeping future waypoints as labels/eval only.

Rows: train `146809`, val `101446`, test `89736`. Test full-waypoint rows `89736`; source overlaps train/val/test `{'train_val': 0, 'train_test': 0, 'val_test': 0}`.

No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
<!-- STAGE43_L_FULL_WAYPOINT_SUPERVISION_CACHE:END -->

<!-- STAGE43_M_FULL_WAYPOINT_LATENT_DYNAMICS:START -->
## Stage43-M protected full-waypoint latent dynamics

Result source: `fresh_run`. A torch latent dynamics head was trained on the frozen Stage43-L full-waypoint supervision cache, with future waypoints used only as labels/eval and with the frozen protected floor kept as the deployment guard.

- mode: `small`
- gate: `11 / 11`
- verdict: `stage43_m_protected_full_waypoint_latent_candidate_pass`
- deploy neural full-waypoint head: `True`
- full-waypoint ADE improvement vs floor: `37.23%`
- t50 full-waypoint ADE improvement vs floor: `32.94%`
- hard/failure full-waypoint ADE improvement vs floor: `38.77%`
- easy degradation: `0.00%`
- t50 bootstrap CI: `[31.46%, 34.39%]`

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.
<!-- STAGE43_M_FULL_WAYPOINT_LATENT_DYNAMICS:END -->

<!-- STAGE43_N_FULL_WAYPOINT_LATENT_ROBUSTNESS:START -->
## Stage43-N full-waypoint latent robustness audit

Result source: `fresh_full_test_replay_from_stage43_m_checkpoint`. The Stage43-M checkpoint and validation-selected protected policy were replayed on the full Stage43-L test cache, then broken down by domain, horizon, source, and scene.

- gate: `12 / 12`
- verdict: `stage43_n_full_test_positive_with_source_t100_blockers`
- full-test rows: `89736`
- full-waypoint ADE improvement vs floor: `36.74%`
- t50 full-waypoint ADE improvement vs floor: `33.30%`
- t100 raw-frame diagnostic: `-27.88%`
- hard/failure ADE improvement vs floor: `38.45%`
- easy degradation: `0.47%`
- negative source count: `1`

Boundary: this supports a protected full-test latent dynamics candidate with t100 and source-level caveats; it is still dataset-local/raw-frame 2.5D only, with no metric/seconds-level claim, no Stage5C, and no SMC.
<!-- STAGE43_N_FULL_WAYPOINT_LATENT_ROBUSTNESS:END -->

<!-- STAGE43_O_FULL_WAYPOINT_LATENT_SAFE_REPAIR:START -->
## Stage43-O full-waypoint latent safe repair

Result source: `fresh_validation_only_safe_repair_from_stage43_m_checkpoint`. This step keeps the Stage43-M latent model frozen and uses validation-only source-family/horizon support to repair the Stage43-N negative source and t100 harm.

- gate: `12 / 12`
- verdict: `stage43_o_safe_repair_pass_t100_fallback_not_positive`
- full-test rows: `89736`
- full-waypoint ADE improvement vs floor: `39.22%`
- t50 full-waypoint ADE improvement vs floor: `30.06%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure ADE improvement vs floor: `39.85%`
- easy degradation: `0.00%`
- negative source count after repair: `0`

Boundary: Stage43-O is a safety repair. It removes negative-source and t100 harm by fallback, but t100 is not a positive success and some source families are fallback-only. The result remains dataset-local/raw-frame 2.5D evidence with no metric/seconds-level claim, no Stage5C, and no SMC.
<!-- STAGE43_O_FULL_WAYPOINT_LATENT_SAFE_REPAIR:END -->

<!-- STAGE43_P_TAIL_HORIZON_WAYPOINT_ADAPTER:START -->
## Stage43-P tail-horizon full-waypoint adapter

Result source: `fresh_train_val_selected_tail_horizon_adapter`. A train-split ridge full-waypoint adapter was trained on tail horizons and selected on validation only, then tested once against the Stage43-O safe repair floor.

- gate: `13 / 13`
- verdict: `stage43_p_tail_horizon_adapter_pass_t100_still_fallback`
- full-test rows: `89736`
- full-waypoint ADE improvement vs floor: `50.25%`
- t50 full-waypoint ADE improvement vs floor: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure ADE improvement vs floor: `47.88%`
- easy degradation: `0.00%`
- t50 bootstrap CI: `[50.76%, 51.74%]`

Boundary: this is a stronger protected tail-horizon full-waypoint adapter, but t100 remains fallback-only rather than positive. The result remains dataset-local/raw-frame 2.5D evidence with no metric/seconds-level claim, no Stage5C, and no SMC.
<!-- STAGE43_P_TAIL_HORIZON_WAYPOINT_ADAPTER:END -->

<!-- STAGE43_Q_T100_GUARDED_TRIAL:START -->
## Stage43-Q t100 guarded trial

Result source: `fresh_validation_selected_t100_guarded_trial`. This is a validation-selected h100 add-on trial over the Stage43-P safety floor; the test set is used only once for confirmation.

- gate: `11 / 11`
- verdict: `stage43_q_t100_guarded_trial_honest_blocker`
- t100 status: `honest_blocker_no_t100_deployment`
- t100 blocker: `validation_positive_h100_did_not_generalize_to_test_safely`
- allowed h100 rules: `UCY|100`
- full-waypoint ADE improvement vs floor: `50.25%`
- t50 full-waypoint ADE improvement vs floor: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- t100 delta vs Stage43-P: `0.00%`
- rejected h100 candidate t100 delta vs Stage43-P: `-2.22%`
- easy degradation: `0.00%`

Boundary: Stage43-Q does not execute Stage5C or SMC, does not make metric/seconds claims, and does not deploy h100 unless the validation-selected h100 rule is test-safe.
<!-- STAGE43_Q_T100_GUARDED_TRIAL:END -->

<!-- STAGE43_R_T100_SOURCE_STABILITY_GUARD:START -->
## Stage43-R t100 source-stability guard

Result source: `fresh_validation_source_stable_t100_guard`. This adds a validation-only source-stability requirement for h100 deployment over the Stage43-P safety floor.

- gate: `13 / 13`
- verdict: `stage43_r_source_stable_h100_guard_blocks_t100_false_positive`
- h100 status: `h100_blocked_insufficient_source_stability`
- h100 allowed rules: `none`
- blocks Stage43-Q false positive: `True`
- full-waypoint ADE improvement vs floor: `50.25%`
- t50 full-waypoint ADE improvement vs floor: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- easy degradation: `0.00%`

Boundary: t100 remains fallback-only. The h100 blocker is now localized to insufficient source-stable validation evidence; no metric/seconds claim, Stage5C, or SMC.
<!-- STAGE43_R_T100_SOURCE_STABILITY_GUARD:END -->

<!-- STAGE43_S_T100_SOURCE_COVERAGE_PREFLIGHT:START -->
## Stage43-S t100 source coverage preflight

Result source: `fresh_h100_source_coverage_preflight`. This audits h100 source coverage and prepares a source-stable split preflight without rewriting caches or tuning test thresholds.

- gate: `8 / 8`
- verdict: `stage43_s_t100_source_coverage_preflight_pass`
- feasible h100 families: `TrajNet_crowds`
- blocked h100 families: `ETH_UCY, TrajNet_biwi, UCY`
- rebuild source-stable h100 split recommended: `True`
- uniform t100 blocker remains: `True`

Boundary: this is a preflight audit, not a new t100 deployment. It keeps the Stage43-P/R fallback and preserves the dataset-local/raw-frame 2.5D claim boundary.
<!-- STAGE43_S_T100_SOURCE_COVERAGE_PREFLIGHT:END -->

<!-- STAGE43_T_T100_SOURCE_STABLE_SPECIALIST:START -->
## Stage43-T source-stable h100 specialist

Result source: `fresh_source_stable_trajnet_crowds_h100_specialist`. This trains a source-stable h100 specialist only for the feasible TrajNet_crowds family, using source-level train/val/test split from Stage43-S.

- gate: `12 / 12`
- verdict: `stage43_t_source_stable_h100_specialist_deployable`
- positive h100 dynamics signal: `True`
- validation source safe: `True`
- easy safe: `True`
- deployed: `True`
- held-out h100 ADE improvement: `2.59%`
- held-out h100 easy degradation: `0.00%`
- deployment ADE improvement: `2.59%`

Boundary: this is a source-stable h100 family trial, not a uniform t100 claim. Deployment is allowed only when validation-source safety and held-out easy preservation both pass; otherwise Stage43-P/R remain the safety floor and t100 remains fallback-only.
<!-- STAGE43_T_T100_SOURCE_STABLE_SPECIALIST:END -->

<!-- STAGE43_U_INTEGRATED_TAIL_H100_POLICY:START -->
## Stage43-U integrated tail + h100 policy

Result source: `fresh_integrated_stage43_p_tail_adapter_plus_stage43_t_h100_specialist`. This composes Stage43-P with the Stage43-T source-stable h100 specialist without changing test thresholds.

- gate: `15 / 15`
- verdict: `stage43_u_integrated_tail_h100_policy_pass_family_limited`
- deployed: `True`
- full-waypoint ADE improvement vs floor: `50.28%`
- t50 full-waypoint ADE improvement vs floor: `51.23%`
- t100 raw-frame diagnostic: `0.18%`
- hard/failure ADE improvement vs floor: `47.91%`
- easy degradation: `0.00%`
- h100 source-slice ADE lift: `2.59%`
- h100 source-slice endpoint FDE lift: `-0.55%`
- all delta vs Stage43-P: `0.03%`

Boundary: this is a family-limited h100 full-waypoint ADE improvement integrated into the protected policy. It is not a uniform t100 solution, not endpoint-FDE success, not metric/seconds-level, not Stage5C, and not SMC.
<!-- STAGE43_U_INTEGRATED_TAIL_H100_POLICY:END -->

<!-- STAGE43_V_WORLD_STATE_HEAD_AUDIT:START -->
## Stage43-V world-state head audit

Result source: `fresh_checkpoint_replay_world_state_head_audit`. I replayed the Stage43-M latent checkpoint and audited failure/gain/harm/density/validity heads on the test split without test threshold tuning.

- gate: `9 / 9`
- verdict: `stage43_v_world_state_head_audit_partial`
- informative binary heads: `failure, gain, harm`
- failure AUROC/AUPRC: `0.8648` / `0.7901`
- gain AUROC/AUPRC: `0.8737` / `0.9215`
- harm AUROC/AUPRC: `0.9047` / `0.8891`
- density R2/corr: `-0.5639` / `0.2055`
- latent mean variance: `0.482653`
- physical validity head deployable: `False` (no explicit training loss yet)

Boundary: this is an auxiliary world-state head audit, not a Stage5C/SMC/generative rollout. Physical validity remains a gap because the current checkpoint exposes a validity logit but did not train it with a dedicated loss.
<!-- STAGE43_V_WORLD_STATE_HEAD_AUDIT:END -->

<!-- STAGE43_W_AUXILIARY_HEAD_REPAIR:START -->
## Stage43-W auxiliary density/validity head repair

Result source: `fresh_train_val_selected_auxiliary_head_repair`. I froze the Stage43-M latent checkpoint and trained train/val-selected ridge calibrators for the weak density and waypoint-validity proxy heads.

- gate: `10 / 10`
- verdict: `stage43_w_density_proxy_repaired_validity_proxy_diagnostic`
- density feature set: `latent_heads_causal_x`
- density R2 before -> after: `-0.5639` -> `0.8178`
- density corr after: `0.9252`
- validity proxy R2 before -> after: `-2.5067` -> `0.9223`
- deploy density proxy head: `True`
- true physical validity claim: `False`

Boundary: this repairs a causal history-density proxy head from frozen latent/context features. It is not a future occupancy claim. The validity head remains a label-availability proxy, not a true physical-validity certificate.
<!-- STAGE43_W_AUXILIARY_HEAD_REPAIR:END -->

<!-- STAGE43_X_INTERACTION_VALIDITY_PROXY:START -->
## Stage43-X interaction / validity proxy head audit

Result source: `fresh_future_label_proxy_head_audit`. I froze the Stage43-M latent checkpoint and trained train/val-selected proxy heads for future-proximity interaction risk and waypoint smoothness/validity.

- gate: `10 / 10`
- verdict: `stage43_x_interaction_proxy_signal_validity_proxy_diagnostic`
- interaction feature set: `causal_x`
- interaction AUROC/AUPRC: `0.7694` / `0.3254`
- interaction positive rate: `0.1349`
- smoothness proxy R2/corr: `0.9216` / `0.9617`
- deploy interaction risk proxy head: `True`
- true physical validity claim: `False`

Boundary: future waypoints are labels/evaluation only, never inference inputs. Interaction risk is a future-proximity proxy, not human interaction annotation; smoothness/validity remains diagnostic, not true physical validity.
<!-- STAGE43_X_INTERACTION_VALIDITY_PROXY:END -->

<!-- STAGE43_Y_MULTIMODAL_LATENT_HEAD_SUITE:START -->
## Stage43-Y multimodal latent head suite

Result source: `fresh_consolidated_stage43_vwx_head_suite`. I consolidated Stage43-V/W/X into a single head-suite contract for the protected multimodal latent-state candidate.

- gate: `12 / 12`
- verdict: `stage43_y_protected_multimodal_latent_head_suite_candidate`
- latent min/mean variance: `0.108561` / `0.482653`
- failure/gain/harm AUROC: `0.8648` / `0.8737` / `0.9047`
- density proxy R2/corr: `0.8178` / `0.9252`
- interaction proxy AUROC/AUPRC: `0.7694` / `0.3254`
- protected multimodal latent state candidate: `True`
- standalone ungated deployment: `False`

Boundary: this consolidates deployable proxy heads under the existing safety floor. It does not create a true physical-validity certificate, future occupancy claim, metric/seconds-level result, Stage5C execution, or SMC.
<!-- STAGE43_Y_MULTIMODAL_LATENT_HEAD_SUITE:END -->

<!-- STAGE43_Z_LATENT_TOKEN_SCHEMA_COVERAGE:START -->
## STAGE43_Z_LATENT_TOKEN_SCHEMA_COVERAGE

source = `fresh_stage43_z_latent_token_schema_coverage`
verdict = `stage43_z_latent_token_schema_coverage_pass`
gate = `12 / 12`
feature_schema_hash = `fba36ccddae43a4776793fb92ef305162abf4649f632f0d9696a463bac31022b`
train_row_hash = `92421511e9e27416567537326bfc9b6f9bf469d3e81f8788b8bebd4d5072fae0`

Stage43-Z audits what the current protected latent-state cache actually covers. It confirms causal agent/history, goal-prototype, baseline-rollout, safety-floor, horizon/domain, density, interaction-risk proxy, and failure/gain/harm heads are represented under row/schema hashes.

It also records the limits: there is still no explicit scene image/raster token, no SDF token, no full all-agent graph tensor, no future occupancy label, no human-gold interaction label, and no true physical-validity label. These are gaps, not hidden successes. The current claim remains protected dataset-local/raw-frame 2.5D multimodal latent-state evidence, not true 3D or foundation modeling.
<!-- STAGE43_Z_LATENT_TOKEN_SCHEMA_COVERAGE:END -->

<!-- STAGE43_AA_SCENE_RASTER_PROXY_TOKENS:START -->
## STAGE43_AA_SCENE_RASTER_PROXY_TOKENS

source = `fresh_stage43_aa_scene_raster_proxy_tokens`
verdict = `stage43_aa_scene_raster_proxy_tokens_pass`
gate = `10 / 10`
source_proxy_hash = `c56451af9f0be2fb8720e67111d7fb39e58014dbfc358794ff4e1e584acc1cc5`
manifest_sha256 = `87dc956e45a682b3a1c6bb8397e3c393745dc5f1ea44edb57a9d76f87693dfec`

Stage43-AA fills the explicit scene/raster/SDF-token gap with a train-only proxy: source/domain route bounds, route occupancy grids, boundary-SDF proxy, density prior, and scene-agnostic goal-vector priors. It writes row-aligned auxiliary features for train/val/test and records row/feature hashes.

This is still a proxy, not raw scene imagery, not annotated walkable geometry, and not verified metric SDF. It is not yet retrained into Stage43-M; it is an auxiliary scene/raster token cache for the next latent-state training step. No future endpoints, future waypoints, central velocity, or test endpoint goals are used.
<!-- STAGE43_AA_SCENE_RASTER_PROXY_TOKENS:END -->

<!-- STAGE43_AB_SCENE_PROXY_AUGMENTED_LATENT_DYNAMICS:START -->
## STAGE43_AB_SCENE_PROXY_AUGMENTED_LATENT_DYNAMICS

source = `fresh_stage43_ab_scene_proxy_augmented_latent_dynamics`
verdict = `stage43_ab_scene_proxy_augmented_latent_lift_candidate`
gate = `11 / 11`
deploy_scene_proxy_augmented_neural = `True`
scene_proxy_lift_over_stage43_m = `True`

full_waypoint_ade_vs_floor = `38.97%`; delta_vs_stage43_m = `9.20%`
t50_full_waypoint_ade_vs_floor = `35.42%`; delta_vs_stage43_m = `18.97%`
hard_failure_vs_floor = `39.66%`; delta_vs_stage43_m = `10.91%`
easy_degradation = `0.14%`

Stage43-AB retrains the full-waypoint latent dynamics head with the Stage43-AA train-only scene/raster proxy features appended to the causal input. It compares against Stage43-M and only promotes the augmented model if it improves Stage43-M while preserving easy cases.

Boundary unchanged: scene proxy is not raw image/SDF, future labels are loss/eval only, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AB_SCENE_PROXY_AUGMENTED_LATENT_DYNAMICS:END -->

<!-- STAGE43_AC_SCENE_PROXY_GUARDED_LATENT_POLICY:START -->
## STAGE43_AC_SCENE_PROXY_GUARDED_LATENT_POLICY

source = `fresh_stage43_ac_scene_proxy_guarded_latent_policy`
verdict = `stage43_ac_guarded_scene_proxy_latent_candidate`
gate = `11 / 11`
deploy_guarded_scene_proxy_latent = `True`

full_waypoint_ade_vs_floor = `41.17%`; delta_vs_stage43_m = `11.40%`
t50_full_waypoint_ade_vs_floor = `35.42%`; delta_vs_stage43_m = `18.97%`
hard_failure_vs_floor = `42.34%`; delta_vs_stage43_m = `13.58%`
t100_raw_frame_diagnostic = `-17.79%`; delta_vs_stage43_m = `0.00%`
easy_degradation = `0.00%`
scene_proxy_override_rate = `80.12%`

Stage43-AC is the guarded deployment version of Stage43-AB. It keeps the scene-proxy latent head where validation says it helps, but falls back to Stage43-M on risky slices, especially raw-frame t100.

Boundary unchanged: scene proxy is not raw image/SDF, future labels are loss/eval only, t100 is raw-frame diagnostic, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AC_SCENE_PROXY_GUARDED_LATENT_POLICY:END -->

<!-- STAGE43_AD_SCENE_PROXY_GUARDED_ROBUSTNESS_AUDIT:START -->
## STAGE43_AD_SCENE_PROXY_GUARDED_ROBUSTNESS_AUDIT

source = `fresh_stage43_ad_scene_proxy_guarded_robustness_audit`
verdict = `stage43_ad_guarded_scene_proxy_caveated_audit_pass`
gate = `12 / 12`
all_powered_domains_positive = `False`

full_waypoint_ade_vs_floor = `41.17%`; delta_vs_stage43_m = `11.40%`
t50_full_waypoint_ade_vs_floor = `35.42%`; delta_vs_stage43_m = `18.97%`
hard_failure_vs_floor = `42.34%`; delta_vs_stage43_m = `13.58%`
t100_raw_frame_diagnostic = `-17.79%`; delta_vs_stage43_m = `0.00%`
easy_degradation = `0.00%`
caveat_slice_count = `9`

Stage43-AD audits the guarded Stage43-AC policy by domain, source, horizon, hard/failure, and easy slices. It records caveats instead of turning the average gain into a uniform success claim.

Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AD_SCENE_PROXY_GUARDED_ROBUSTNESS_AUDIT:END -->

<!-- STAGE43_AE_SCENE_PROXY_SLICE_SAFE_POLICY:START -->
## STAGE43_AE_SCENE_PROXY_SLICE_SAFE_POLICY

source = `fresh_stage43_ae_scene_proxy_slice_safe_policy`
verdict = `stage43_ae_slice_safe_scene_proxy_candidate`
gate = `14 / 14`
deploy_slice_safe_scene_proxy = `True`

full_waypoint_ade_vs_floor = `23.95%`; delta_vs_stage43_m = `-5.81%`
t50_full_waypoint_ade_vs_floor = `37.16%`; delta_vs_stage43_m = `20.71%`
hard_failure_vs_floor = `23.38%`; delta_vs_stage43_m = `-5.38%`
t100_raw_frame_diagnostic = `0.00%`
max_domain_easy_degradation = `0.00%`
max_horizon_easy_degradation = `0.00%`

Stage43-AE is the slice-safe repair after the AD caveat audit. It uses a validation-selected three-route policy over floor, Stage43-M, and Stage43-AB, so weak easy slices can fall all the way back to the floor rather than only to Stage43-M.

Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 remains diagnostic; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AE_SCENE_PROXY_SLICE_SAFE_POLICY:END -->

<!-- STAGE43_AF_SCENE_PROXY_COUNTERFACTUAL_ABLATION:START -->
## STAGE43_AF_SCENE_PROXY_COUNTERFACTUAL_ABLATION

source = `fresh_stage43_af_scene_proxy_counterfactual_ablation`
verdict = `stage43_af_scene_proxy_counterfactual_contribution_pass`
gate = `12 / 12`
scene_proxy_counterfactual_contribution_supported = `True`

actual_slice_safe_all = `23.95%`
actual_slice_safe_t50 = `37.16%`
scene_proxy_t50_contribution_vs_stage43_m_counterfactual = `20.71%`
scene_proxy_t50_endpoint_contribution_vs_stage43_m_counterfactual = `15.81%`
scene_proxy_hard_contribution_vs_stage43_m_counterfactual = `9.23%`
actual_easy_degradation = `0.00%`

Stage43-AF uses the same Stage43-AE route and replaces only the scene-proxy AB branch with a no-scene Stage43-M counterfactual. This gives a direct model-family contribution estimate for scene/goal proxy latent features under the same safety contract.

Boundary unchanged: same-route counterfactual, not full factorial retraining; dataset-local/raw-frame 2.5D only; t100 remains diagnostic; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AF_SCENE_PROXY_COUNTERFACTUAL_ABLATION:END -->

<!-- STAGE43_AG_SCENE_PROXY_RETRAINED_ABLATION:START -->
## STAGE43_AG_SCENE_PROXY_RETRAINED_ABLATION

source = `fresh_stage43_ag_scene_proxy_retrained_ablation`
verdict = `stage43_ag_scene_proxy_retrained_ablation_pass`
gate = `11 / 11`
scene_proxy_retrained_ablation_supports_contribution = `True`
best_t50_variant = `full_scene`
best_t50_delta_vs_retrained_no_scene = `5.79%`
best_safe_t50_variant = `geometry_route`
best_safe_t50_delta_vs_retrained_no_scene = `5.02%`
best_hard_variant = `full_scene`

Stage43-AG fresh-trains no-scene, geometry/route, goal-only, and full-scene proxy variants under the same protected full-waypoint latent dynamics protocol. This is a focused retrained scene-proxy subset ablation, not a full all-module factorial ablation. The report separates raw-best t50 from safety-preserving t50 evidence.

Boundary unchanged: dataset-local/raw-frame 2.5D only; scene proxy is not raw image/SDF; future labels are supervision/eval only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AG_SCENE_PROXY_RETRAINED_ABLATION:END -->

<!-- STAGE43_AH_FEATURE_FAMILY_RETRAINED_ABLATION:START -->
## STAGE43_AH_FEATURE_FAMILY_RETRAINED_ABLATION

source = `fresh_stage43_ah_feature_family_retrained_ablation`
verdict = `stage43_ah_feature_family_retrained_ablation_pass`
gate = `12 / 12`
feature_family_retrained_ablation_supports_modules = `True`
positive_t50_contribution_variants = `['no_goal', 'no_neighbor_interaction', 'no_baseline_floor', 'no_domain']`
positive_hard_or_all_contribution_variants = `['no_goal', 'no_neighbor_interaction', 'no_baseline_floor', 'no_domain']`

Stage43-AH fresh-trains full_features plus no_history, no_goal, no_neighbor_interaction, no_baseline_floor, and no_domain variants. This moves Stage43 causal ablation evidence beyond inference masking, while still remaining a focused single-seed/small retrained ablation rather than a complete factorial study. It is contribution evidence, not a deployment policy: positive contribution can coexist with unsafe easy harm, and some removed-family variants outperform full_features in this small run.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are supervision/eval only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_AH_FEATURE_FAMILY_RETRAINED_ABLATION:END -->

<!-- STAGE43_AI_FEATURE_FAMILY_MULTISEED_CONFIRMATION:START -->
## STAGE43_AI_FEATURE_FAMILY_MULTISEED_CONFIRMATION

source = `fresh_stage43_ai_feature_family_multiseed_confirmation`
verdict = `stage43_ai_feature_family_multiseed_confirmation_pass`
gate = `8 / 8`
seeds = `[431, 443, 457]`
stable_positive_t50_contribution_variants = `['no_neighbor_interaction', 'no_baseline_floor', 'no_domain']`
stable_positive_hard_or_all_contribution_variants = `['no_goal', 'no_baseline_floor', 'no_domain']`

Stage43-AI repeats the Stage43-AH retrained feature-family ablation across multiple seeds. It tests whether baseline/floor, goal, history, neighbor/interaction, and domain feature-family contributions survive seed variation. Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AI_FEATURE_FAMILY_MULTISEED_CONFIRMATION:END -->

<!-- STAGE43_AJ_SAFETY_FLOOR_NECESSITY_AUDIT:START -->
## STAGE43_AJ_SAFETY_FLOOR_NECESSITY_AUDIT

source = `fresh_stage43_aj_safety_floor_necessity_audit`
verdict = `stage43_aj_safety_floor_necessity_confirmed`
gate = `13 / 13`
floor_necessity_confirmed = `True`
protected_easy_vs_ungated_easy = `0.00%` vs `7.86%`
no_baseline_floor_t50_delta_mean = `11.12%`
latest_stage43_p_t50 = `51.23%`
latest_stage43_p_easy = `0.00%`
latest_stage43_p_t100 = `0.00%`

Stage43-AJ consolidates current Stage43 floor evidence: protected-vs-ungated neural dynamics, multi-seed no-baseline-floor ablation, self/conformal safety gates, bounded residual relaxation, scene-proxy safe-vs-unsafe variants, and the latest tail-horizon adapter. Conclusion: the safety floor is currently a core safety mechanism and cannot be globally removed; only validation-selected partial relaxation is supported.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AJ_SAFETY_FLOOR_NECESSITY_AUDIT:END -->

<!-- STAGE43_AK_SELF_GATE_CONFORMAL_AUDIT:START -->
## STAGE43_AK_SELF_GATE_CONFORMAL_AUDIT

source = `fresh_stage43_ak_self_gate_conformal_audit`
result_source = `fresh_replay_and_audit_over_frozen_stage43_m_checkpoint`
verdict = `stage43_ak_self_gate_conformal_audit_pass`
gate = `12 / 12`
stored_policy_replay_max_abs_diff = `0.00000000`
stored_self_gate_all_t50_easy = `29.77%` / `16.45%` / `0.00%`
ungated_easy_t100 = `55.72%` / `-72.12%`
conformal_style_all_t50_t100_easy = `32.41%` / `16.45%` / `0.00%` / `0.00%`

Stage43-AK replayed the frozen Stage43-M checkpoint and policy, then compared stored self-gate, fresh self-gate search, ungated neural deployment, and a validation-calibrated conformal-style h100/easy guard. The audit keeps the global safety floor: ungated neural remains unsafe, while guarded policies preserve easy cases.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AK_SELF_GATE_CONFORMAL_AUDIT:END -->

<!-- STAGE43_AL_BOUNDED_RESIDUAL_SAFETY_AUDIT:START -->
## STAGE43_AL_BOUNDED_RESIDUAL_SAFETY_AUDIT

source = `fresh_stage43_al_bounded_residual_safety_audit`
result_source = `fresh_bounded_residual_audit_over_frozen_stage43_m_checkpoint`
verdict = `stage43_al_bounded_residual_candidate_pass`
gate = `12 / 12`
deploy_bounded_residual = `True`
safe_bounded_all_t50_t100_hard_easy = `38.00%` / `26.96%` / `0.00%` / `37.71%` / `0.00%`
safe_minus_stored_all_t50_t100_hard_easy = `8.23%` / `10.52%` / `17.79%` / `8.96%` / `0.00%`

Stage43-AL tested bounded residual relaxation over the frozen Stage43-M latent waypoint model. Residual deltas are norm-clipped and validation-selected, with future labels used only for validation/eval. If the bounded residual does not beat the stored hard switch while preserving easy/t100 safety, it remains diagnostic and the Stage43-M floor policy stays active.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AL_BOUNDED_RESIDUAL_SAFETY_AUDIT:END -->

<!-- STAGE43_AM_BOUNDED_RESIDUAL_STATISTICAL_CONFIRMATION:START -->
## STAGE43_AM_BOUNDED_RESIDUAL_STATISTICAL_CONFIRMATION

source = `fresh_stage43_am_bounded_residual_statistical_confirmation`
result_source = `fresh_bootstrap_confirmation_over_frozen_stage43_al_candidate`
verdict = `stage43_am_bounded_residual_statistically_confirmed`
gate = `12 / 12`
bounded_residual_statistically_confirmed = `True`
bootstrap_n = `2000`
all_delta_ci = `[7.78%, 8.68%]`
t50_delta_ci = `[9.86%, 11.21%]`
hard_failure_delta_ci = `[8.47%, 9.45%]`
easy_degradation_bounded_ci = `[0.00%, 0.00%]`

Stage43-AM bootstrap-confirms the Stage43-AL bounded residual candidate against the stored Stage43-M hard-switch policy on frozen rows. The candidate remains floor-protected and h100-guarded; this is not global floor removal.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AM_BOUNDED_RESIDUAL_STATISTICAL_CONFIRMATION:END -->

<!-- STAGE43_AN_BOUNDED_RESIDUAL_POLICY_FREEZE:START -->
## STAGE43_AN_BOUNDED_RESIDUAL_POLICY_FREEZE

source = `fresh_stage43_an_bounded_residual_policy_freeze`
result_source = `fresh_freeze_from_statistically_confirmed_stage43_am_candidate`
verdict = `stage43_an_bounded_residual_policy_frozen`
gate = `11 / 11`
policy_frozen = `True`
policy_hash = `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
frozen_all_t50_t100_hard_easy = `38.00%` / `26.96%` / `0.00%` / `37.71%` / `0.00%`

Stage43-AN freezes the statistically confirmed Stage43 bounded-residual latent waypoint policy into a reproducible artifact with policy/config/checkpoint/report/row hashes. It remains floor-protected and h100-guarded; this is not global floor removal.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AN_BOUNDED_RESIDUAL_POLICY_FREEZE:END -->

<!-- STAGE43_AO_BOUNDED_RESIDUAL_REVIEWER_REPLAY:START -->
## STAGE43_AO_BOUNDED_RESIDUAL_REVIEWER_REPLAY

source = `fresh_stage43_ao_bounded_residual_reviewer_replay`
result_source = `fresh_replay_from_frozen_policy_artifact`
verdict = `stage43_ao_bounded_residual_reviewer_replay_pass`
gate = `11 / 11`
reviewer_replay_passed = `True`
policy_hash = `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
policy_hash_match = `True`
replay_max_abs_diff = `0.00000000`
replayed_all_t50_t100_hard_easy = `38.00%` / `26.96%` / `0.00%` / `37.71%` / `0.00%`

Stage43-AO independently replays the frozen bounded-residual policy artifact and verifies policy hash, checkpoint/report hashes, row hashes, replay diff, and no-leakage boundaries. This makes the Stage43 bounded-residual policy reviewer-replayable rather than report-only.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AO_BOUNDED_RESIDUAL_REVIEWER_REPLAY:END -->

<!-- STAGE43_AP_PAPER_EVIDENCE_REFRESH:START -->
## STAGE43_AP_PAPER_EVIDENCE_REFRESH

source = `fresh_stage43_ap_paper_evidence_refresh`
result_source = `fresh_paper_evidence_refresh_from_stage43_aj_to_ao_plus_stage43_p`
verdict = `stage43_ap_paper_evidence_refresh_pass`
gate = `10 / 10`
paper_evidence_refreshed = `True`
policy_hash = `03497313f878a1ec69fd7d2824842fee0acfa79c38dc9d667c6d6ac53ef4c331`
frozen_replayable_policy_hash = `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
current_all_t50_t100_hard_easy = `50.25%` / `51.23%` / `0.00%` / `47.88%` / `0.00%`
latest_t50_ci = `[50.76%, 51.74%]`
latest_vs_frozen_all_t50_hard_delta = `12.25%` / `24.27%` / `10.16%`

Stage43-AP consolidates AJ-AO plus Stage43-P evidence into paper-facing claim boundaries, evidence table, and A-journal gap refresh. The strongest allowed claim is a floor-protected, validation-selected tail-horizon full-waypoint adapter in dataset-local/raw-frame 2.5D space; the frozen bounded-residual policy remains the exact reviewer-replayable safety artifact.

Boundary unchanged: not true 3D; not foundation; no metric/seconds claim; no Stage5C; no SMC.
<!-- STAGE43_AP_PAPER_EVIDENCE_REFRESH:END -->

<!-- STAGE43_AQ_INTEGRATED_CANDIDATE_GATE:START -->
## STAGE43_AQ_INTEGRATED_CANDIDATE_GATE

source = `fresh_stage43_aq_integrated_candidate_gate`
result_source = `fresh_integrated_manifest_from_stage43_aj_to_ap_plus_stage43_p_artifacts`
verdict = `stage43_aq_integrated_protected_latent_state_candidate_pass`
gate = `18 / 18`
policy_hash = `03497313f878a1ec69fd7d2824842fee0acfa79c38dc9d667c6d6ac53ef4c331`
frozen_replayable_policy_hash = `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`
current_candidate_supported = `True`
long_objective_complete = `False`
current_all_t50_t100_hard_easy = `50.25%` / `51.23%` / `0.00%` / `47.88%` / `0.00%`

Stage43-AQ integrates AJ-AO/AP plus Stage43-P into one current candidate manifest and world-model gate. The current strongest deployable evidence is the Stage43-P protected tail-horizon full-waypoint adapter under the Stage37/teacher safety floor; the frozen Stage43 bounded-residual policy remains the exact replayable safety artifact. This is a protected dataset-local/raw-frame 2.5D latent waypoint candidate, not a true 3D/foundation/metric/seconds-level model.

Boundary unchanged: Stage5C is not executed; SMC is not enabled; global floor removal is not supported; the long Stage43 objective remains active.
<!-- STAGE43_AQ_INTEGRATED_CANDIDATE_GATE:END -->

<!-- STAGE43_AR_FULL_SUITE_REPLAY_AUDIT:START -->
## STAGE43_AR_FULL_SUITE_REPLAY_AUDIT

source = `fresh_stage43_ar_full_suite_replay_audit`
result_source = `fresh_full_test_suite_replay_reparsed_from_existing_capture`
verdict = `stage43_ar_full_suite_replay_pass`
gate = `9 / 9`
full_suite_replay_passed = `True`
pytest_summary = `1360 passed in 3750.05s`
wall_seconds = `3750.94`

Stage43-AR records a fresh full test-suite replay using the active arm64 Python runtime. It is a reproducibility/software health audit only; it does not change model claims, execute Stage5C, enable SMC, or create metric/seconds/true-3D/foundation evidence.
<!-- STAGE43_AR_FULL_SUITE_REPLAY_AUDIT:END -->

<!-- STAGE43_AS_DATA_CALIBRATION_REFRESH:START -->
## STAGE43_AS_DATA_CALIBRATION_REFRESH

source = `fresh_stage43_as_data_calibration_refresh`
result_source = `fresh_local_path_audit_plus_cached_verified_stage42_time_geometry`
verdict = `stage43_as_data_calibration_refresh_pass`
gate = `10 / 10`
external_domains_ready = `opentraj, eth_ucy, trajnet, ucy`
global_metric_claim_allowed = `False`
global_seconds_claim_allowed = `False`

Stage43-AS refreshes the data/calibration state by rerunning local path audits for SDD, OpenTraj, ETH/UCY, TrajNet, UCY, TGSIM, and AerialMPT, then reconciling with Stage42-BN source time/geometry evidence. The result keeps global M3W in raw-frame/dataset-local 2.5D language while preserving source-specific ETH/UCY calibration candidates for separately gated future work.

No training, no auto-download, no Stage5C, no SMC, and no metric/seconds/true-3D/foundation claim.
<!-- STAGE43_AS_DATA_CALIBRATION_REFRESH:END -->

<!-- STAGE43_AT_EXTERNAL_VALIDATION_MATRIX:START -->
Stage43-AT builds a fresh external validation matrix from verified Stage43 artifacts. Gate: `13 / 13` with verdict `stage43_at_external_validation_matrix_pass`.

It compares the safety floor, M3W-Neural v1, ungated source-level neural dynamics, domain-capped protected neural, source-family guarded repair, protected full-waypoint dynamics, frozen bounded-residual replay, and the latest tail-horizon full-waypoint adapter. The practical boundary is unchanged: ungated neural is still not deployable, source-family repair is safe but not uniformly positive per source, and every deployable learned candidate remains protected by the floor.

Frozen integrated candidate: all `38.00%`, t50 `26.96%`, t100 raw-frame diagnostic `0.00%`, hard/failure `37.71%`, easy degradation `0.00%`.
Latest protected tail adapter: all `50.25%`, t50 `51.23%`, t100 raw-frame diagnostic `0.00%`, hard/failure `47.88%`, easy degradation `0.00%`.
Source-safe protected neural repair: all `23.11%`, t50 `11.36%`, hard/failure `24.41%`, easy degradation `0.00%`.

This is still dataset-local/raw-frame 2.5D evidence. It is not true 3D, not foundation-scale, not metric/seconds-level, and it does not execute Stage5C or SMC.
<!-- STAGE43_AT_EXTERNAL_VALIDATION_MATRIX:END -->

<!-- STAGE43_AU_DOMAIN_FAILURE_REPAIR:START -->
Stage43-AU runs a bounded validation-only repair attempt for the weak external t50 slices exposed by Stage43-AT. Gate: `12 / 12` with verdict `stage43_au_domain_failure_repair_attempt_pass`.

Selected trial `all_guard0.05_eth0.25_traj0.25` test metrics: all `27.28%`, t50 `13.79%`, t100 raw-frame diagnostic `0.48%`, hard/failure `28.89%`, easy degradation `0.30%`.
Delta vs Stage43-K source-safe repair: all `4.17%`, t50 `2.42%`, hard/failure `4.49%`, easy degradation `0.30%`.

Deployment is not upgraded by this repair attempt. The selected validation policy improves aggregate/t50 over Stage43-K, but per-domain easy harm and remaining weak t50/t100/source slices make it unsafe. Stage5C and SMC remain disabled.
<!-- STAGE43_AU_DOMAIN_FAILURE_REPAIR:END -->

<!-- STAGE43_AV_SOURCE_HORIZON_SAFETY_ENVELOPE:START -->
Stage43-AV audits the full source/horizon safety envelope of the Stage43-AU bounded repair trials. Gate: `12 / 12` with verdict `stage43_av_source_horizon_safety_envelope_pass`.

Across `30` diagnostic trials, aggregate-safe trial count = `29`, domain-easy-safe trial count = `19`, deployable-like trial count = `6`.
Best t50 diagnostic trial `all_guard0.05_eth0.40_traj0.35` reaches t50 `15.06%` but is not a deployment selection.

The result confirms the next useful work is source/horizon-specific safety modeling rather than another global cap. Stage5C and SMC remain disabled.
<!-- STAGE43_AV_SOURCE_HORIZON_SAFETY_ENVELOPE:END -->

<!-- STAGE43_AW_SOURCE_HORIZON_EXPERT_POLICY:START -->
Stage43-AW turns the Stage43-AV diagnostic into a validation-selected source/horizon expert policy: Stage43-K remains the non-t50 base, and a t50 expert is selected on validation only. Gate: `12 / 12` with verdict `stage43_aw_source_horizon_expert_policy_pass`.

Selected t50 expert `t50_guard0.03_eth0.15_traj0.10` test metrics: all `23.40%`, t50 `12.80%`, t100 raw-frame diagnostic `1.35%`, hard/failure `24.73%`, easy degradation `0.00%`.
Delta vs Stage43-K: all `0.29%`, t50 `1.44%`, hard/failure `0.32%`, easy degradation `0.00%`.

Decision: `candidate_requires_reviewer_replay_before_deployment`. This remains dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain disabled.
<!-- STAGE43_AW_SOURCE_HORIZON_EXPERT_POLICY:END -->

<!-- STAGE43_AX_SOURCE_HORIZON_EXPERT_REPLAY:START -->
Stage43-AX exact-replays the Stage43-AW source/horizon expert artifact without validation reselection or test threshold tuning. Gate: `12 / 12` with verdict `stage43_ax_source_horizon_expert_replay_pass`.

Replay metrics: all `23.40%`, t50 `12.80%`, t100 raw-frame diagnostic `1.35%`, hard/failure `24.73%`, easy degradation `0.00%`.
Replay max metric diff vs AW artifact: `0.0000000000`. Policy hash `824b4be3be967cc96872fa8c627eb141e9859f2a01eb15d92697507441e38f22`, row hash `9d27e3a5fba7583152ed8fb175f21685e989a7be93f23d12c2e4aba36bd1212c`.

Decision: reviewer replay passed = `True`; candidate for deployment update = `True`. This remains dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain disabled.
<!-- STAGE43_AX_SOURCE_HORIZON_EXPERT_REPLAY:END -->

<!-- STAGE43_AY_CURRENT_CANDIDATE_RECONCILIATION:START -->
## STAGE43_AY_CURRENT_CANDIDATE_RECONCILIATION

source = `fresh_stage43_ay_current_candidate_reconciliation`
result_source = `fresh_reconciliation_from_stage43_p_ap_ao_ax_aq`
verdict = `stage43_ay_current_candidate_reconciliation_pass`
gate = `12 / 12`
current_candidate_supported = `True`
long_objective_complete = `False`

performance_leader = `Stage43-P`, all/t50/t100_raw/hard/easy = `50.25%` / `51.23%` / `0.00%` / `47.88%` / `0.00%`
source_horizon_replay_leader = `Stage43-AX`, all/t50/t100_raw/hard/easy = `23.40%` / `12.80%` / `1.35%` / `24.73%` / `0.00%`

Stage43-AY reconciles the current evidence stack: Stage43-P is the aggregate performance leader, Stage43-AX is the source/horizon exact-replay leader, and Stage43-AO remains the frozen reviewer-replayable artifact. These are protected dataset-local/raw-frame 2.5D results, not true 3D, metric, seconds-level, foundation, Stage5C, or SMC claims.
<!-- STAGE43_AY_CURRENT_CANDIDATE_RECONCILIATION:END -->

<!-- STAGE43_AZ_TAIL_ADAPTER_REVIEWER_REPLAY:START -->
## STAGE43_AZ_TAIL_ADAPTER_REVIEWER_REPLAY

source = `fresh_stage43_az_tail_adapter_reviewer_replay`
result_source = `fresh_exact_recompute_replay_from_stage43_p_artifact`
verdict = `stage43_az_tail_adapter_reviewer_replay_pass`
gate = `12 / 12`
policy_hash = `9155067aacf42bc8d8e67745c1cf5e05b729f95a88cf65d33d88b9a06c21484b`
model_hash_match = `True`
replay_max_metric_diff = `0.0000000000`

replayed_all_t50_t100_hard_easy = `50.25%` / `51.23%` / `0.00%` / `47.88%` / `0.00%`

Stage43-AZ recomputes the Stage43-P tail-horizon full-waypoint adapter from the artifact-selected config and allowed rules. It performs no validation reselection and no test threshold tuning. This strengthens Stage43-P from a performance leader into an exact recompute replay artifact while preserving the claim boundary: dataset-local/raw-frame 2.5D only; t100 diagnostic only; no Stage5C; no SMC.
<!-- STAGE43_AZ_TAIL_ADAPTER_REVIEWER_REPLAY:END -->

<!-- STAGE43_BA_TAIL_ADAPTER_SOURCE_BLOCKER_AUDIT:START -->
## STAGE43_BA_TAIL_ADAPTER_SOURCE_BLOCKER_AUDIT

source = `fresh_stage43_ba_tail_adapter_source_blocker_audit`
result_source = `fresh_source_family_blocker_audit_from_stage43_p_and_az`
verdict = `stage43_ba_tail_adapter_source_blocker_audit_pass`
gate = `13 / 13`
positive_sources = `2 / 4`
safe_floor_blocked_sources = `2`
catastrophic_ungated_blocked_sources = `2`
uniform_positive_transfer_claim_allowed = `False`

Stage43-BA audits why the replayed Stage43-P tail adapter cannot be claimed as uniform positive source transfer. The source-level blocker is not hidden: TrajNet_biwi and TrajNet_mot remain floor-only because ungated full-waypoint transfer is catastrophically negative. The safety floor is therefore necessary for these slices.
<!-- STAGE43_BA_TAIL_ADAPTER_SOURCE_BLOCKER_AUDIT:END -->

<!-- STAGE43_BB_BLOCKED_SOURCE_REPAIR_FEASIBILITY:START -->
## STAGE43_BB_BLOCKED_SOURCE_REPAIR_FEASIBILITY

source = `fresh_stage43_bb_blocked_source_repair_feasibility`
result_source = `fresh_blocked_source_repair_feasibility_from_validation_support_and_split_counts`
verdict = `stage43_bb_blocked_source_repair_feasibility_pass`
gate = `12 / 12`
blocked_sources = `2`
repairable_now = `0`
floor_only_now = `2`

I checked the blocked tail-adapter sources before attempting another repair. The result is deliberately conservative: TrajNet_biwi and TrajNet_mot stay floor-only because ungated transfer is strongly negative and validation support is not strong enough to justify a source-specific switch policy.
<!-- STAGE43_BB_BLOCKED_SOURCE_REPAIR_FEASIBILITY:END -->

<!-- STAGE43_BC_BLOCKED_FAMILY_SUPPORT_SCAN:START -->
## STAGE43_BC_BLOCKED_FAMILY_SUPPORT_SCAN

source = `fresh_stage43_bc_blocked_family_support_scan`
result_source = `fresh_raw_external_scan_for_blocked_source_family_support`
verdict = `stage43_bc_blocked_family_support_scan_pass`
gate = `12 / 12`
raw_files_scanned = `59`
blocked_families = `2`
repair_training_allowed_now = `0`

I scanned the raw TrajNet/OpenTraj files behind the blocked source families. The result is useful but conservative: biwi has possible raw support to convert, while mot lacks an independent support file. I am not training a repair from this scan; it only defines what support must be rebuilt before any safe source-specific repair can be tested.
<!-- STAGE43_BC_BLOCKED_FAMILY_SUPPORT_SCAN:END -->

<!-- STAGE43_BD_BIWI_SUPPORT_REBUILD_PREFLIGHT:START -->
## STAGE43_BD_BIWI_SUPPORT_REBUILD_PREFLIGHT

source = `fresh_stage43_bd_biwi_support_rebuild_preflight`
result_source = `fresh_biwi_source_family_support_rebuild_preflight`
verdict = `stage43_bd_biwi_support_rebuild_preflight_pass`
gate = `14 / 14`
biwi_sources = `2`
current_train_val_test_rows = `0 / 459 / 7685`
deployable_repair_options_now = `0`

I checked whether the raw biwi support found in BC can actually become a safe repair split. It cannot yet: the useful `biwi_hotel` rows are the current held-out biwi test source, and the small `biwi_eth` support is not enough for an independent deployable train/val/test story. I am keeping biwi floor-only and treating any within-source support split as diagnostic only.
<!-- STAGE43_BD_BIWI_SUPPORT_REBUILD_PREFLIGHT:END -->

<!-- STAGE43_BE_BLOCKED_SOURCE_SUPPORT_ACQUISITION_PREFLIGHT:START -->
## STAGE43_BE_BLOCKED_SOURCE_SUPPORT_ACQUISITION_PREFLIGHT

source = `fresh_stage43_be_blocked_source_support_acquisition_preflight`
result_source = `fresh_support_acquisition_preflight_from_local_candidates_and_stage43_blocker_artifacts`
verdict = `stage43_be_blocked_source_support_acquisition_preflight_pass`
gate = `13 / 13`
local_candidates = `3`
technical_support_candidates = `3`
conversion_ready_now = `0`
repair_training_allowed_now = `0`

I checked the local source-support options for the blocked biwi/mot families. The useful takeaway is not a new model win: biwi still needs an independent held-out source before repair training, while PETS/Town-Center/Wild-Track are technical MOT-like support candidates but still need terms/source-identity/calibration closure before guarded conversion. I am keeping these sources floor-only until those support gates clear.
<!-- STAGE43_BE_BLOCKED_SOURCE_SUPPORT_ACQUISITION_PREFLIGHT:END -->

<!-- STAGE43_BF_BLOCKED_SOURCE_TERMS_IDENTITY_PACKET:START -->
## STAGE43_BF_BLOCKED_SOURCE_TERMS_IDENTITY_PACKET

source = `fresh_stage43_bf_blocked_source_terms_identity_packet`
result_source = `fresh_terms_identity_packet_from_stage43_be_local_candidates`
verdict = `stage43_bf_blocked_source_terms_identity_packet_pass`
gate = `15 / 15`
dataset_packets = `3`
conversion_ready_now = `0`
training_allowed_now = `0`

I turned the BE local support candidates into a concrete source/terms/identity packet. This is not permission and not a conversion: PETS, Town-Center, and Wild-Track still need official source and terms confirmation before guarded conversion; biwi still needs an independent held-out source. Blocked source families remain floor-only.
<!-- STAGE43_BF_BLOCKED_SOURCE_TERMS_IDENTITY_PACKET:END -->

<!-- STAGE43_BG_BLOCKED_SOURCE_TERMS_VALIDATOR:START -->
## STAGE43_BG_BLOCKED_SOURCE_TERMS_VALIDATOR

source = `fresh_stage43_bg_blocked_source_terms_validator`
result_source = `fresh_validation_of_stage43_bf_terms_identity_template`
verdict = `stage43_bg_blocked_source_terms_validation_pass`
gate = `13 / 13`
datasets_validated = `3`
ready_for_guarded_conversion_preflight_rows = `0`
training_allowed_now = `0`

I validated the Stage43-BF terms/source template as-is. The result is intentionally blocked: no candidate source has user-confirmed terms, source identity, calibration scope, conversion scope, or Stage43 support permission yet. The manifest is useful for the next guarded conversion step, but it is not permission and it does not convert or train anything.
<!-- STAGE43_BG_BLOCKED_SOURCE_TERMS_VALIDATOR:END -->

<!-- STAGE43_BH_PROTECTED_MULTIMODAL_LATENT_CANDIDATE_LOCK:START -->
## STAGE43_BH_PROTECTED_MULTIMODAL_LATENT_CANDIDATE_LOCK

source = `fresh_stage43_bh_protected_multimodal_latent_candidate_lock`
result_source = `fresh_evidence_lock_from_verified_stage43_artifacts`
verdict = `stage43_bh_protected_multimodal_latent_candidate_lock_pass`
gate = `16 / 16`
protected_multimodal_latent_state_candidate = `True`
standalone_world_model_deployable = `False`
latest_candidate_all = `50.25%`
latest_candidate_t50 = `51.23%`
latest_candidate_hard_failure = `47.88%`

I locked the current Stage43 evidence stack into a single protected multimodal latent-state candidate record. The model family has real protected latent/head/full-waypoint evidence, but it still needs the safety floor, still uses dataset-local/raw-frame units, and still has source/terms blockers for additional support data. This is not a true-3D, foundation, metric, seconds-level, Stage5C, or SMC claim.
<!-- STAGE43_BH_PROTECTED_MULTIMODAL_LATENT_CANDIDATE_LOCK:END -->

<!-- STAGE43_BI_LOCKED_CANDIDATE_PAPER_PACKAGE_REFRESH:START -->
## STAGE43_BI_LOCKED_CANDIDATE_PAPER_PACKAGE_REFRESH

source = `fresh_stage43_bi_locked_candidate_paper_package_refresh`
result_source = `fresh_package_refresh_from_stage43_bh_candidate_lock`
verdict = `stage43_bi_locked_candidate_paper_package_refresh_pass`
gate = `14 / 14`
paper_package_refreshed = `True`
protected_multimodal_latent_state_candidate = `True`
standalone_world_model_deployable = `False`
latest_all_t50_hard_easy = `50.25%` / `51.23%` / `47.88%` / `0.00%`

I refreshed the paper-facing package from the BH evidence lock. The current claim is now easy to state: M3W has protected multimodal latent-state candidate evidence, but it remains safety-floor protected, dataset-local/raw-frame, not true 3D/foundation, and source terms for extra data are still blocked.

Boundary unchanged: no Stage5C execution, no SMC, no metric/seconds claim, and no standalone ungated deployment claim.
<!-- STAGE43_BI_LOCKED_CANDIDATE_PAPER_PACKAGE_REFRESH:END -->

<!-- STAGE43_BJ_LONG_OBJECTIVE_EVIDENCE_AUDIT:START -->
## STAGE43_BJ_LONG_OBJECTIVE_EVIDENCE_AUDIT

source = `fresh_stage43_bj_long_objective_evidence_audit`
result_source = `fresh_requirement_audit_from_stage43_bi_locked_candidate_evidence`
verdict = `stage43_bj_long_objective_evidence_audit_pass_keep_goal_active`
gate = `14 / 14`
long_objective_complete = `False`
candidate_all_t50_hard_easy = `50.25%` / `51.23%` / `47.88%` / `0.00%`

I audited the Stage43 long objective against the current BH/BI evidence stack. The protected multimodal latent-state candidate is real enough to keep as current evidence, but the full long objective is still active: source terms, metric/time calibration, t100, raw multimodal evidence, and ungated/floor-free deployment remain open blockers.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE43_BJ_LONG_OBJECTIVE_EVIDENCE_AUDIT:END -->

<!-- STAGE43_BK_T100_FAMILY_LIMITED_RECONCILIATION:START -->
## STAGE43_BK_T100_FAMILY_LIMITED_RECONCILIATION

source = `fresh_stage43_bk_t100_family_limited_reconciliation`
result_source = `fresh_reconciliation_from_stage43_p_t_u_bi_bj_verified_artifacts`
verdict = `stage43_bk_t100_family_limited_reconciliation_pass`
gate = `15 / 15`
t100_family_limited_ade_signal = `True`
uniform_t100_success = `False`
t100_endpoint_success = `False`

Stage43-BK reconciles the t100/h100 evidence: Stage43-U gives integrated t100 raw-frame full-waypoint ADE diagnostic `0.18%` with CI `[0.14%, 0.22%]`; the source-stable h100 slice gives ADE lift `2.59%`.

The blocker remains: h100 endpoint FDE is `-0.55%`, so this is not endpoint success and not a uniform t100 solution. The long objective stays active.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BK_T100_FAMILY_LIMITED_RECONCILIATION:END -->

<!-- STAGE43_BL_RAW_SCENE_GRAPH_ABLATION_READINESS:START -->
## STAGE43_BL_RAW_SCENE_GRAPH_ABLATION_READINESS

source = `fresh_stage43_bl_raw_scene_graph_ablation_readiness`
result_source = `fresh_readiness_audit_from_stage43_proxy_ablation_and_cache_schema`
verdict = `stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented`
gate = `15 / 15`
raw_scene_retrained_ablation_ready_now = `False`
graph_rich_retrained_ablation_ready_now = `False`
raw_scene_or_graph_rich_main_claim_allowed = `False`

Stage43-BL audits the scene/goal/interaction evidence after BK. Existing proxy evidence is real: full-scene proxy minus no-scene t50 is `5.79%`, and full minus no-neighbor/interaction t50 is `14.07%`. But the cache still lacks raw scene/SDF tensors and graph-rich all-agent edge tensors, so raw-scene or graph-rich interaction main claims remain blocked.

Next executable artifacts are `stage43_all_agent_graph_cache` and `stage43_raw_scene_patch_or_sdf_cache`, followed by retrained full_graph/no_graph and full_raw_scene/no_scene ablations.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future labels are supervision/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BL_RAW_SCENE_GRAPH_ABLATION_READINESS:END -->

<!-- STAGE43_BM_ALL_AGENT_CURRENT_GRAPH_CACHE:START -->
## STAGE43_BM_ALL_AGENT_CURRENT_GRAPH_CACHE

source = `fresh_stage43_bm_all_agent_current_graph_cache`
result_source = `fresh_build_current_frame_all_agent_knn_graph_cache_from_stage43_full_waypoint_rows`
verdict = `stage43_bm_all_agent_current_graph_cache_pass_partial_history_blocker`
gate = `14 / 14`
all_agent_current_graph_cache_ready = `True`
all_agent_history_graph_cache_ready = `False`
raw_scene_or_sdf_cache_ready = `False`

Stage43-BM builds current-frame all-agent KNN graph tensors from the full-waypoint row cache. Test rows `89736`, test edges `630502`, test multi-agent rows `88199`, mean degree `7.026`.

This closes the current-state neighbor graph cache gap needed for future graph-aware retraining, but it does not close the all-agent history graph or raw-scene/SDF blocker. No graph ablation or training was executed in BM.

Boundary unchanged: current-frame/past-available graph inputs only; future labels are not cached as inputs; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BM_ALL_AGENT_CURRENT_GRAPH_CACHE:END -->

<!-- STAGE43_BN_ALL_AGENT_HISTORY_GRAPH_CACHE:START -->
## STAGE43_BN_ALL_AGENT_HISTORY_GRAPH_CACHE

source = `fresh_stage43_bn_all_agent_history_graph_cache`
result_source = `fresh_build_past_only_all_agent_history_graph_cache_from_stage37_history_and_stage43_current_graph`
verdict = `stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker`
gate = `13 / 13`
all_agent_history_graph_cache_ready = `True`
raw_scene_or_sdf_cache_ready = `False`
retrained_graph_ablation_executed = `False`

Stage43-BN builds past-only target and neighbor history graph tensors from Stage37 history windows plus Stage43-BM current graph neighbors. Test rows `89736`, rows with full target history `52050`, rows with any neighbor history `88199`, edge count `630502`.

This makes graph-history retraining feasible next, but no retrained graph ablation was executed in BN. Raw-scene/SDF remains the next unresolved cache blocker.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future labels are not cached as inputs; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BN_ALL_AGENT_HISTORY_GRAPH_CACHE:END -->

<!-- STAGE43_BO_GRAPH_HISTORY_RETRAINED_ABLATION:START -->
## STAGE43_BO_GRAPH_HISTORY_RETRAINED_ABLATION

source = `fresh_stage43_bo_graph_history_retrained_ablation`
result_source = `fresh_retrained_graph_history_ablation`
verdict = `stage43_bo_graph_history_retrained_ablation_pass_contribution_supported`
gate = `14 / 14`
graph_history_retrained_ablation_executed = `True`
graph_history_contribution_supported = `True`
deployable_policy_changed = `False`

Stage43-BO fresh-trains graph-history ablation variants. Full_graph minus no_graph: all `7.73%`, t50 `15.37%`, hard/failure `6.49%`.

This is retrained contribution evidence, not a deployment policy update. Raw-scene/SDF remains outside this ablation.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BO_GRAPH_HISTORY_RETRAINED_ABLATION:END -->

<!-- STAGE43_BP_SCENE_GRAPH_MULTIMODAL_ABLATION:START -->
## STAGE43_BP_SCENE_GRAPH_MULTIMODAL_ABLATION

source = `fresh_stage43_bp_scene_graph_multimodal_ablation`
result_source = `fresh_retrained_scene_graph_multimodal_ablation`
verdict = `stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic`
gate = `16 / 16`
multimodal_scene_graph_ablation_executed = `True`
multimodal_contribution_supported = `True`
best_single_lift_supported = `False`
full_multimodal_unsafe = `True`
deployable_policy_changed = `False`

Stage43-BP fresh-trains no_context, scene_proxy_only, graph_history_only, and scene_graph_full variants. Scene_graph_full minus no_context: all `-1.03%`, t50 `-1.81%`, hard/failure `1.19%`.

Against the best single-context t50 variant `graph_history_only`, scene_graph_full delta is all `-5.22%`, t50 `-11.30%`, hard/failure `-3.49%`.

This is multimodal retrained contribution evidence, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BP_SCENE_GRAPH_MULTIMODAL_ABLATION:END -->

<!-- STAGE43_BQ_GATED_SCENE_GRAPH_FUSION:START -->
## STAGE43_BQ_GATED_SCENE_GRAPH_FUSION

source = `fresh_stage43_bq_gated_scene_graph_fusion`
result_source = `fresh_gated_scene_graph_latent_fusion`
verdict = `stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic`
gate = `13 / 13`
gated_fusion_executed = `True`
beats_best_single = `False`
beats_no_context = `False`
full_multimodal_unsafe = `False`
deployable_policy_changed = `False`

Stage43-BQ trains a learned gated scene-proxy + graph-history latent fusion model after Stage43-BP showed raw concatenation was unsafe. Protected metrics: all `29.97%`, t50 `1.18%`, hard/failure `31.60%`, easy degradation `0.50%`.

Against the Stage43-BP best single-context t50 variant `graph_history_only`, gated fusion delta is all `-6.95%`, t50 `-14.44%`, hard/failure `-6.04%`.

This is gated multimodal latent-fusion evidence, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BQ_GATED_SCENE_GRAPH_FUSION:END -->

<!-- STAGE43_BR_SCENE_GRAPH_SLICE_FORENSICS:START -->
## STAGE43_BR_SCENE_GRAPH_SLICE_FORENSICS

source = `fresh_stage43_br_scene_graph_slice_forensics`
result_source = `fresh_slice_forensics_from_stage43_bp_bq_checkpoints`
verdict = `stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal`
gate = `11 / 11`
slice_forensics_executed = `True`
targeted_scene_signal = `True`
weak_scene_signal = `True`
deployable_policy_changed = `False`

Stage43-BR replays Stage43-BP checkpoints at row level and audits scene/graph utility by source, horizon, hard/failure, and easy slices. Eligible scene-over-graph slices: `8`; scene-over-no-context slices: `18`; full-over-graph slices: `5`.

Best variant counts across eligible slices: `{'no_context': 2, 'scene_proxy_only': 7, 'graph_history_only': 17, 'scene_graph_full': 1}`.

This is slice forensics only, not a deployment policy update. Scene remains train-only proxy scene/goal/raster evidence, not raw image/SDF evidence.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BR_SCENE_GRAPH_SLICE_FORENSICS:END -->

<!-- STAGE43_BS_SCENE_GRAPH_CONTEXT_ROUTER:START -->
## STAGE43_BS_SCENE_GRAPH_CONTEXT_ROUTER

source = `fresh_stage43_bs_scene_graph_context_router`
result_source = `fresh_validation_selected_scene_graph_context_router`
verdict = `stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic`
gate = `12 / 12`
validation_selected_router = `True`
beats_graph_history_on_any_core_metric = `False`
easy_safe = `True`
deployable_policy_changed = `False`

Stage43-BS builds a validation-only source/domain/horizon route table from Stage43-BP context variants after Stage43-BR found targeted scene signal. Selected routes: `0`; validation-safe candidates: `5`.
Unsafe full scene+graph context blocked by BP prior: `True`.

Test metrics: all `36.91%`, t50 `15.62%`, t100 raw-frame diagnostic `-3.26%`, hard/failure `37.64%`, easy degradation `0.00%`.
Delta vs graph-history-only: all `0.00%`, t50 `0.00%`, hard/failure `0.00%`, easy degradation `0.00%`.

This is a diagnostic context-routing experiment, not a deployment policy update.

Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BS_SCENE_GRAPH_CONTEXT_ROUTER:END -->

<!-- STAGE43_BT_CONTEXT_ADMISSIBILITY_MODEL:START -->
## STAGE43_BT_CONTEXT_ADMISSIBILITY_MODEL

source = `fresh_stage43_bt_context_admissibility_model`
result_source = `fresh_row_level_harm_aware_context_admissibility`
verdict = `stage43_bt_context_admissibility_pass_safe_lift_diagnostic`
gate = `14 / 14`
row_level_admissibility_trained = `True`
beats_graph_history_on_any_core_metric = `True`
easy_safe = `True`
deployable_policy_changed = `False`

Stage43-BT trains a row-level harm-aware context admissibility model over the Stage43-BP scene/graph variants. It uses graph-history causal features as input and future variant error only as train/eval labels.
Validation selected policy: `{'gain_threshold': 0.5, 'harm_threshold': 0.5, 'predicted_gain_threshold': 0.0}`; safe validation candidates: `125 / 125`.

Test metrics: all `39.06%`, t50 `16.02%`, t100 raw-frame diagnostic `-3.21%`, hard/failure `39.66%`, easy degradation `0.00%`.
Delta vs graph-history-only: all `2.15%`, t50 `0.40%`, hard/failure `2.02%`, easy degradation `0.00%`.
Test context counts: `{'graph_history_only': 10322, 'scene_graph_full': 531, 'scene_proxy_only': 1147}`.

Interpretation: this is a context-admissibility diagnostic, not a deployment update unless it safely beats graph-history on core metrics. Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BT_CONTEXT_ADMISSIBILITY_MODEL:END -->

<!-- STAGE43_BU_CONTEXT_ADMISSIBILITY_ROBUSTNESS_AUDIT:START -->
## STAGE43_BU_CONTEXT_ADMISSIBILITY_ROBUSTNESS_AUDIT

source = `fresh_stage43_bu_context_admissibility_robustness_audit`
result_source = `fresh_replay_bootstrap_slice_audit_from_stage43_bt`
verdict = `stage43_bu_context_admissibility_partial_robust_lift_pass`
gate = `12 / 12`
robust_all_hard_lift = `True`
t50_bootstrap_robust = `True`
t100_bootstrap_robust = `False`
t100_ci_crosses_zero = `True`
slice_easy_safe = `False`
easy_safe_ci = `True`
deployable_policy_changed = `False`

Stage43-BU exact-replays Stage43-BT and adds bootstrap plus source/domain/horizon slice evidence. It is a robustness audit, not a deployment update.
Replay diff max: `0.00000000`.
Delta vs graph-history-only: all `2.15%`, t50 `0.40%`, t100 raw-frame diagnostic `0.05%`, hard/failure `2.02%`, easy degradation `0.00%`.
Bootstrap CI low vs graph-history-only: all `1.91%`, t50 `0.00%`, hard/failure `1.78%`, easy high `0.00%`.
Slice audit: `31` positive slices, `2` negative slices, easy hazard slices `7`, core weak slices `[]`.

Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BU_CONTEXT_ADMISSIBILITY_ROBUSTNESS_AUDIT:END -->

<!-- STAGE43_BV_CONTEXT_ADMISSIBILITY_SLICE_SAFE_REPAIR:START -->
## STAGE43_BV_CONTEXT_ADMISSIBILITY_SLICE_SAFE_REPAIR

source = `fresh_stage43_bv_context_admissibility_slice_safe_repair`
result_source = `fresh_validation_selected_slice_safe_context_repair`
verdict = `stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk`
gate = `12 / 12`
selected_repair_mode = `block_t100`
easy_safe = `True`
slice_easy_safe = `False`
core_lift_vs_graph_history = `True`
t100_bootstrap_robust = `False`
deployable_policy_changed = `False`

Stage43-BV applies a validation-only slice-safety repair to Stage43-BT context admissibility. It blocks context on validation-identified unsafe source/domain/horizon slices and evaluates test once.
Delta vs graph-history-only: all `2.14%`, t50 `0.40%`, t100 raw-frame diagnostic `0.00%`, hard/failure `2.01%`, easy degradation `0.00%`.
Bootstrap CI low vs graph-history-only: all `1.91%`, t50 `-0.02%`, t100 raw `0.00%`, hard/failure `1.79%`, easy high `0.00%`.
Slice audit: positive `28`, negative `2`, easy hazards `10`, core weak `['horizon_100']`.

Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BV_CONTEXT_ADMISSIBILITY_SLICE_SAFE_REPAIR:END -->

<!-- STAGE43_BW_CONTEXT_HAZARD_ATTRIBUTION_GUARD:START -->
## STAGE43_BW_CONTEXT_HAZARD_ATTRIBUTION_GUARD

source = `fresh_stage43_bw_context_hazard_attribution_guard`
result_source = `fresh_validation_selected_context_hazard_attribution_guard`
verdict = `stage43_bw_context_hazard_attribution_pass_floor_inherent_risk`
gate = `13 / 13`
selected_guard = `guard_domain_horizon_rate_0.20_plus_block_t100`
deployable_policy_changed = `False`

Stage43-BW distinguishes graph-history floor-inherent absolute easy risk from context-induced easy harm. This matters because BV's remaining easy-hazard slices can be inherited from the floor rather than caused by scene/graph context.
Absolute easy hazard slices: graph-history `11`, BT unrepaired `7`, selected guard `10`.
Context-induced hazard slices: BT unrepaired `12`, selected guard `9`.
Delta vs graph-history-only: all `1.86%`, t50 `-0.02%`, t100 raw-frame diagnostic `0.00%`, hard/failure `1.77%`, easy degradation `0.00%`.
Bootstrap CI low vs graph-history-only: all `1.65%`, t50 `-0.33%`, t100 raw `0.00%`, hard/failure `1.56%`, easy high `0.00%`.
Source overlap audit: `{'val_source_count': 4, 'test_source_count': 4, 'overlap_count': 0, 'overlap_examples': [], 'held_out_source_level': True}`.

Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BW_CONTEXT_HAZARD_ATTRIBUTION_GUARD:END -->

<!-- STAGE43_BX_LATENT_RISK_HEAD_ROBUSTNESS_AUDIT:START -->
## STAGE43_BX_LATENT_RISK_HEAD_ROBUSTNESS_AUDIT

source = `fresh_stage43_bx_latent_risk_head_robustness_audit`
result_source = `fresh_checkpoint_replay_latent_risk_head_robustness`
verdict = `stage43_bx_latent_risk_head_robustness_pass_horizon_caveat`
gate = `12 / 12`
deployable_policy_changed = `False`

Stage43-BX fresh-replays the latent checkpoint and audits failure/gain/harm risk heads across domain and horizon slices with row-subsampled bootstrap CIs.
Global AUROC: failure `0.8709`, gain `0.8845`, harm `0.9050`.
Bootstrap AUROC low: failure `0.8626`, gain `0.8769`, harm `0.8981`.
Weak horizon slices: `5`; minimum horizon AUROC `0.6147`.

Boundary unchanged: protected dataset-local/raw-frame 2.5D only; no ungated deployment, no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BX_LATENT_RISK_HEAD_ROBUSTNESS_AUDIT:END -->

<!-- STAGE43_BY_LATENT_TRANSITION_CONSISTENCY_AUDIT:START -->
## STAGE43_BY_LATENT_TRANSITION_CONSISTENCY_AUDIT

source = `fresh_stage43_by_latent_transition_consistency_audit`
result_source = `fresh_checkpoint_replay_latent_transition_consistency`
verdict = `stage43_by_latent_transition_consistency_pass_with_readout_caveat`
gate = `13 / 13`
deployable_policy_changed = `False`

Stage43-BY fresh-replays the Stage43-M latent checkpoint and audits the latent transition `z_t -> z_next` against future target latent labels.
Raw global transition gain vs identity: `0.7450`.
Raw global transition gain vs train target-centroid: `-0.0357`.
Train-only calibrated readout gain vs identity: `-0.0177`.
Train-only calibrated readout gain vs train target-centroid: `0.3097`.
Bootstrap gain-vs-identity CI low: `0.7417`.
Weak transition slices: `4` raw, `5` calibrated.

Interpretation: raw dynamics clearly moves away from identity toward the future latent, and a train-only calibrated readout beats the centroid baseline; however calibrated identity remains slightly stronger, so this is partial latent-dynamics evidence, not proof of an independent ungated dynamics advantage. Future target latents are label/eval only, not inference input. Boundary unchanged: protected dataset-local/raw-frame 2.5D only; no ungated deployment, no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.
<!-- STAGE43_BY_LATENT_TRANSITION_CONSISTENCY_AUDIT:END -->

<!-- STAGE43_BZ_LATENT_TRANSITION_ADAPTER_REPAIR:START -->
## STAGE43_BZ_LATENT_TRANSITION_ADAPTER_REPAIR

source = `fresh_stage43_bz_latent_transition_adapter_repair`
result_source = `fresh_train_only_latent_transition_adapter_repair`
verdict = `stage43_bz_latent_transition_adapter_repair_pass`
gate = `15 / 15`
deployable_policy_changed = `False`

Stage43-BZ trains a train-only, past-only latent transition adapter with frozen Stage43-M encoders to repair the Stage43-BY calibrated readout caveat.
Adapter raw gain vs identity: `0.8404`.
Adapter raw gain vs train centroid: `0.3516`.
Adapter calibrated gain vs identity: `0.2014`.
Adapter calibrated gain vs train centroid: `0.4583`.
Calibrated gain-vs-identity CI low: `0.1884`.

Interpretation: this is a targeted latent transition repair experiment. It does not change deployment, does not remove the safety floor, and does not enable Stage5C or SMC.
<!-- STAGE43_BZ_LATENT_TRANSITION_ADAPTER_REPAIR:END -->

<!-- STAGE43_CA_LATENT_ADAPTER_DOWNSTREAM_HEADS:START -->
## STAGE43_CA_LATENT_ADAPTER_DOWNSTREAM_HEADS

source = `fresh_stage43_ca_latent_adapter_downstream_heads`
result_source = `fresh_train_only_downstream_head_audit`
verdict = `stage43_ca_latent_adapter_downstream_heads_partial_lift`
gate = `13 / 16`
deployable_policy_changed = `False`

Stage43-CA fits train-only downstream heads on identity, Stage43-M transition, Stage43-BZ adapter, and current+future-latent concatenations.
Selected adapter variant by validation objective: `identity_stage43m_adapter_z`.
Adapter downstream mean ADE: `0.2961`.
Adapter risk mean AUROC: `0.8910`.
Protected all improvement vs floor: `0.0324`.
Protected t50 improvement vs floor: `-0.0022`.
Protected easy degradation: `0.0427`.

Interpretation: downstream readouts test whether the repaired latent transition supports future waypoint/risk/density heads. This does not change deployment, remove the safety floor, or enable Stage5C/SMC.
<!-- STAGE43_CA_LATENT_ADAPTER_DOWNSTREAM_HEADS:END -->

<!-- STAGE43_CB_DOWNSTREAM_EASY_GUARD_AUDIT:START -->
## STAGE43_CB_DOWNSTREAM_EASY_GUARD_AUDIT

source = `fresh_stage43_cb_downstream_easy_guard_audit`
result_source = `fresh_validation_only_easy_guard_replay`
verdict = `stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch`
gate = `12 / 13`
deployable_policy_changed = `False`

Stage43-CB reruns the Stage43-CA selected latent downstream heads with a validation-only easy guard using predicted risk and model-vs-floor disagreement.
Validation all / hard / easy: `0.1313` / `0.1688` / `0.0002`.
Test all / t50 / hard / easy: `0.0321` / `-0.0083` / `0.0656` / `0.0528`.
Validation-test easy degradation gap: `0.0526`.

Interpretation: downstream latent heads still show all/hard signal, but easy-safety does not reliably transfer from validation to test. Deployment remains unchanged.
<!-- STAGE43_CB_DOWNSTREAM_EASY_GUARD_AUDIT:END -->

<!-- STAGE43_CC_SHADOW_EASY_GUARD_REPAIR:START -->
## STAGE43_CC_SHADOW_EASY_GUARD_REPAIR

source = `fresh_stage43_cc_shadow_easy_guard_repair`
result_source = `fresh_shadow_validation_easy_guard_repair`
verdict = `stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch`
gate = `12 / 13`
deployable_policy_changed = `False`

Stage43-CC repairs the Stage43-CB validation/test easy mismatch with a validation-only shadow holdout and source-family support guard.
Selected shadow policy: `base_threshold_only`.
Shadow all / hard / easy: `0.1321` / `0.1690` / `0.0010`.
Test all / t50 / hard / easy: `0.0321` / `-0.0083` / `0.0655` / `0.0527`.

Interpretation: this is a safety protocol repair for latent downstream heads. Deployment remains unchanged unless test easy safety and lift both hold.
<!-- STAGE43_CC_SHADOW_EASY_GUARD_REPAIR:END -->

<!-- STAGE43_CD_SOURCE_FAMILY_COVERAGE_GUARD:START -->
## STAGE43_CD_SOURCE_FAMILY_COVERAGE_GUARD

source = `fresh_stage43_cd_source_family_coverage_guard`
result_source = `fresh_validation_source_family_coverage_guard`
verdict = `stage43_cd_source_family_coverage_guard_pass`
gate = `14 / 14`
deployable_policy_changed = `False`

Stage43-CD promotes the validation source-family support gap from Stage43-CC into an explicit coverage guard: source families absent from validation fall back to the floor.
Selected policy: `domain_source_family_coverage_guard`.
Unsupported test families: `{'pets': 1926, 'zara': 9540}`.
Test all / t50 / hard / easy: `0.0111` / `-0.0000` / `0.0135` / `0.0000`.
Shadow all / hard / easy: `0.1321` / `0.1690` / `0.0010`.

Interpretation: the guard restores easy safety and keeps small all-row lift, but hard/failure lift drops and t50 remains negative. It is evidence for source-coverage safety, not a deployment replacement.
<!-- STAGE43_CD_SOURCE_FAMILY_COVERAGE_GUARD:END -->

<!-- STAGE43_CE_SOURCE_FAMILY_COVERAGE_SPLIT_REPAIR:START -->
## STAGE43_CE_SOURCE_FAMILY_COVERAGE_SPLIT_REPAIR

source = `fresh_stage43_ce_source_family_coverage_split_repair`
result_source = `fresh_metadata_only_source_family_coverage_split_repair`
verdict = `stage43_ce_source_family_coverage_split_repair_ready`
gate = `14 / 14`
deployable_policy_changed = `False`
new_model_training_run = `False`

Stage43-CE builds a metadata-only coverage-aware source split so validation covers every test source family/domain-family where feasible. This directly addresses the Stage43-CD over-conservative fallback caused by validation support gaps.

Split rows train/val/test = `192531` / `62796` / `82664`.
Test families without validation support = `[]`.
Test domain-families without validation support = `[]`.
Tradeoff: the repaired test split is intentionally coverage-aware and narrower than the broad external stress split; singleton/unsupported source families remain blockers, not successes.

Interpretation: this is split-protocol repair readiness, not a new model result. The next step is to rebuild the full-waypoint supervision cache and retrain/evaluate latent dynamics on the repaired split.
<!-- STAGE43_CE_SOURCE_FAMILY_COVERAGE_SPLIT_REPAIR:END -->

<!-- STAGE43_CF_COVERAGE_AWARE_FULL_WAYPOINT_CACHE:START -->
## STAGE43_CF_COVERAGE_AWARE_FULL_WAYPOINT_CACHE

source = `fresh_stage43_cf_coverage_aware_full_waypoint_cache`
result_source = `fresh_cache_rebuild_from_stage43_ce_assignment`
verdict = `stage43_cf_coverage_aware_full_waypoint_cache_ready`
gate = `14 / 14`
deployable_policy_changed = `False`
new_model_training_run = `False`
cache_committed = `False`

Stage43-CF materializes the CE coverage-aware source split into a local full-waypoint supervision cache. Future endpoints and waypoints are labels/eval targets only, not inference inputs.

Cache rows train/val/test = `192531` / `62796` / `82664`.
Cache dir = `data/stage43_ce_full_waypoint_supervision_cache`.

Interpretation: this closes the repaired-split cache blocker. It is not a model result; the next step is training/evaluating latent dynamics on this repaired cache.
<!-- STAGE43_CF_COVERAGE_AWARE_FULL_WAYPOINT_CACHE:END -->

<!-- STAGE43_CG_COVERAGE_AWARE_LATENT_DYNAMICS:START -->
## STAGE43_CG_COVERAGE_AWARE_LATENT_DYNAMICS

source = `fresh_stage43_cg_coverage_aware_latent_dynamics`
result_source = `fresh_run`
mode = `medium`
verdict = `stage43_cg_coverage_aware_latent_dynamics_candidate_pass`
gate = `15 / 15`
deploy_coverage_aware_latent_dynamics = `True`

I retrained the full-waypoint latent dynamics head on the CE coverage-aware source split. This is the first model run using the repaired split cache, not just another cache audit.

- all full-waypoint ADE improvement vs floor: `51.47%`
- t50 full-waypoint ADE improvement vs floor: `31.13%`
- hard/failure full-waypoint ADE improvement vs floor: `49.72%`
- easy degradation: `0.00%`
- switch rate: `71.25%`

Boundary: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.
<!-- STAGE43_CG_COVERAGE_AWARE_LATENT_DYNAMICS:END -->

<!-- STAGE43_CH_COVERAGE_AWARE_T100_FAILURE_AUDIT:START -->
## STAGE43_CH_COVERAGE_AWARE_T100_FAILURE_AUDIT

source = `fresh_stage43_ch_coverage_aware_t100_failure_audit`
result_source = `fresh_replay_audit_from_stage43_cg_medium_checkpoint`
verdict = `stage43_ch_t100_failure_audit_pass_blocker_confirmed`
gate = `11 / 11`

I replayed the Stage43-CG medium checkpoint on the CE test subset and isolated the long-horizon t100 failure. This is an audit, not a new t100 deployment.

- t100 rows: `8443`
- t100 full-waypoint ADE improvement: `-5.51%`
- t100 bootstrap CI: `[-6.18%, -4.85%]`
- t100 switch rate: `12.78%`
- all/t50 CG remains positive, but t100 remains diagnostic-only.

Boundary: no metric/seconds-level claim; no Stage5C; no SMC; future waypoints remain labels/eval only.
<!-- STAGE43_CH_COVERAGE_AWARE_T100_FAILURE_AUDIT:END -->

<!-- STAGE43_CI_COVERAGE_AWARE_T100_SAFE_SWITCH:START -->
## STAGE43_CI_COVERAGE_AWARE_T100_SAFE_SWITCH

source = `fresh_stage43_ci_coverage_aware_t100_safe_switch`
result_source = `fresh_stage43_ci_coverage_aware_t100_safe_switch`
verdict = `stage43_ci_t100_safe_switch_pass_floor_repair`
gate = `15 / 15`
deploy_t100_latent_switch = `False`
deploy_t100_safe_floor_repair = `True`

I repaired the Stage43-CG t100 blocker with a validation-selected t100 safe-switch rule. The key point is conservative: if t100 latent switching is not demonstrably safe, t100 rows fall back to the CE floor instead of carrying the negative switch found in Stage43-CH.

- all full-waypoint ADE improvement: `52.03%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- t100 delta vs Stage43-CG unsafe base: `5.51%`

Boundary: this is still dataset-local/raw-frame 2.5D evidence. No metric/seconds-level claim, no Stage5C execution, and no SMC.
<!-- STAGE43_CI_COVERAGE_AWARE_T100_SAFE_SWITCH:END -->

<!-- STAGE43_CJ_COVERAGE_AWARE_T100_LONG_HORIZON_SPECIALIST:START -->
## STAGE43_CJ_COVERAGE_AWARE_T100_LONG_HORIZON_SPECIALIST

source = `fresh_stage43_cj_coverage_aware_t100_long_horizon_specialist`
result_source = `fresh_stage43_cj_coverage_aware_t100_long_horizon_specialist`
verdict = `stage43_cj_t100_long_horizon_specialist_pass_keep_ci_floor`
gate = `15 / 15`
deploy_t100_specialist = `False`

I trained a t100-only long-horizon neural specialist on the coverage-aware split. It uses causal features and Stage43-CG latent outputs, with future waypoints only as labels. Deployment remains protected by the Stage43-CI floor if the specialist is not t100-positive and easy-safe.

- all full-waypoint ADE improvement: `52.03%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- t100 delta vs Stage43-CI floor: `0.00%`

Boundary: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.
<!-- STAGE43_CJ_COVERAGE_AWARE_T100_LONG_HORIZON_SPECIALIST:END -->

<!-- STAGE43_CK_COVERAGE_AWARE_T100_CAUSAL_FEATURE_REPAIR:START -->
## STAGE43_CK_COVERAGE_AWARE_T100_CAUSAL_FEATURE_REPAIR

I reran the t100 specialist as a causal-only repair after auditing Stage43-CJ. CJ never deployed its t100 specialist, but its diagnostic feature set included true-error values derived from future waypoints, so it should not be cited as clean no-leakage specialist evidence.

- result source: `fresh_stage43_ck_coverage_aware_t100_causal_feature_repair`
- gate: `18 / 18`
- verdict: `stage43_ck_t100_causal_feature_repair_pass_keep_ci_floor`
- deploy t100 causal specialist: `False`
- deployed all improvement: `52.03%`
- deployed t50 improvement: `31.13%`
- deployed t100 raw-frame diagnostic: `0.00%`
- deployed hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- validation-selected causal specialist t100 diagnostic: `-0.01%`
- raw causal candidate t100 diagnostic: `-9.05%`

Current interpretation: the deployed policy remains the Stage43-CI t100 floor unless CK is t100-positive and easy-safe. No future endpoint, future waypoint, central velocity, test endpoint goal, or label-derived true-error feature is used as inference input in CK.
<!-- STAGE43_CK_COVERAGE_AWARE_T100_CAUSAL_FEATURE_REPAIR:END -->

<!-- STAGE43_CL_T100_SOURCE_STABLE_COMPATIBILITY_AUDIT:START -->
## STAGE43_CL_T100_SOURCE_STABLE_COMPATIBILITY_AUDIT

I reconciled the earlier Stage43-T source-stable h100 result with the current CK global t100 floor. The result is deliberately conservative: Stage43-T remains useful as local source-level evidence, but it is not a global t100 deployment result.

- gate: `12 / 12`
- verdict: `stage43_cl_t100_source_stable_compatibility_pass_local_only`
- Stage43-T local h100 ADE lift: `2.59%` on `1440` rows
- Stage43-T local easy degradation: `0.00%`
- current CK global t100 diagnostic: `0.00%`
- global t100 deployment allowed: `False`

Current interpretation: t100 has a small local source-stable positive signal, but current deployable t100 remains the floor. Future t100 work needs current-matrix-compatible source-family gates with persisted feature names and stronger source support.
<!-- STAGE43_CL_T100_SOURCE_STABLE_COMPATIBILITY_AUDIT:END -->

<!-- STAGE43_CM_CURRENT_MATRIX_T100_SOURCE_FAMILY_GATE:START -->
## STAGE43_CM_CURRENT_MATRIX_T100_SOURCE_FAMILY_GATE

I rebuilt the t100 source-family check on the current Stage43 full-waypoint matrix rather than relying on the earlier small Stage43-T local split. The run persists causal feature names, feature hashes, and source/split hashes so the evidence can be audited later.

- gate: `13 / 13`
- verdict: `stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor`
- current test rows: `89736`
- current t100 rows: `18070`
- raw validation-rule t100 lift: `-3.86%`
- raw easy degradation: `2.26%`
- deployed t100 lift: `0.00%`
- deployed easy degradation: `0.00%`
- deploy t100 source-family gate: `False`

Interpretation: this is a current-matrix compatibility audit for t100. If the validation-selected rule is not positive and easy-safe on the current matrix, the deployed policy remains the floor; no t100 success, metric, seconds-level, Stage5C, or SMC claim is made.
<!-- STAGE43_CM_CURRENT_MATRIX_T100_SOURCE_FAMILY_GATE:END -->

<!-- STAGE43_CN_T100_VALIDATION_SHIFT_FORENSICS:START -->
## STAGE43_CN_T100_VALIDATION_SHIFT_FORENSICS

I replayed the Stage43-CM selected model and audited why the validation-allowed t100 source-family rule failed on the current matrix test split.

- gate: `11 / 11`
- verdict: `stage43_cn_t100_validation_shift_forensics_pass_ucy_shift_blocker`
- raw validation-allowed t100 lift: `-3.86%`
- raw easy degradation: `2.26%`
- UCY validation lift: `1.54%`
- UCY test lift: `-4.32%`
- UCY test easy degradation: `21.00%`
- root causes: `UCY_test_easy_harm, UCY_test_lift_nonpositive, low_val_test_scene_overlap, low_val_test_source_file_overlap, validation_allowed_family_failed_current_test`

Interpretation: t100 remains floor-only. The current blocker is validation/test source-scene shift inside the validation-allowed UCY t100 rule, not a lack of current-matrix t100 rows. Future t100 work needs source-file or scene-level validation support before switching.
<!-- STAGE43_CN_T100_VALIDATION_SHIFT_FORENSICS:END -->

<!-- STAGE43_CO_T100_SOURCE_SCENE_SUPPORT_GATE:START -->
## Stage43-CO: t100 source/scene support gate

I added a stricter t100 safety rule after the Stage43-CN shift audit: the model can only switch at t100 when the exact source file or scene has validation-positive, easy-safe support.

- gate: `14 / 14`
- verdict: `stage43_co_t100_source_scene_support_gate_pass_floor_required`
- current t100 rows: `18070`
- switched t100 rows: `0`
- blocked t100 rows: `18070`
- raw family-rule t100 lift: `-3.86%`
- source/scene-supported t100 lift: `0.00%`
- source/scene-supported easy degradation: `0.00%`

What this means: t100 is still floor-only. The gate blocks every current t100 switch because validation and test do not share source-file or scene support. That is not a t100 improvement, but it does prevent the unsafe UCY family-level switch from being deployed.
<!-- STAGE43_CO_T100_SOURCE_SCENE_SUPPORT_GATE:END -->

<!-- STAGE43_CP_T100_SOURCE_SCENE_SUPPORT_SPLIT_REPAIR:START -->
## Stage43-CP: t100 source/scene-supported split

I built a separate agent-disjoint split protocol for t100 work where validation and test share source/scene support without sharing rows or source-agent tracks.

- gate: `13 / 13`
- verdict: `stage43_cp_t100_source_scene_support_split_ready`
- test t100 rows: `11820`
- source-or-scene-supported t100 ratio: `100.00%`
- exact source-scene-supported t100 ratio: `99.19%`
- row disjoint: `True`
- source-agent disjoint: `True`

This is not a new model result and not cross-source generalization. It is the protocol I need before trying another t100 learner: current heldout t100 stays floor-only, while this supported split can test whether t100 learning is possible when validation actually covers the source/scene.
<!-- STAGE43_CP_T100_SOURCE_SCENE_SUPPORT_SPLIT_REPAIR:END -->
