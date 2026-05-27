# M3W 长期目标证据总账：路线、失败、成功与当前边界

更新时间：2026-05-27

工作目录：`/Users/yangyue/Downloads/World`

结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README、`research_state.json`，并纳入最近已经完成并提交的 Stage42-ES 到 Stage42-FP 结果。本文是总结与索引，不是新的训练或新的评估。

## 1. 最短结论

M3W 目前已经从早期的 SDD-only 2.5D trajectory scaffold，推进到一个有 SDD 与外部 top-down pedestrian 数据证据的 **protected dataset-local / raw-frame 2.5D multi-agent world-state candidate**。

当前不能声称：

- 不是 true 3D world model。
- 不是 large-scale foundation world model。
- 不是全局 metric / meter-level predictor。
- t+50 / t+100 仍是 raw-frame horizon，不能说成 seconds-level。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external 是 dataset-local / unverified weak-metric diagnostic，不是统一物理坐标。
- JEPA 不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- self-audited / visual-prior / auto-silver 不是 human gold。

当前最强可部署结论：

| 场景 | 当前 best deployable / best evidence | 状态 |
| --- | --- | --- |
| SDD pixel raw-frame | Stage26 cost-aware selector | SDD t+50 与 hard/failure 正提升，仍是 pixel/raw-frame。 |
| External t+50 transfer | Stage37 causal-history + goal-prototype safe selector | external all/t50/hard/easy 同时过 gate，是 external selector best deployable。 |
| Source-level protected full-waypoint / group-consistency | Stage42-FH/FI frozen policy family，后续 FJ/FK/FL/FM/FN/FO 审计 | dual-domain/source robust 可以写，uniform horizon 不能写。 |
| Neural dynamics | Stage39/40/41/42 protected neural / full-waypoint evidence | 只能写 protected world-state candidate，不能写 ungated neural world model 已部署。 |

最新 Stage42-FP 结论：TrajNet|100 与 UCY|100 的 blocker 不是简单模型容量或 threshold 问题，而是 h100 source/support/context 问题。两个切片都同时出现 low-margin ambiguity、low material headroom、validation-to-test source-family shift、single/sparse validation source support，以及 source-specific easy-safety CI failure。因此 uniform horizon robustness 仍然禁止 claim。

## 2. 你问的“这个目标内做了什么”

我做的事情不是一条线，而是围绕“真实世界多模态多智能体世界模型 M3W”不断尝试、失败、修复、冻结和审计。主线可以分成九类。

### 2.1 强因果基线与安全 fallback

尝试内容：

- 建立 constant position、constant velocity causal finite difference、damped velocity、constant acceleration、turn-rate、scene-clamped、goal-directed 等强因果基线。
- 建立 strongest baseline 表、oracle headroom、selector regret、hard/failure/easy 切片。
- 后续所有 learned selector / neural model 都必须和 strongest causal baseline 以及 Stage26 / Stage37 floor 比。

结果：

- 成功。
- 这条路线最终产生了 Stage26、Stage37 和 Stage42 protected policy family。
- 它是目前最可靠的可部署基座。

原因：

- 原始轨迹预测中，强 causal baseline 已经非常强。
- 任何模型如果不能避免伤害 easy case，就不能部署。
- fallback floor 把“学到一点点”变成“安全地只在值得切换时切换”。

### 2.2 JEPA 表征路线

尝试内容：

- Stage18/19/22/23/24 以及后续 M3W 阶段做了多轮 JEPA-only、scene/trajectory JEPA、interaction-aware JEPA。
- 检查 latent variance / non-collapse。
- 用 JEPA latent 做 selector、failure predictor、goal predictor、hard/failure correction probe。

结果：

- 主要失败。
- 多次 non-collapse，但没有稳定 downstream lift。
- JEPA 不能作为当前主贡献。

失败原因：

- non-collapse 只说明表征没有塌缩，不说明它对 selector/failure/t50 有用。
- 当前 JEPA 目标和部署损失之间有错位。
- scene/video/raw frame grounding 仍弱，外部数据也缺统一 metric/seconds calibration。
- JEPA latent 缩小分布距离不等于能预测 gain/harm。

当前处理：

- JEPA 保留为 representation auxiliary / diagnostic。
- 不把 JEPA 写成生成式 world model。
- 不启用 latent generative rollout。

### 2.3 Transformer / Hybrid neural dynamics

尝试内容：

- 训练 Transformer-only、JEPA+Transformer hybrid、Causal Transformer with history/neighbor tokens、protected neural dynamics、full-waypoint sequence dynamics。
- 输出 trajectory、failure/gain/harm、selector、interaction、occupancy、physical validity proxy。
- 所有 neural outputs 必须经过 Stage37 safety floor。

结果：

- 无保护 neural 不安全。
- protected neural / full-waypoint family 有 evidence，但没有成为完全独立可部署 neural dynamics。
- 不能说“神经网络已经独立超过 Stage37 并可部署”。

失败原因：

- neural without fallback 很容易伤 easy cases。
- Stage37/teacher floor 已经很强，直接 residual/correction 很难稳定超过它。
- 数据仍是 dataset-local/raw-frame，不是统一物理世界坐标。
- t100 长时程仍受 horizon/data/support 限制。

当前处理：

- 神经部分作为 protected world-state candidate 的组成部分。
- 部署仍以 Stage37/teacher floor 保护。
- 只有在 easy degradation、hard/failure、t50、source robustness 全过 gate 时才允许升级。

### 2.4 SDD official pixel-space benchmark

尝试内容：

- 下载并转换 SDD。
- 构建 SDD world-state shards、scene packs、lazy episodes。
- 建立 GoalBench、HardBench、BaselineFailureBench。
- 计算 strongest causal baseline。
- 训练 cost-aware selector、failure predictor、JEPA heads。

成功结果：

```text
Stage26 selector:
  t+50 improvement: about +14.58%
  hard/failure improvement: about +11.23%
  easy degradation: about +1.81%
```

意义：

- Stage26 是 SDD pixel raw-frame 上的 best deployable。
- 它修复了 Stage24 hard-class selector 的过度切换。

限制：

- SDD 仍是 pixel-space。
- t+50/t+100 是 raw annotation-frame horizon。
- homography、scale、effective seconds 未全局验证。

### 2.5 External zero-shot / domain alignment

尝试内容：

- 把 SDD / M3W-LAS 迁移到 OpenTraj、UCY、ETH-UCY、TrajNet。
- 做 normalization、relative target、coordinate-invariant features、latent alignment、CORAL、linear adapter、domain-conditioned selector。

失败结果：

```text
Stage31 zero-shot:
  all improvement: about -92.67%
  t50 improvement: about -278.57%
```

失败原因：

- SDD pixel 与 external dataset-local 坐标不兼容。
- 外部 scene/goal/interaction context 不完整。
- agent type 不一致。
- horizon 定义和 track length 分布不一致。
- latent distribution alignment 没有保证 target alignment。

修复方向：

- 不再直接 zero-shot。
- 补逐行几何、train-only goals、relative-error targets、history windows、scene-agnostic goal prototypes。

### 2.6 External selective transfer

尝试内容：

- Stage34/35 构建外部 row geometry、train-only goals、hard/easy/failure label、selective transfer policy。
- 只在 hard/failure probability 高、predicted gain 高、harm 低时切换。

中间结果：

```text
Stage35:
  all improvement: +12.13%
  hard/failure improvement: +13.98%
  easy degradation: 0.041%
  t+50 improvement: 0.0
```

结论：

- all/hard/easy 过了，但 t50 没过，所以不可部署。

失败原因：

- t50 objective 被 all-test objective 淹没。
- 缺少完整 past-only history window。
- held-out scene 下 train-only scene goals 不足。
- existing policy 对长时程切换太保守。

### 2.7 Stage37 causal history + scene-agnostic goal prototypes

尝试内容：

- 为 external 每行构建 K=8/16/32/64 past-only history window。
- 构建 scene-agnostic goal prototypes：straight、slow-stop、left-turn、right-turn、group-follow、density-avoid、exit-like direction。
- 训练 t50 switchability、gain、harm、安全选择器。
- 用 conformal / validation safety guard 控制 easy degradation。

成功结果：

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

- 修复了 Stage35/36 的 t50=0 blocker。
- external selector-level deployable 成立。
- 这是当前 external best deployable。

限制：

- 仍是 dataset-local raw-frame。
- 不是 metric/seconds-level。
- 不是 true 3D/foundation。

### 2.8 Bounded correction / residual

尝试内容：

- 训练 bounded correction head：`prediction = selected_baseline + alpha * bounded_delta`。
- 尝试 linear/ridge/small MLP/horizon-specific/hard-only/t50-only correction。
- 与 Stage37 frozen policy、Stage35、Stage38 correction、external strongest baseline 比。

结果：

- 不部署。
- 没有稳定超过 Stage37。

失败原因：

- residual 直接改轨迹风险很高。
- easy case 容易被 residual 伤害。
- Stage37 selector floor 已经捕获了大部分安全收益。
- correction 如果不能同时提升 all/t50/hard 并保持 easy，就不能部署。

### 2.9 Stage42 source-level full-waypoint / group-consistency

尝试内容：

- 从 endpoint bridge 走向 full-waypoint sequence / source-level evaluation。
- 做 source-level split、common-validation bridge/shape composer、proximity guard、group consistency、risk repair、objective-level proximity training、policy freeze、bootstrap、source/domain/horizon audit。

关键成功：

Stage42-FE constrained FC/safety composer：

```text
all/t50/hard: 26.41% / 23.15% / 24.81%
near@0.05: 1.32%
gate: 19 / 19
```

Stage42-FH UCY-supported FE composer：

```text
all/t50/t100raw/hard: 34.98% / 28.97% / 20.57% / 33.10%
TrajNet and UCY: positive-safe
gate: 20 / 20
```

Stage42-FI frozen replay / bootstrap：

```text
policy hash: f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6
replay max diff: 0
bootstrap CI low:
  all: 34.62%
  t50: 28.46%
  t100 raw: 19.96%
  hard: 32.73%
gate: 25 / 25
```

Stage42-FJ robustness audit：

```text
robust domains: TrajNet, UCY
weak domains: none
robust sources: powered sources robust
weak horizons: TrajNet|100, UCY|50, UCY|100
dual-domain positive-safe claim: allowed
uniform horizon claim: not allowed
```

Stage42-FM row-level weak-horizon specialist：

```text
all/t50/t100raw/hard: 35.20% / 29.03% / 21.14% / 33.35%
weak horizons before: TrajNet|100, UCY|50, UCY|100
weak horizons after: TrajNet|100, UCY|100
```

Stage42-FN conservative easy guard：

```text
all/t50/t100raw/hard: 34.86% / 29.03% / 20.19% / 32.96%
weak horizons after: TrajNet|100, UCY|100
```

Stage42-FO gain/harm specialist：

```text
all/t50/t100raw/hard: 35.20% / 29.03% / 21.14% / 33.35%
TrajNet|100 switched rows: 1962
UCY|100: keep_fm
weak horizons after: TrajNet|100, UCY|100
```

结论：

- Stage42-FH/FI/FJ 让 source/domain robustness 成立。
- Stage42-FM 修复了 UCY|50。
- Stage42-FN/FO 证明剩余 TrajNet|100 与 UCY|100 不是简单 easy guard 或 gain/harm model 能修好的。
- 现在允许写 dual-domain/source robust protected raw-frame 2.5D evidence。
- 不能写 uniform horizon robust。

## 3. 失败路线总表

| 路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| Hard-class selector | Stage24 t50 约 -43.3%，easy degradation 约 11.33% | oracle label low-margin、class ambiguity、过度切换 easy | 改成 expected-FDE / regret-aware / fallback-safe selector |
| JEPA main contribution | non-collapse 但 downstream 无稳定 lift | representation objective 和部署目标错位 | 保留为 auxiliary，不作主 claim |
| SDD to external zero-shot | all/t50 大幅负迁移 | 坐标、scale、horizon、scene/goal、agent type mismatch | 做 row geometry、relative target、history/prototype |
| Latent adapter | MMD/cosine gap 缩小但预测不提升 | distribution alignment 不等于 target alignment | 不计为成功，只保留诊断 |
| External early selective transfer | all/hard 正，但 t50=0 | t50 被 all objective 淹没；缺 long-horizon context | Stage37 专门修 t50 |
| Bounded residual/correction | 没稳定超过 Stage37 | residual 容易伤 easy，strong floor 太强 | 不部署 correction |
| Ungated neural dynamics | easy/harm 不安全 | raw-frame/dataset-local 数据和目标不足 | neural 必须 protected |
| Scalar proximity/occupancy | all 有时提升但 safety/hard 不稳 | scalar target 不能完整表达 group dynamics | 转向 source/frame/horizon group-consistency |
| Temporal / waypoint repel | 单边改善，不 Pareto dominate | post-hoc 几何修复牺牲 ADE 或 hard | 用 constrained composer / fallback |
| Conservative easy guard | FN 保持安全但牺牲 all/t100/hard | guard 过保守，不解决 h100 low-margin | 需要更强 source/horizon support |
| Gain/harm specialist | FO 未修复 TrajNet|100 / UCY|100 | 现有 past/prototype/rollout features 不足以区分低 margin h100 | 下一步做 source/support audit 和更强 long-horizon context |

## 4. 成功路线总表

| 成功点 | 证据 | 价值 |
| --- | --- | --- |
| Stage26 SDD selector | t50 +14.58%，hard/failure +11.23%，easy +1.81% | SDD pixel raw-frame best deployable |
| Stage37 external selector | all +13.48%，t50 +8.46%，hard +15.54%，easy 0.041%，CI positive | external t50 transfer repaired deployable |
| Stage42-CO/CP bridge composer | all +3.02%，t50 +1.50%，t100raw +6.12%，hard +3.28%，bootstrap positive | full-waypoint bridge evidence |
| Stage42-CQ proximity guard | all +1.77%，t50 +1.07%，near@0.05 improves | safety-sensitive composer |
| Stage42-DL/DQ/ES/ET group consistency | all 约 +24.72%，t50 +22.36%，hard +23.89%，near@0.05 1.94% 到 1.38% | explicit group-consistency 有真实贡献 |
| Stage42-FE constrained composer | all/t50/hard 26.41% / 23.15% / 24.81%，near@0.05 1.32% | FC 精度和 DI safety 的可部署组合 |
| Stage42-FH/FI frozen policy | all/t50/t100raw/hard 34.98% / 28.97% / 20.57% / 33.10%；exact replay diff 0；bootstrap CI low positive | dual-domain positive-safe policy frozen |
| Stage42-FJ source audit | TrajNet/UCY robust，powered sources robust，uniform horizon blocked | claim 边界精确化 |
| Stage42-FM row-level specialist | all/t50/t100raw/hard 35.20% / 29.03% / 21.14% / 33.35%；UCY|50 repaired | row-level weak-horizon repair 有效但不充分 |
| Stage42-FN/FO negative evidence | stricter guard / gain-harm model 仍修不好 h100 weak slices | 证明剩余 blocker 不是简单 threshold/model 调整 |
| Stage42-FP h100 source/support audit | TrajNet|100 与 UCY|100 均有 source-family shift、稀疏 validation support、low-margin/low-headroom、easy-safety CI failure | 下一步需要补 source support 或更强 h100 long-horizon context，不应继续盲目调全局 policy |

## 5. 为什么现在还不能叫“真正世界模型成功”

因为真正世界模型至少需要：

1. 不只 selector policy，而是稳定的 dynamics head 能贡献预测能力。
2. 多数据集、多 domain、多 horizon 都稳定。
3. scene/goal/interaction 有独立正贡献。
4. hard/failure 提升同时 easy 不受损。
5. 统计证据与 replay/freeze/no-leakage 都完整。
6. 最好有 metric/seconds calibration 或至少明确 source-specific calibration。

当前已有：

- Stage26 SDD 成功。
- Stage37 external t50 成功。
- Stage42-FH/FI source/domain robust 成功。
- Stage42-FJ/FK/FL/FM/FN/FO 把 horizon 边界审清楚。

当前缺口：

- TrajNet|100 和 UCY|100 仍是 weak horizons。
- t100 仍只能 raw-frame diagnostic。
- 全局 metric/seconds claim 仍不允许。
- JEPA 没有稳定 downstream lift。
- 无保护 neural dynamics 不能部署。
- uniform horizon robust claim 仍 blocked。

## 6. 当前最强模型/策略怎么用

如果目标是部署稳定预测：

1. SDD 上用 Stage26 cost-aware selector。
2. external t50 transfer 用 Stage37 causal-history + goal-prototype safe selector。
3. source-level protected evidence 用 Stage42-FH/FI frozen policy family。
4. 若强调 proximity safety，用 Stage42-CQ/FE safety-sensitive guarded variant。
5. Stage42-FM/FO 只能作为 weak-horizon repair/diagnostic，不允许写 uniform horizon success。

## 7. 下一步最短路径

下一步不是再盲目堆模型，而是解决剩余 h100 weak horizon blocker：

1. 对 TrajNet|100 / UCY|100 做 source/support audit：拆到 source、scene、easy/hard、oracle margin、validation support。
2. 如果是 support 不足，补 train-only source 或 stronger held-out h100 support。
3. 如果是 feature 不足，增加更长 history、neighbor trajectory、goal prototype sequence、source-specific h100 context。
4. 如果是 label low-margin，改成 abstain/fallback-safe policy，而不是强行学习 best label。
5. 如果是 horizon 本身不稳定，保留 t100 diagnostic，不写 deployable horizon claim。

## 8. 最终 verdict

```text
项目是否有真实进展：是
当前是否是 true 3D world model：否
当前是否是 foundation world model：否
当前是否是 metric / seconds-level：否
当前是否可称 protected 2.5D multi-agent world-state candidate：是
SDD best deployable：Stage26 selector
external t50 best deployable：Stage37 selector
source/domain robust protected policy：Stage42-FH/FI family
uniform horizon robust：否
Stage5C ready：否
SMC ready：否
最重要 blocker：TrajNet|100 与 UCY|100 low-margin / support / context blocker
```

## 9. 关联文件

- `README_RESULTS.md`
- `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`
- `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- `research_state.json`
- `outputs/stage42_long_research/fh_horizon_gain_harm_specialist_stage42.md`
- `outputs/stage42_long_research/fh_horizon_row_switch_specialist_stage42.md`
- `outputs/stage42_long_research/fh_horizon_weak_slice_forensics_stage42.md`
- `outputs/stage42_long_research/fh_source_robustness_audit_stage42.md`

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
