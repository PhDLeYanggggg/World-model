# M3W 长期目标工作总结：做过什么、试过什么、失败原因、成功证据

更新时间：2026-05-27  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已生成报告、gate、README、`research_state.json`，并纳入最新 Stage42-DP / DQ / DR 证据刷新。  
最近完整测试记录：Stage42-DR 后 `.venv-pytorch/bin/python -m pytest tests` 通过，`671 passed in 31.83s`。  

这份 README 是给人的总账。它回答：在 M3W 这个长期目标里到底做了什么、尝试了哪些路线、哪些失败了、失败原因是什么、哪些成功了、现在模型大概是什么质量，以及哪些结论仍然禁止写。

## 0. 一句话结论

M3W 已经从早期的 SDD-only 2.5D trajectory scaffold，推进到一个有 SDD 与 external top-down dataset-local raw-frame 正迁移证据的 protected multi-agent world-state candidate。

但当前仍然不是：

- true 3D world model
- large-scale foundation world model
- global metric predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- latent generative Stage5C model
- SMC-ready model

当前最诚实的定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

当前 best deployable / best evidence 分层：

| 用途 | 当前最强结果 | 当前结论 |
| --- | --- | --- |
| SDD pixel-space official benchmark | Stage26 cost-aware selector | SDD t+50 和 hard/failure 已稳定超过 strong causal floor，但仍是 pixel raw-frame。 |
| External t+50 selector | Stage37 history + goal-prototype safe transfer policy | 第一次让 external all / t50 / hard/failure 同时为正且 easy 安全。 |
| Protected neural/world-state candidate | M3W-Neural v1 / Stage41-42 protected composer family | 有 protected neural / full-waypoint / runtime replay 证据，但仍依赖 safety floor。 |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware composer guard | 牺牲部分 ADE 增益，修复 near-collision caveat。 |
| Source-level full-waypoint runtime policy | Stage42-DL / DQ group-consistency full-waypoint runtime policy | 可调用、exact replay、source-level promotable；不是 global ungated replacement。 |
| Paper/evidence package | Stage42-DR post-DP/DQ refresh | 论文包已同步 latest negative/positive evidence；仍不是 full A-journal-ready / foundation claim。 |

## 1. 永久边界

所有结果必须继续遵守下面边界：

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
- 不用 test endpoints 建 goals。
- 不用 test metrics 调 threshold。
- 无保护 neural dynamics 不部署。

## 2. 实际做过的主要路线

| 路线 | 做了什么 | 结果 | 当前解释 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建立 per-agent multi-agent 2.5D world-state、causal baseline fallback、failure diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 3D/foundation。 |
| Stage18 / Stage19 JEPA + WAM-style data | 做 SAM-JEPA-2.5D、WAM-style data registry、simulation/top-down/ego-video 分角色。 | JEPA non-collapse，但 downstream heads 没有 lift。 | non-collapse 不等于 selector/failure/correction/t50 改善。 |
| Stage20 / Stage21 数据采集 | 合法登记 OpenTraj / SDD，验证数据路径，转换 SDD world-state shards。 | SDD 数据基座成功。 | SDD no-leakage、causal velocity、raw-frame horizon 被显式审计。 |
| Stage22 / Stage23 SDD benchmark | 构建 SDD scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、strong causal baselines。 | SDD official pixel-space benchmark 成功。 | quick-plus / medium 区分清楚，没有把 quick 包装成 full。 |
| Stage24 hard-class selector | 在 true medium SDD index 上训练 validation-selected selector。 | 失败。 | oracle headroom 大，但 hard classification 过度切换，easy 被伤害。 |
| Stage25 / Stage26 cost-aware selector | 改成 expected-FDE、regret-aware、confidence-gated、fallback-safe selector。 | 成功。 | 修复 low-margin label 与 easy over-switch，是 SDD best deployable。 |
| Stage31 / Stage32 external transfer / domain alignment | 做 SDD -> external zero-shot、normalization、CORAL、latent adapter、mixed-domain selector。 | 失败。 | 坐标、horizon、scene/goal、agent type、scale/homography 不兼容。 |
| Stage33 / Stage34 row geometry | 构建 coordinate-invariant features、relative targets、external row geometry、train-only goals。 | 局部正信号但不可部署。 | t50/hard 有提升，但 all/easy 不稳。 |
| Stage35 selective transfer | 扩容 external 数据，建 hard/easy/failure labels、gain/harm/easy gate。 | 部分成功。 | all/hard/easy 过，但 t+50 仍是 0。 |
| Stage36 t50 repair | horizon-specific selector、t50 policy search、t50 curriculum。 | 失败。 | t50 有 oracle headroom，但缺可靠 causal history / goal prototype / switchability signal。 |
| Stage37 causal history + goal prototypes | 建 K=8/16/32/64 past-only history windows、scene-agnostic goal prototypes、t50 gain/harm/failure、安全 conformal policy。 | 成功。 | external t50 被修复为可部署正迁移。 |
| Stage38 bounded correction | 在 Stage37 保护下训练 bounded delta / correction head。 | 不部署。 | correction 没有稳定超过 Stage37，普通 residual 容易伤 easy。 |
| Stage39 / Stage40 neural dynamics | 训练 Causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 训练了，但不部署。 | 无保护 neural 灾难性失败；受保护 neural 没稳定超过 Stage37。 |
| Stage41 protected neural breakthrough | 做 composite-tail、safe-switch、bounded neural dynamics、teacher floor、多 seed / bootstrap / source-heldout。 | 成功形成 protected neural candidate。 | 神经贡献成立在 Stage37/teacher safety floor 下，不是无保护 neural。 |
| Stage42 long research | 做 full-waypoint、source-level ablation、paper package、runtime replay、proximity guard、group consistency、reviewer replay。 | 成功形成 evidence package。 | 支持 protected 2.5D world-state paper candidate，不支持 foundation/metric/Stage5C/SMC claim。 |

## 3. 哪些路线失败了，为什么失败

| 失败路线 | 失败表现 | 根因 | 后续处理 |
| --- | --- | --- | --- |
| JEPA-only / SAM-JEPA-2.5D | non-collapse，但 selector/failure/correction/t50 没有稳定 lift。 | 表征目标和部署目标错位；non-collapse 不等于 downstream usefulness。 | 保留为 auxiliary / diagnostic，不写主贡献。 |
| Stage24 hard-class selector | t+50 improvement 约 `-43.3%`，easy degradation 约 `11.33%`。 | low-margin oracle labels、label ambiguity、confidence calibration 差、过度切换。 | 改成 expected-FDE / regret / gain-harm / fallback-safe selector。 |
| SDD -> external zero-shot | all / t50 严重负迁移。 | SDD pixel-space 和 external dataset-local 坐标、horizon、goal、agent type、scale 不兼容。 | 建 external geometry、relative target、history window、goal prototype。 |
| 普通 normalization / CORAL / latent adapter | 分布距离缩小，但预测没有 lift。 | 统计分布对齐不等于 decision target 对齐。 | 转向 row-level geometry 和 selective transfer。 |
| Stage35/36 t50 | all/hard/easy 正，但 t50=0。 | t50 有约 `22.98%` oracle headroom，但特征不足，policy 不敢安全切。 | Stage37 加 past-only history、goal prototypes、switchability。 |
| ordinary residual / bounded correction | hard 有时变好，但不稳定超过 Stage37。 | selected baseline 已强，residual 容易伤 easy；bounded 后收益不足。 | 不部署 correction，只保留 diagnostic。 |
| ungated Transformer / Hybrid | neural without fallback 灾难性失败。 | 无保护 neural 会伤 easy，不能稳定替代 Stage37。 | 所有 neural output 必须经过 Stage37/teacher safety floor。 |
| goal/scene gated expert | 没超过 baseline-family control。 | 当前 goal/scene proxy 没在 source-level ridge/full-waypoint 协议下提供独立增量。 | 写成 mixed / diagnostic，不写主贡献。 |
| neighbor/interaction gated expert | kNN graph rows 足够，但低于 baseline-family control。 | 当前 hand-built neighbor/interaction 特征不足以独立超过 rollout-family context。 | 写成 auxiliary / diagnostic，下一步需要更强 graph/scene-rich protocol。 |
| sequence/graph residual context | Stage42-AR/AS rerun 显示 all/t50/hard 均低于 baseline-family first-stage control。 | 当前 residual target 没抽出独立 scene/goal/interaction value；baseline-family rollout context 已解释主要可学空间。 | Stage42-DP 决定关闭当前 sequence/graph residual context protocol。 |
| t100 global claim | 多次局部正，最终 source-CV / easy gate 不稳。 | independent t100 source support 不足；ETH_UCY/TrajNet/UCY legal/source/time blocker 未完全关闭。 | t100 保持 raw-frame diagnostic，不能写 deployable/global claim。 |
| metric / seconds claim | 有些 source 有 H/FPS evidence，但不全局闭环。 | homography方向、坐标 convention、annotation stride、scale、license/source confirmation 不完整。 | 继续禁止 global metric / seconds-level claim。 |

## 4. 哪些路线成功了，有什么具体证据

### 4.1 Stage26：SDD cost-aware selector 成功

Stage26 修复了 Stage24 hard-class selector 的失败，把任务从“预测 best baseline class”改成“预测 expected FDE / risk，再做 conservative fallback”。

关键结果：

```text
t+50 improvement: about +14.58%
hard/failure improvement: about +11.23%
easy degradation: about 1.81%
```

意义：

- 这是 SDD pixel raw-frame 上的 current best deployable selector。
- 它证明 cost-aware / regret-aware / fallback-safe policy 比 hard classification 更可靠。
- 它仍然不是 metric / 3D / foundation。

### 4.2 Stage37：external t+50 正迁移修复成功

Stage35 已经 all/hard/easy 正，但 t50=0；Stage36 也没修好。Stage37 加入 past-only history window 和 scene-agnostic goal prototypes 后成功。

关键结果：

```text
external rows: 66303
all improvement: +13.48%
t+50 improvement: +8.46%
t+50 bootstrap CI: [+7.69%, +9.15%]
hard/failure improvement: +15.54%
easy degradation: 0.041%
gates: 16 / 16
verdict: stage37_t50_transfer_repaired_deployable
```

意义：

- 这是第一次 external all / t50 / hard/failure / easy 同时过 gate。
- 成功机制不是盲目 neural，而是：past-only history + goal prototype + gain/harm/failure switchability + conservative fallback。
- 仍然是 dataset-local raw-frame external，不能写 metric/seconds。

### 4.3 Stage41 / M3W-Neural v1：protected neural candidate 成立

Stage39/40 的无保护 neural 没超过 Stage37 后，Stage41 转向 protected neural：bounded neural dynamics + teacher safety floor + composite-tail safe switch。

关键结果：

```text
M3W-Neural v1 verdict: composite-tail safe-switch bounded neural dynamics candidate
gates: 41 / 41
all ADE improvement vs Stage37 floor: +21.03%
t+50 ADE improvement vs Stage37 floor: +13.65%
t+100 raw-frame diagnostic ADE improvement: +14.69%
hard/failure ADE improvement: +20.38%
easy degradation: 0.00%
positive external domains: 3
bootstrap evidence pass: true
multiseed replication pass: true
pure UCY source-heldout gate: true
```

意义：

- 这是目前最强的 protected neural/world-state candidate。
- 它不是“无保护 neural 单独成功”，而是 neural 在 Stage37/teacher safety floor 下提供增益。
- JEPA deployable path 被关闭：`disable_jepa_in_deployable_path`。

### 4.4 Stage42-CQ / CR：proximity guard 修复 safety caveat

Stage42-CO/CP 的 no-guard composer 准确率更高，但 near-collision@0.05 变差。Stage42-CQ 加 validation-selected proximity guard，Stage42-CR 做 Pareto ablation。

关键结果：

```text
no_proximity_guard:
  all +3.02%
  t50 +1.50%
  t100 raw +6.12%
  hard +3.28%
  near@0.05 worsens by +0.34%

proximity_guard:
  all +1.77%
  t50 +1.07%
  t100 raw +3.48%
  hard +1.93%
  near@0.05 improves by -0.06% vs endpoint-linear
```

意义：

- no-guard 是 accuracy-priority diagnostic。
- proximity_guard 是 safety-sensitive deployable bridge/shape policy。
- 这条结果说明“准确率最大”不等于“部署最安全”。

### 4.5 Stage42-DI / DJ / DK / DL / DQ：group-consistency full-waypoint runtime 证据

Stage42-DI 修复 group-consistency full-waypoint；DJ 冻结 policy；DK 做 exact replay；DL 做 runtime API；DQ 做 full-waypoint promotion checkpoint。

关键结果：

```text
runtime rows: 47458
switch exact match: true
selected_xy max abs diff: 0.0
selected ADE/FDE max abs diff: 0.0
all improvement: +24.72%
t50 improvement: +22.36%
t100 raw diagnostic improvement: +14.35%
hard/failure improvement: +23.89%
easy degradation: -25.63%
base near@0.05: 1.94%
final near@0.05: 1.38%
Stage42-DQ gate: 24 / 24
verdict: stage42_dq_full_waypoint_promotion_checkpoint_pass
```

意义：

- frozen policy 可以通过 runtime API 精确复现。
- 这支持 protected source-level group-consistency full-waypoint runtime policy。
- 它不支持 ungated/global primary full-waypoint replacement。
- 它不是 Stage5C，不是 SMC。

### 4.6 Stage42-DP：当前 sequence/graph residual context protocol 关闭

Stage42-DP fresh 复核 Stage42-AR sequence-context 和 Stage42-AS graph-context。结论是当前 residual context protocol 不值得继续重复。

关键结果：

```text
baseline-family control:
  all +28.78%
  t50 +31.54%
  t100 raw diagnostic +14.28%
  hard/failure +27.58%

best context delta vs baseline-family:
  all -2.30%
  t50 -8.31%
  hard/failure -2.62%

positive_context_rows: []
verdict: stage42_dp_context_model_closure_pass
closure_decision: close_current_sequence_graph_residual_context_protocol
```

意义：

- 不是说 context 永远没用。
- 是说当前 sequence/graph residual target 不能在 baseline-family control 之上抽出独立 value。
- 下一步如果重开 context，必须换 target、换数据支持或换架构，不能重复同一 protocol。

### 4.7 Stage42-DR：paper package 已同步最新证据

Stage42-DR 把 Stage42-DP / DQ 的正负证据刷新进 paper package。

关键结果：

```text
Stage42-DR gate: 14 / 14
verdict: stage42_dr_post_dq_paper_refresh_pass
updated paper files: 9
pytest after DR: 671 passed
```

意义：

- 论文包明确写入 context residual protocol negative evidence。
- 论文包明确写入 protected group-consistency full-waypoint runtime evidence。
- Claim 仍然严格：不是 true 3D，不是 foundation，不是 metric/seconds，不执行 Stage5C/SMC。

## 5. 当前模型质量大概是什么水平

当前质量可以这样描述：

```text
强于普通 baseline selector 的 protected 2.5D multi-agent world-state candidate。
有 SDD 与 external dataset-local raw-frame 证据。
有 source-level full-waypoint runtime replay。
有 bootstrap、multi-seed、proximity/safety guard、paper package。
但仍依赖 Stage37/teacher floor，未形成无保护神经动力学，未形成 metric/seconds-level/true-3D/foundation claim。
```

更直接：

- 已经不是 early demo。
- 已经不是只会 fallback 的 scaffold。
- 已经有 external 正迁移、t50 修复、full-waypoint runtime、exact replay 和 safety guard。
- 但也还不是“真正物理 3D 世界模型”。
- 还不能说 foundation-track prototype，除非后续完成更多独立外部数据、多模态预训练贡献、metric/time calibration、无保护或更少 floor 依赖的神经动力学。

## 6. 当前 best deployable 是谁

当前最好写法：

```text
best deployable family = protected M3W-Neural / full-waypoint policy family under Stage37 / teacher safety floor
runtime-ready source-level variant = Stage42-DL/DQ group-consistency full-waypoint runtime policy
safety-sensitive bridge/shape variant = Stage42-CQ proximity-aware composer guard
SDD-specific best = Stage26 cost-aware selector
external t50 floor = Stage37 history + goal-prototype safe transfer policy
```

仍然不能说：

- neural without fallback 已经超过 Stage37。
- JEPA 是主贡献。
- Transformer dynamics 单独成为主模型。
- goal/scene 或 neighbor/interaction 是当前独立主贡献。
- t100 已经 global deployable。
- metric / seconds-level 已闭环。

## 7. 为什么有些看起来强的结果仍不能部署

### 7.1 Oracle headroom 大，不等于模型成功

Stage24 已证明：即使 selector oracle headroom 很大，hard-class selector 也可能因为 low-margin labels 和 easy over-switch 而失败。部署必须看：

- selected model vs strongest floor；
- easy degradation；
- harm over fallback；
- confidence / predicted gain；
- validation-only threshold；
- test 只评一次。

### 7.2 JEPA non-collapse，不等于 world model 成功

JEPA 多次 non-collapse，但没有稳定提升 selector/failure/correction/t50。当前只能写：

```text
JEPA representation diagnostic / auxiliary
```

不能写：

```text
JEPA generative world model
JEPA main contribution
latent rollout success
```

### 7.3 Full-waypoint 正提升，不等于 Stage5C

Stage42 full-waypoint 是 protected supervised full-waypoint / policy family，不是 latent generative rollout。future waypoints 是 loss/eval labels，不是 inference input。Stage5C 没执行，SMC 没启用。

### 7.4 t100 raw diagnostic，不等于 seconds-level long horizon

t100 仍是 raw-frame horizon。没有全局 FPS/stride/homography/metric scale 闭环，就不能写 seconds-level 或 metric long horizon。

## 8. 可以写进论文/报告的主 claim

可以写，但必须带边界：

1. 在严格 no-leakage、past-only input、validation-only policy selection 下，M3W 从 SDD-only scaffold 推进到 external dataset-local raw-frame positive transfer。
2. Cost-aware / regret-aware / fallback-safe baseline policy 比 hard-class selector 更稳，能避免 easy over-switch。
3. External t50 修复依赖 past-only history window、scene-agnostic goal prototypes、gain/harm/easy safe switching。
4. Protected neural/full-waypoint policy family 在 Stage37/teacher floor 下提供正向 world-state evidence。
5. Baseline-family rollout context 是当前最稳定的 dominant mechanism。
6. Proximity-aware guard 在牺牲部分 accuracy 的同时修复 near-collision caveat，是安全部署的重要机制。
7. Group-consistency full-waypoint policy 已经从 repair、freeze、replay 推进到 runtime API exact replay。

## 9. 不能写的主 claim

不能写：

1. M3W 是 true 3D world model。
2. M3W 是 foundation world model。
3. SDD/external 结果是统一 metric result。
4. raw-frame t50/t100 是 seconds-level horizon。
5. JEPA 是生成式 world model。
6. Transformer/Hybrid 无保护超过 Stage37。
7. goal/scene 或 neighbor/interaction 当前是独立主贡献。
8. t100 是 global deployable success。
9. Stage5C 已执行或 ready。
10. SMC 已启用或 ready。

## 10. 当前最重要的未解决问题

| 问题 | 当前状态 | 最短路径 |
| --- | --- | --- |
| true metric / seconds-level claim | blocked | source-specific homography/FPS/stride/scale/legal conversion 全闭环。 |
| t100 global deployable | blocked / diagnostic | 增加 independent t100-capable sources，做 source-CV 和 easy guard。 |
| JEPA/Transformer 独立主贡献 | not proven | 需要更强 graph/scene-rich neural protocol，且必须超过 baseline-family control。 |
| goal/scene contribution | mixed / diagnostic | 需要更可靠 scene packs / train-only goals / source-level held-out evidence。 |
| neighbor/interaction contribution | mixed / diagnostic | 当前 kNN/graph hand-built 特征不够，需要 real graph-neural or interaction-token protocol。 |
| foundation-track claim | not ready | 至少需要多数据集、多域、多模态预训练、跨场景泛化、metric/time calibration。 |
| Stage5C / SMC | forbidden | 只能未来生成 plan，不能执行，除非 gates 明确通过且用户确认。 |

## 11. 下一步建议

1. **冻结当前 evidence package**：以 Stage42-DQ/DR 为当前 source-level full-waypoint runtime / paper package checkpoint，避免继续用 test 调参。
2. **补 source/legal/time closure**：优先关闭 ETH_UCY / TrajNet / UCY 的 legal/source/time/t100 blocker。
3. **如果重开 context，不要重复同一 residual protocol**：Stage42-DP 已经关闭当前 sequence/graph residual route；下一次必须换 target 或换架构。
4. **继续保留 Stage37/teacher safety floor**：除非无保护 neural 通过 easy、hard、t50、all、proximity gates，否则不要部署。
5. **把 full-waypoint runtime 做成可复现包**：保留 frozen policy hash、cache hash、schema hash、reviewer replay commands、pytest 状态。

## 12. 当前总判定

```text
项目是否跑通：是
是否训练/验证过 neural world dynamics：是，但必须 protected
是否超过 Stage37：受保护 policy family 在多个协议下超过；无保护 neural 没有通过
当前 best deployable：Stage37/teacher-floor protected M3W policy family；source-level runtime 用 Stage42-DL/DQ group-consistency full-waypoint
是否 true 3D：否
是否 foundation：否
是否 metric/seconds-level：否
Stage5C 是否执行：否
SMC 是否启用：否
当前 verdict：protected dataset-local/raw-frame 2.5D multi-agent world-state candidate with source-level full-waypoint runtime evidence
```

<!-- STAGE42_DS_SOURCE_CONVERSION_READINESS_RECHECK:START -->
## Stage42-DS Source Conversion Readiness Recheck

- source: `fresh_local_path_scan_after_stage42_do`
- role: separates local raw-path/derived-cache hints from legal conversion readiness.
- gate: `13 / 13`; verdict `stage42_ds_source_conversion_readiness_recheck_pass`.
- targets checked: `7`; raw-path found: `6`; derived-cache found: `6`.
- technical preflight possible: `6`; conversion-ready targets: `0`.
- No dataset was converted or evaluated in this step; legal/source blockers remain preserved.
- report: `outputs/stage42_long_research/source_conversion_readiness_recheck_stage42.md`.
<!-- STAGE42_DS_SOURCE_CONVERSION_READINESS_RECHECK:END -->

<!-- STAGE42_DT_RAW_SOURCE_PARSEABILITY_DRY_RUN:START -->
## Stage42-DT Raw Source Parseability Dry Run

- source: `fresh_sample_only_raw_source_parseability_dry_run`
- role: sample-only technical parser preflight after Stage42-DS; no conversion, no evaluation.
- gate: `11 / 11`; verdict `stage42_dt_raw_source_parseability_dry_run_pass`.
- dry-run parseable targets: `4`; targets with homography/time hints: `2`.
- legal conversion ready targets: `0`; generated rows: `0`.
- Homography/time hints remain hints only; no metric/seconds claim is made.
- report: `outputs/stage42_long_research/raw_source_parseability_dry_run_stage42.md`.
<!-- STAGE42_DT_RAW_SOURCE_PARSEABILITY_DRY_RUN:END -->

<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:START -->
## Stage42-DU Raw Source Time/Geometry Hint Audit

- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`
- role: extracts H/FPS/stride hints only; no conversion, no evaluation, no metric/seconds claim.
- gate: `14 / 14`; verdict `stage42_du_raw_source_time_geometry_hint_audit_pass`.
- H-hint targets: `2`; time-hint targets: `3`; stride-hint targets: `4`.
- metric/time subset hint targets: `2`; legal conversion ready targets: `0`.
- report: `outputs/stage42_long_research/raw_source_time_geometry_hint_audit_stage42.md`.
<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:END -->

<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:START -->
## Stage42-DV Calibration Candidate Manifest

- source: `fresh_synthesis_from_stage42_du_bn`
- role: ranks source-specific calibration candidates from raw H/FPS/stride hints; no conversion/evaluation.
- gate: `13 / 13`; verdict `stage42_dv_calibration_candidate_manifest_pass`.
- source-specific candidate targets: `2`; time/stride candidate targets: `1`.
- conversion-ready targets: `0`; global metric/seconds claim remains `False`.
- report: `outputs/stage42_long_research/calibration_candidate_manifest_stage42.md`.
<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:END -->
