# M3W 当前目标总账：做过什么、试过什么、失败原因、成功证据

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总 Stage18-Stage42 已提交/已生成报告、gate、README、`research_state.json`；最新可核验证据包含 Stage42-DI/DJ/DK/DL group-consistency full-waypoint policy chain，以及 Stage42-CX/CZ provenance / paper-freeze manifest refresh。  
测试状态：`.venv-pytorch/bin/python -m pytest tests` 已通过，`655 passed in 32.28s`。  

这份 README 是给人的总账，不是论文宣传稿。它回答：在 M3W 长期目标内到底做了什么、尝试了哪些路线、哪些失败了、为什么失败、哪些成功了、当前最强可部署模型是谁、哪些结论仍然不能写。

## 0. 一句话结论

M3W 已经从 SDD-only 2.5D trajectory scaffold，推进到一个有 SDD 与 external top-down dataset-local raw-frame 正迁移证据的 protected multi-agent world-state candidate。

当前最诚实的系统定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

当前仍然不能说：

```text
true 3D world model
large-scale foundation world model
global metric predictor
seconds-level long-horizon predictor
ungated neural dynamics model
latent generative Stage5C execution
SMC-ready model
```

当前 best deployable / best evidence 分层：

| 用途 | 当前 best deployable / best evidence | 解释 |
| --- | --- | --- |
| SDD pixel-space | Stage26 cost-aware selector | SDD 内 t+50 与 hard/failure 有稳定提升，仍是 pixel raw-frame。 |
| External t+50 selector floor | Stage37 safety-selected t50 transfer policy | 第一次让 external all / t50 / hard 同时为正且 easy 安全。 |
| Protected neural/world-state package | M3W-Neural v1 / Stage41-42 protected composer family | 有 protected neural evidence，但不是无保护 neural，也不是 foundation。 |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware guarded composer | 修复 proximity caveat，保留 all/t50/t100 raw-frame/hard 正提升。 |
| Group-consistency full-waypoint runtime | Stage42-DL group-consistency runtime policy API | frozen policy 可调用并 exact replay，仍是 protected raw-frame policy，不是 Stage5C/SMC。 |
| 当前 paper-freeze package | Stage42-CZ candidate manifest | 证据包可冻结为 candidate，claim 边界仍严格。 |

## 1. 永久边界

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
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

## 2. 我们实际走过的路线

| 路线 | 做了什么 | 结果 | 当前解释 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建 per-agent multi-agent 2.5D world-state、causal baseline fallback、failure diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 3D/foundation。 |
| Stage18/19 JEPA / WAM-style data | SAM-JEPA-2.5D、WAM-style registry、simulation / top-down / ego video 分角色。 | JEPA non-collapse，但 downstream lift 未证明。 | non-collapse 不等于 selector/failure/correction/t50 改善。 |
| Stage20/21 SDD data | 合法数据登记、OpenTraj/SDD 路径、SDD world-state shards。 | 成功建立 SDD official pixel-space 数据基座。 | SDD 数据、no-leakage、causal velocity 被显式审计。 |
| Stage22/23 SDD benchmark | scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、baselines。 | SDD benchmark 成功；quick-plus/medium 区分清楚。 | 仍是 pixel raw-frame，不是 metric / seconds。 |
| Stage24 hard-class selector | medium SDD 上训练 validation-selected selector。 | 失败。 | oracle headroom 大，但 hard classification 过度切换。 |
| Stage25/26 cost-aware selector | expected-FDE / regret-aware / confidence-gated fallback selector。 | 成功。 | 修复低-margin label 与 easy 过切换。 |
| Stage31/32 external zero-shot / alignment | SDD -> external transfer，normalization、CORAL、latent adapter、mixed-domain selector。 | 失败。 | 坐标、horizon、scene/goal、agent type、scale/homography 不一致。 |
| Stage33/34 external geometry | 坐标不变 features、relative targets、external row geometry、train-only goals。 | 局部正信号，不可部署。 | t50/hard 有提升，但 all/easy 不稳。 |
| Stage35 selective transfer | external hard/easy/failure labels、gain/harm/easy gate。 | 部分成功。 | all/hard/easy 过，但 t+50 仍为 0。 |
| Stage36 t50 repair | horizon-specific selector / t50 policy search / curriculum。 | 失败。 | t50 有 oracle headroom，但缺 past-only history / goal prototype / switchability signal。 |
| Stage37 causal history + goal prototype | K=8/16/32/64 history windows、scene-agnostic goal prototypes、t50 gain/harm/failure、安全 conformal policy。 | 成功。 | external t50 被修复为可部署正迁移。 |
| Stage38 bounded correction | Stage37 保护下训练 bounded delta / correction head。 | 不部署。 | correction 没稳定超过 Stage37，普通 residual 容易伤 easy。 |
| Stage39/40 Transformer / JEPA / Hybrid neural | Causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 训练了，但不部署。 | 无保护 neural 不安全；受保护 neural 没有稳定超过 Stage37。 |
| Stage41 protected neural breakthrough | composite-tail、safe-switch、bounded neural dynamics、teacher floor。 | 成功形成 protected neural candidate。 | 强证据依赖 Stage37/teacher safety floor。 |
| Stage42 long research | full-waypoint、source-level、ablation、paper package、runtime replay、proximity guard、group consistency。 | 成功形成 evidence package，边界更清楚。 | 支持 protected 2.5D paper candidate，不支持 foundation/metric/Stage5C/SMC claim。 |

## 3. 关键失败和原因

| 失败路线 | 失败表现 | 根因 | 后续处理 |
| --- | --- | --- | --- |
| JEPA-only / SAM-JEPA-2.5D | non-collapse，但 selector/failure/correction/t50 没有稳定 lift。 | 表征目标和部署目标错位；non-collapse 不等于 downstream usefulness。 | 保留为 auxiliary / diagnostic，不写主贡献。 |
| Stage24 hard-class selector | t+50 improvement = -43.3%，easy degradation = 11.33%。 | low-margin oracle labels、label ambiguity、confidence calibration 差、过度切换。 | 改成 expected-FDE / regret / gain-harm / fallback-safe selector。 |
| SDD -> external zero-shot | all / t50 严重负迁移。 | SDD pixel-space 和 external dataset-local 坐标、horizon、goal、agent type、scale 不兼容。 | 建 external geometry、relative target、history window、goal prototype。 |
| 普通 normalization / CORAL / latent adapter | 分布距离缩小，但预测没有 lift。 | 统计分布对齐不等于 decision target 对齐。 | 转向 row-level geometry 和 selective transfer。 |
| Stage35/36 t50 | all/hard/easy 正，但 t50=0。 | t50 有约 22.98% oracle headroom，但特征不足，policy 不敢安全切。 | Stage37 加 past-only history、goal prototypes、switchability。 |
| ordinary residual / bounded correction | hard 有时变好，但不稳定超过 Stage37。 | selected baseline 已强，residual 容易伤 easy；bounded 后收益不足。 | 不部署 correction，只保留 diagnostic。 |
| ungated Transformer / Hybrid | neural without fallback 灾难性失败。 | 无保护 neural 会伤 easy，不能稳定替代 Stage37。 | 所有 neural output 必须经过 Stage37/teacher safety floor。 |
| goal/scene gated expert | 没超过 baseline-family control。 | 当前 goal/scene proxy 没在 source-level ridge/full-waypoint协议下提供独立增量。 | 写成 mixed / diagnostic，不写主贡献。 |
| neighbor/interaction gated expert | kNN graph rows 足够，但低于 baseline-family control。 | 当前 hand-built neighbor/interaction 特征不足以独立超过 rollout-family context。 | 写成 auxiliary / diagnostic，下一步需要更强 graph/scene-rich protocol。 |
| t100 global claim | 多次局部正，最终 source-CV / easy gate 不稳。 | independent t100 source support 不足；ETH_UCY/TrajNet/UCY legal/source/time blocker 未完全关闭。 | t100 保持 raw-frame diagnostic，不能写 deployable/global claim。 |
| metric / seconds claim | 有些 source 有 H/FPS evidence，但不全局闭环。 | homography方向、坐标 convention、annotation stride、scale、license/source confirmation 不完整。 | 继续禁止 global metric / seconds-level claim。 |

## 4. 关键成功证据

| 阶段 | 结果 | 指标 |
| --- | --- | --- |
| Stage26 SDD selector | feature-complete cost-aware selector 过 gate | t+50 +14.58%；hard/failure +11.23%；easy degradation 1.81%。 |
| Stage37 external t50 repair | external deployable selector candidate | all +13.48%；t50 +8.46%；t50 bootstrap CI [+7.69%, +9.15%]；hard/failure +15.54%；easy degradation 0.041%；16/16 gates。 |
| Stage40 neural optimization | 训练了 neural，但没有超过 Stage37 | best neural with fallback 等同 Stage37 子集；neural without fallback all -126.36%、t50 -292.10%、easy degradation 612.31%。 |
| M3W-Neural v1 / Stage41 | protected neural candidate 成立 | all ADE +21.03%；t50 ADE +13.65%；t100 raw diagnostic +14.69%；hard/failure +20.38%；easy 0。 |
| Stage42-CO common-validation composer | endpoint-linear 与 full-waypoint row 对齐后，validation-only composer 正提升 | vs endpoint-linear: all +3.02%；t50 +1.50%；t100 raw +6.12%；hard +3.28%；easy +0.25%。 |
| Stage42-CP bootstrap/safety | 2000 bootstrap 支持 CO composer | all CI [+2.64%, +3.37%]；t50 CI [+0.90%, +2.09%]；t100 raw CI [+5.39%, +6.94%]；hard CI [+2.90%, +3.68%]。 |
| Stage42-CQ proximity guard | 修复 near-collision caveat | all +1.77%；t50 +1.07%；t100 raw +3.48%；hard +1.93%；near-collision@0.05 vs endpoint-linear -0.06%。 |
| Stage42-CR Pareto ablation | 明确 no-guard accuracy vs guarded safety tradeoff | no-guard 更准但 near@0.05 +0.34%；guarded 稍弱但 near@0.05 -0.06%。 |
| Stage42-DI group-consistency full-waypoint repair | all-agent group-consistency repair 成功 | gates 17/17；all +24.72%；t50 +22.36%；t100 raw +14.35%；hard +23.89%；group safety 改善。 |
| Stage42-DJ frozen policy | 冻结 group-consistency full-waypoint policy | gates 22/22；policy artifact hash 固化；仍不调 test threshold。 |
| Stage42-DK replay | frozen policy replay 精确复现 | gates 34/34；exact replay，selected_xy/ADE/FDE diff 0。 |
| Stage42-DL runtime API | frozen group-consistency policy 变成 callable runtime policy | gates 30/30；real batch rows 47,458；switch exact match true；near@0.05 从 1.94% 降到 1.38%。 |
| Stage42-CX provenance refresh | evidence provenance 纳入 DI/DJ/DK/DL | gates 20/20；25 artifacts audited；fresh/cached 来源标记完整。 |
| Stage42-CZ paper-freeze manifest refresh | paper freeze candidate manifest 纳入 group-consistency policy artifact | gates 15/15；87 files hashed；candidate_clean。 |

## 5. 当前 best deployable 是谁

当前最好写法：

```text
best deployable = protected M3W-Neural / full-waypoint policy family under Stage37 / teacher safety floor
latest runtime-ready variant = Stage42-DL group-consistency full-waypoint runtime policy API
```

更具体：

- SDD 场景：Stage26 cost-aware selector 仍是 SDD best deployable baseline。
- External selector floor：Stage37 policy 是第一个 external all/t50/hard/easy 全过 gate 的 deployable selector。
- Protected neural/world-state package：M3W-Neural v1 / Stage41-42 evidence package 是当前 strongest research candidate。
- Full-waypoint/runtime policy：Stage42-DL 把 frozen group-consistency full-waypoint policy 做成可调用 runtime API，并 exact replay 原始 repair。

仍然不能说：

- “神经网络无保护超过 Stage37”。
- “JEPA 是主贡献”。
- “Transformer dynamics 单独成为主模型”。
- “goal/scene 或 neighbor/interaction 是当前独立主贡献”。
- “t100 已经 global deployable”。
- “metric / seconds-level 已闭环”。

## 6. 为什么有些看起来很强的东西仍不能部署

### 6.1 Oracle headroom 大，不等于模型成功

Stage24 已证明：即使 selector oracle headroom 很大，hard-class selector 也可能因为低-margin label 和 easy 过切换而失败。部署必须看：

- selected model vs strongest floor；
- easy degradation；
- harm over fallback；
- confidence / predicted gain；
- validation-only threshold；
- test 只评一次。

### 6.2 JEPA non-collapse，不等于 world model 成功

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

### 6.3 Full-waypoint 正提升，不等于 Stage5C

Stage42 full-waypoint 是 protected supervised full-waypoint / policy family，不是 latent generative rollout。future waypoints 是 loss/eval labels，不是 inference input。Stage5C 没执行，SMC 没启用。

### 6.4 t100 raw diagnostic，不等于 seconds-level long horizon

t100 仍是 raw-frame horizon。没有全局 FPS/stride/homography/metric scale 闭环，就不能写 seconds-level 或 metric long horizon。

## 7. 现在可以写进论文/报告的主 claim

可以写，但要带边界：

1. 在严格 no-leakage、past-only input、validation-only policy selection 下，M3W 从 SDD-only scaffold 推进到 external dataset-local raw-frame positive transfer。
2. Cost-aware / regret-aware / fallback-safe baseline policy 比 hard-class selector 更稳，能避免 easy over-switch。
3. External t50 修复依赖 past-only history window、scene-agnostic goal prototypes、gain/harm/easy safe switching。
4. Protected neural/full-waypoint policy family 在 Stage37/teacher floor 下提供正向 world-state evidence。
5. Baseline-family rollout context 是当前最稳定的 dominant mechanism。
6. Proximity-aware guard 在牺牲部分 accuracy 的同时修复 near-collision caveat，是安全部署的重要机制。
7. Group-consistency full-waypoint policy 已经从 repair、freeze、replay 推进到 runtime API exact replay。

## 8. 现在不能写的主 claim

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

## 9. 当前最重要的未解决问题

| 问题 | 当前状态 | 最短路径 |
| --- | --- | --- |
| true metric / seconds-level claim | blocked | source-specific homography/FPS/stride/scale/legal conversion 全闭环。 |
| t100 global deployable | blocked / diagnostic | 增加 independent t100-capable sources，做 source-CV 和 easy guard。 |
| JEPA/Transformer 独立主贡献 | not proven | 需要更强 graph/scene-rich neural protocol，且必须超过 baseline-family control。 |
| goal/scene contribution | mixed / diagnostic | 需要更可靠 scene packs / train-only goals / source-level held-out evidence。 |
| neighbor/interaction contribution | mixed / diagnostic | 当前 kNN/graph hand-built 特征不够，需要 real graph-neural or interaction-token protocol。 |
| foundation-track claim | not ready | 至少需要多数据集、多域、多模态预训练、跨场景泛化、metric/time calibration。 |
| Stage5C / SMC | forbidden | 只能未来生成 plan，不能执行，除非 gates 明确通过且用户确认。 |

## 10. 推荐下一步

1. **冻结当前 paper candidate evidence**：以 Stage42-CZ manifest 和 Stage42-DL runtime policy 为当前候选包，避免继续用 test 调参。
2. **补 metric/time/source closure**：优先关闭 ETH_UCY / TrajNet / UCY 的 legal/source/time/t100 blocker。
3. **做真正 context-rich neural protocol**：不要重复当前 goal/scene/kNN ridge/gated expert；下一步应做 graph/scene-token sequence model，并以 baseline-family control 为强对照。
4. **继续保留 Stage37/teacher safety floor**：除非无保护 neural 通过 easy、hard、t50、all、proximity gates，否则不要部署。
5. **t100 继续写 diagnostic**：直到 source-CV、easy safety、metric/time audit 全部闭环。

## 11. 最新核验命令

已运行：

```bash
.venv-pytorch/bin/python -m pytest tests
```

结果：

```text
655 passed in 32.28s
```

最新 provenance / manifest 刷新：

```text
Stage42-CX evidence provenance: 20 / 20 gates
Stage42-CZ paper freeze candidate manifest: 15 / 15 gates
```

## 12. 最终当前判定

```text
项目是否跑通：是
当前是否 true 3D：否
当前是否 foundation：否
当前是否 metric/seconds-level：否
当前是否 Stage5C：否
当前是否 SMC：否
当前 best deployable：protected M3W-Neural / full-waypoint policy family under Stage37/teacher floor
最新 runtime-ready policy：Stage42-DL group-consistency full-waypoint runtime policy
当前论文候选状态：protected dataset-local/raw-frame 2.5D evidence package candidate
最主要短板：metric/time/source closure、t100 global robustness、independent neural context contribution
```

<!-- STAGE42_DM_REVIEWER_REPLAY_PACKAGE:START -->
## Stage42-DM Reviewer Replay Package

- source: `fresh_reviewer_replay_package_from_stage42_runtime_and_manifest_artifacts`
- role: reviewer-facing minimal replay package for provenance, manifest, and runtime policy exact replay.
- gate: `21 / 21`; verdict `stage42_dm_reviewer_replay_package_pass`.
- commands file: `outputs/stage42_long_research/reviewer_replay_commands_stage42.sh`.
- group-consistency runtime all/t50/t100 raw/hard: `0.24715658317833844` / `0.2236298792899738` / `0.1434611214781808` / `0.23887420070464105`.
- This is replay/provenance packaging only: no training, no threshold tuning, no Stage5C, no SMC, no metric/seconds-level claim.
<!-- STAGE42_DM_REVIEWER_REPLAY_PACKAGE:END -->

<!-- STAGE42_DN_DEPLOYMENT_VARIANT_CARD:START -->
## Stage42-DN Deployment Variant Card

- source: `fresh_deployment_variant_card_from_stage42_cr_cq_di_dl_dm`
- role: separates safety-sensitive deployment, accuracy-priority diagnostics, and protocol-specific group-consistency runtime policy.
- gate: `20 / 20`; verdict `stage42_dn_deployment_variant_card_pass`.
- safety-sensitive default: `proximity_guard` for endpoint-linear bridge/shape deployment with joint-proximity safety.
- strongest full-waypoint runtime evidence: `group_consistency_full_waypoint_runtime`, but it uses train-horizon causal-floor comparison and must not be rank-mixed with endpoint-linear composer variants without that caveat.
- accuracy-priority diagnostic: `no_proximity_guard`; it has higher ADE gains but worsens near-collision@0.05 and is not the safety-sensitive deployment claim.
- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
<!-- STAGE42_DN_DEPLOYMENT_VARIANT_CARD:END -->
