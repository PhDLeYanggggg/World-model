# M3W 长期目标结果总账：路线、失败、成功与当前质量

更新时间：2026-05-27  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已生成报告、gate、README、`research_state.json`；最新纳入到 Stage42-DM reviewer replay package。  
最近一次完整测试：Stage42-DM 后 `.venv-pytorch/bin/python -m pytest tests` 通过，`658 passed in 32.06s`。  

这份 README 回答用户当前问题：在 M3W 这个长期目标内，我到底做了什么、尝试过哪些路线、哪些失败了、为什么失败、哪些成功了、现在模型大概是什么质量，以及哪些结论仍然不能写。

## 0. 当前一句话结论

M3W 已经从一个 SDD-only 的 2.5D 多智能体轨迹 scaffold，推进成一个有 SDD 与 external dataset-local raw-frame 正迁移证据的 protected multi-agent world-state candidate。

但它仍然不是：

- true 3D world model
- large-scale foundation world model
- metric / seconds-level physical predictor
- ungated neural dynamics deployable model
- latent generative Stage5C model
- SMC-ready model

当前最诚实的定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

当前 best deployable / best evidence 分层：

| 用途 | 当前最强结果 | 结论 |
| --- | --- | --- |
| SDD pixel-space official benchmark | Stage26 cost-aware selector | SDD t+50 与 hard/failure 已稳定超过 strong causal floor，但仍是 pixel raw-frame。 |
| External t+50 deployable selector | Stage37 history + goal-prototype safe transfer policy | 第一次让 external all / t50 / hard 同时为正且 easy 安全。 |
| Protected neural/world-state candidate | M3W-Neural v1 / Stage41-42 protected composer family | 有 protected neural / full-waypoint / runtime replay 证据，但仍依赖 safety floor。 |
| Safety-sensitive composer | Stage42-CQ proximity-aware composer guard | 牺牲部分 ADE 增益，修复 near-collision caveat。 |
| Full-waypoint runtime policy | Stage42-DL group-consistency full-waypoint runtime API | frozen policy 可调用并 exact replay，不是 Stage5C/SMC。 |
| Reviewer evidence package | Stage42-DM reviewer replay package | 提供最小复现命令、artifact hash、exact replay 与 gate。 |
| Deployment variant card | Stage42-DN deployment variant card | 明确区分 safety-sensitive deployable、accuracy-priority diagnostic、source-level runtime policy，避免 claim 混用。 |

## 1. 永久边界

所有报告和 README 都必须继续写清楚：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一米制世界坐标。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
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

## 2. 我实际走过的路线

| 阶段 / 路线 | 做了什么 | 结果 | 解释 |
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

## 3. 关键成功证据

### 3.1 Stage26：SDD cost-aware selector 成功

Stage26 修复了 Stage24 hard-class selector 的失败，把任务从“预测 best baseline class”改成“预测 expected FDE / risk，再做 conservative fallback”。

关键结果：

```text
t+50 improvement: about +14.58%
hard/failure improvement: about +11.23%
easy degradation: about 1.81%
```

意义：

- 这是 SDD pixel raw-frame 上的当前 best deployable selector。
- 它证明 cost-aware / regret-aware / fallback-safe policy 比 hard classification 更可靠。
- 它仍然不是 metric / 3D / foundation。

### 3.2 Stage37：external t+50 正迁移修复成功

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

### 3.3 Stage41 / M3W-Neural v1：protected neural candidate 成立

Stage39/40 的无保护 neural 没超过 Stage37 后，Stage41 转向 protected neural：bounded neural dynamics + teacher safety floor + composite-tail safe switch。

关键结果：

```text
M3W-Neural v1 verdict: composite-tail safe-switch bounded neural dynamics candidate
gates: 41 / 41
all improvement vs Stage37 floor: +21.03%
t+50 improvement vs Stage37 floor: +13.65%
t+100 raw-frame diagnostic improvement: +14.69%
hard/failure improvement: +20.38%
easy degradation: 0.00%
positive external domains: 3
bootstrap evidence pass: true
multiseed replication pass: true
pure UCY source-heldout gate: true
```

意义：

- 这是目前最强的 protected neural/world-state candidate。
- 它不是“无保护 neural 单独成功”，而是神经模型在 Stage37/teacher safety floor 下提供增益。
- JEPA deployable path 被关闭：`disable_jepa_in_deployable_path`。

### 3.4 Stage42-CQ / CR：proximity guard 修复 safety caveat

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

### 3.5 Stage42-DI / DJ / DK / DL / DM：group-consistency full-waypoint runtime 证据包

Stage42-DI 修复 group-consistency full-waypoint；DJ 冻结 policy；DK 做 exact replay；DL 做 runtime API；DM 做 reviewer replay package。

Stage42-DL / DM 关键结果：

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
Stage42-DM gate: 21 / 21
package hash: 422f6b3d47c33b6f87edda7b825e15825840712fc792852f15718ac51b14df90
```

意义：

- frozen policy 可以通过 runtime API 精确复现。
- reviewer replay commands 已生成。
- 这是 protected full-waypoint runtime policy 证据，不是 Stage5C，不是 SMC。

### 3.6 Stage42-DN：部署变体卡片，防止 claim 混用

Stage42-DN 没有重新训练，也没有调 threshold；它把已有 Stage42-CR/CQ/DI/DL/DM 证据整理成部署变体卡片，明确不同 policy 的可用语境。

关键结果：

```text
gate: 20 / 20
verdict: stage42_dn_deployment_variant_card_pass

endpoint_linear_reference:
  role = reference_floor
  comparison_baseline = endpoint_linear_bridge

no_proximity_guard:
  role = accuracy_priority_diagnostic
  all +3.02%, t50 +1.50%, t100 raw +6.12%, hard +3.28%
  caveat = near-collision@0.05 worsens vs endpoint-linear

proximity_guard:
  role = safety_sensitive_deployable_bridge_shape_policy
  all +1.77%, t50 +1.07%, t100 raw +3.48%, hard +1.93%
  safety = near-collision@0.05 not worse than endpoint-linear

group_consistency_full_waypoint_runtime:
  role = source_level_full_waypoint_group_consistency_runtime_policy
  comparison_baseline = train_horizon_causal_floor_not_endpoint_linear_bridge
  all +24.72%, t50 +22.36%, t100 raw +14.35%, hard +23.89%
```

意义：

- `no_proximity_guard` 只能作为 accuracy-priority diagnostic，不能写成 safety-sensitive deployment。
- `proximity_guard` 是 endpoint-linear bridge/shape 场景下的 safety-sensitive deployment variant。
- `group_consistency_full_waypoint_runtime` 是更强的 source-level runtime evidence，但 baseline/protocol 不同，不能和 endpoint-linear composer 直接混排排名。
- 这一步强化的是 claim hygiene / deployment hygiene，不是新训练结果。

## 4. 关键失败与原因

| 失败路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| Stage18/19 JEPA | non-collapse，但 downstream heads 无 lift。 | representation target 与 deployment target 错位；latent variance 不等于决策有用。 | 保留为 auxiliary / diagnostic，不作为主贡献。 |
| Stage24 hard-class selector | t+50 improvement -43.3%，easy degradation 11.33%。 | 低 margin oracle label 噪声大；hard class forcing 让 selector 过度切换；confidence calibration 差。 | 改成 expected-FDE / regret / gain-harm / conservative fallback。 |
| Stage31/32 external zero-shot | SDD -> external all/t50 严重负迁移。 | SDD pixel 和 external dataset-local 坐标/尺度/horizon/agent type/scene-goal 不一致。 | 做 row geometry、relative target、history window、goal prototypes。 |
| Normalization / CORAL / latent adapter | 分布距离缩小，但预测没有 lift。 | 分布对齐不是 decision target 对齐；latent gap 小不代表能选对 baseline。 | 不把 latent adapter 作为成功 claim。 |
| Stage35/36 t50 | all/hard/easy 正，但 t50=0。 | t50 有约 22.98% oracle headroom，但缺可靠 past-only long-horizon switchability signal。 | Stage37 用 history window + goal prototypes 修复。 |
| Bounded correction / ordinary residual | 不稳定超过 Stage37，容易伤 easy。 | selected baseline 已强；残差学习容易在 easy 样本过修正。 | correction 不部署，除非 fallback 后超过 Stage37。 |
| Stage39/40 ungated neural | neural without fallback all -126.36%，t50 -292.10%，easy degradation 612.31%。 | 无保护 neural 在 dataset-local noisy horizon 上会灾难性伤害 easy / baseline floor。 | 只允许 Stage37/teacher protected neural。 |
| goal/scene gated expert | 低于 baseline-family control。 | 当前 goal/scene proxy 在 source-level ridge/full-waypoint协议下没有独立增量。 | 写成 mixed / diagnostic。 |
| neighbor/interaction gated expert | kNN/graph 特征丰富但仍低于 baseline-family control。 | hand-built interaction 特征不足以独立超过 rollout-family context。 | 写成 auxiliary / diagnostic。 |
| t100 global deployable | 多次局部正，但 source-CV/easy/source support 不闭环。 | independent t100 source support 不足；ETH_UCY/TrajNet/UCY legal/source/time blocker 未完全关闭。 | t100 只写 raw-frame diagnostic。 |
| metric / seconds claim | 仍不能全局声称。 | FPS/stride/homography/scale/source legality 未全局闭环。 | 禁止 metric/seconds-level claim。 |

## 5. 为什么现在还不是 true world model

当前 M3W 有 world-state / protected policy / trajectory dynamics evidence，但距离“真正强的真实世界多模态多智能体世界模型”还有差距：

1. **坐标不是统一物理世界坐标**：SDD 是 pixel，external 是 dataset-local / weak metric diagnostic。
2. **时间不是统一真实秒级**：t50/t100 是 raw-frame horizon。
3. **多模态还不完整**：scene/goal/interaction 有辅助证据，但不是独立主贡献。
4. **JEPA/Transformer 不可无保护部署**：神经模型需要 Stage37/teacher safety floor。
5. **t100 不是 global deployable**：只有 raw-frame diagnostic / 局部证据。
6. **Stage5C/SMC 未执行**：没有 latent generative rollout，没有 stochastic proposal coverage evidence。

## 6. 当前模型质量判断

我会把当前质量分成四层：

### 6.1 工程/实验闭环质量：较强

已经具备：

- 明确数据角色和 no-leakage 审计。
- SDD medium / external feature store / history window / policy replay。
- bootstrap / multiseed / exact replay / provenance / manifest。
- README、model card、data card、failure analysis、paper package。
- runtime API replay 证据。

这部分已经不是 demo，而是一个可审计研究轨道。

### 6.2 SDD / external raw-frame selector 质量：强

Stage26 和 Stage37 都有明确过 gate 的 deployable selector 证据。

### 6.3 Protected neural world-state 质量：中强，但有边界

M3W-Neural v1 / Stage41-42 已经有 protected neural/full-waypoint evidence，但部署依赖 Stage37/teacher safety floor。它可以写成 protected neural world-state candidate，不能写成 ungated neural world model。

### 6.4 Foundation / true-3D / metric world model 质量：未达到

还没有跨大规模多数据集、多模态原始视频、metric/seconds-level 统一校准、true 3D geometry、latent generative rollout、SMC coverage evidence。因此不能称 foundation 或 true 3D。

## 7. 当前可以写的 claim

可以写：

1. 在严格 no-leakage、past-only input、validation-only policy selection 下，M3W 从 SDD-only scaffold 推进到 external dataset-local raw-frame positive transfer。
2. Cost-aware / regret-aware / fallback-safe baseline policy 明显优于 hard-class selector。
3. External t50 修复依赖 past-only history window、scene-agnostic goal prototypes、gain/harm/easy safe switching。
4. Protected neural/full-waypoint policy family 在 Stage37/teacher floor 下提供正向 world-state evidence。
5. Baseline-family rollout context + causal history + guarded domain expert 是当前 dominant mechanism。
6. Proximity guard 和 group-consistency repair 提供了安全部署和 runtime replay evidence。

## 8. 现在不能写的 claim

不能写：

1. M3W 是 true 3D world model。
2. M3W 是 foundation world model。
3. SDD/external 结果是统一 metric result。
4. raw-frame t50/t100 是 seconds-level horizon。
5. JEPA 是生成式 world model。
6. Transformer/Hybrid 无保护超过 Stage37。
7. goal/scene 或 neighbor/interaction 是当前独立主贡献。
8. t100 是 global deployable success。
9. Stage5C 已执行或 ready。
10. SMC 已启用或 ready。

## 9. 当前最重要的 blocker

| Blocker | 当前状态 | 最短路径 |
| --- | --- | --- |
| metric / seconds-level claim | blocked | source-specific FPS/stride/homography/scale/legal conversion 全闭环。 |
| t100 global deployable | diagnostic only | 增加 independent t100-capable sources，做 source-CV 与 easy/proximity guard。 |
| JEPA 独立贡献 | not proven | 需要重设 JEPA target / downstream alignment，并必须超过 non-JEPA control。 |
| Transformer 无保护部署 | unsafe | 需要新的 safety-calibrated neural protocol，不能移除 Stage37 floor。 |
| goal/scene 独立贡献 | mixed/diagnostic | 需要更可靠 scene packs、train-only goals、source-level held-out evidence。 |
| neighbor/interaction 独立贡献 | mixed/diagnostic | 需要更强 graph/scene-token sequence model，而不是 hand-built kNN ridge。 |
| foundation-track | not ready | 需要多数据集、多模态预训练、跨域泛化、metric/time calibration。 |
| Stage5C / SMC | forbidden | 只能未来生成 plan，不能执行，除非 gates 和用户确认。 |

## 10. 我下一步建议

1. **先冻结当前证据包**：以 Stage42-DM reviewer replay package、Stage42-DL runtime policy、Stage42-CZ manifest 为当前可审计候选包。
2. **补 source/time/metric closure**：优先关闭 ETH_UCY / TrajNet / UCY 的 legal/source/time/t100 blocker。
3. **区分部署变体**：把 accuracy-priority diagnostic、safety-sensitive deployable、source-level full-waypoint runtime 三类 policy 分开写，避免 claim 混淆。
4. **只在 safety floor 下推进 neural**：不要再把 ungated neural 当部署路线，除非它真的通过 easy/hard/t50/all/proximity gates。
5. **下一轮真正研究重点**：source-level graph/scene-rich sequence model + stronger source-CV + t100-capable independent sources。

## 11. 文件入口

主要证据入口：

- `README_RESULTS.md`
- `README_M3W_CURRENT_GOAL_SUMMARY_ZH.md`
- `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- `outputs/m3w_neural_v1/report_m3w_neural_v1.md`
- `outputs/stage37_t50_history/report_stage37_final.md`
- `outputs/stage38_external_robustness/report_stage38_final.md`
- `outputs/stage39_neural_dynamics/report_stage39_final.md`
- `outputs/stage40_neural_optimization/report_stage40_final.md`
- `outputs/stage42_long_research/report_stage42_final.md`
- `outputs/stage42_long_research/reviewer_replay_package_stage42.md`

最新本文件：

```text
README_M3W_GOAL_RESULTS_SUMMARY_ZH.md
```

<!-- STAGE42_DO_SOURCE_LEGAL_TIME_ACTION_PACKAGE:START -->
## Stage42-DO Source Legal/Time Action Package

- source: `fresh_synthesis_from_stage42_cg_bn_dd_after_da1_rerun`
- role: closes the current DA-1 pass as an honest blocker/action package, not as conversion or evaluation.
- gate: `13 / 13`; verdict `stage42_do_source_legal_time_action_package_pass`.
- conversion-ready targets: `0`; converted/evaluated now: `0` / `0`.
- source-specific metric/time candidate count: `6`.
- global metric/seconds/t100 deployable claims remain blocked; Stage5C and SMC remain disabled.
- user action file: `outputs/stage42_long_research/user_action_required_source_legal_time_stage42.md`.
<!-- STAGE42_DO_SOURCE_LEGAL_TIME_ACTION_PACKAGE:END -->
