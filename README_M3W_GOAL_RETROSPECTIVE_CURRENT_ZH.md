# M3W 长期目标当前总复盘：路线、失败、成功与边界

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已有 Stage18-Stage42 报告、gate、README、`research_state.json`；最近 source acquisition / blocker matrix 为 `fresh_stage42_bv_source_acquisition_status`。  

这份 README 是给当前长期目标的单文件总结：我在这个目标内做了什么、尝试过哪些路线、哪些失败了、为什么失败、哪些成功了、当前最强可部署模型是谁，以及哪些话现在仍然不能说。

## 0. 必须先写清楚的事实

当前 M3W 仍然不是：

```text
true 3D world model
large-scale foundation world model
global metric predictor
seconds-level long-horizon predictor
ungated neural world dynamics
latent generative world model
SMC-ready model
```

当前最准确的说法是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

严格边界：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external top-down pedestrian 数据仍是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 当前强结果都依赖 safety floor / conservative fallback；无保护 neural dynamics 不能部署。

## 1. 一句话结论

这个长期目标里，我实际推进的是一条从 **SDD pixel-space selector scaffold** 到 **external dataset-local protected multi-agent world-state candidate** 的路线。

最重要的成功是：Stage26 在 SDD 上形成 cost-aware deployable selector；Stage37 修复 external t+50；Stage42 把结果推进到 source-level / full-waypoint / safety-floor / blocker-matrix 证据包。

最重要的失败是：JEPA-only、ungated Transformer / Hybrid、普通 residual correction、hard-class selector、纯 normalization / latent alignment 都不能安全部署。

当前 best deployable：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

当前 honest claim：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

不是：

```text
foundation world model
true 3D world model
metric / seconds-level model
Stage5C generative model
SMC model
ungated neural dynamics model
```

## 2. 总路线图

| 阶段/路线 | 做了什么 | 结果 | 主要原因 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建 per-agent multi-agent 2.5D world-state、causal baseline fallback、diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 true 3D / foundation。 |
| Stage18/19 JEPA / WAM-style data | SAM-JEPA-2.5D、WAM-style simulation/top-down/ego-video 数据策略。 | JEPA non-collapse，但 downstream lift 未证明。 | non-collapse 不等于 selector/failure/correction/t50 改善。 |
| Stage20/21 web data / SDD conversion | 合法数据登记、OpenTraj/SDD 路径、SDD world-state shards。 | 成功建立 SDD official pixel-space 数据基座。 | SDD 数据合法路径和 no-leakage、causal velocity 被显式审计。 |
| Stage22/23 SDD benchmark | scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、baselines。 | SDD benchmark 成功；quick-plus/medium 区分清楚。 | 仍是 pixel raw-frame，不是 metric / seconds。 |
| Stage24 hard selector | medium SDD 上训练 validation-selected selector。 | 失败。 | oracle headroom 大，但 hard classification 过度切换；t+50 -43.3%，easy degradation 11.33%。 |
| Stage25/26 cost-aware selector | expected-FDE / regret-aware / confidence-gated fallback selector。 | 成功。 | 修复低-margin label 与 easy 过切换；SDD t+50 约 +14.58%，hard/failure 约 +11.23%，easy 约 1.81%。 |
| Stage31/32 external zero-shot / alignment | SDD -> external transfer，normalization、CORAL、latent adapter、mixed-domain selector。 | 失败。 | 坐标、horizon、scene/goal、agent type、scale/homography 不一致；分布对齐不等于决策目标对齐。 |
| Stage33/34 external geometry | 坐标不变 features、relative targets、external row geometry、train-only goals。 | 局部正信号，不可部署。 | t50/hard 有提升，但 all/easy 不稳。 |
| Stage35 selective transfer | external hard/easy/failure labels、gain/harm/easy gate。 | 部分成功。 | all +12.13%、hard +13.98%、easy 0.041%，但 t+50 = 0。 |
| Stage36 t50 repair | horizon-specific selector / t50 policy search / curriculum。 | 失败。 | t50 有 oracle headroom，但缺足够 past-only history / goal prototype / switchability signal。 |
| Stage37 causal history + goal prototype | K=8/16/32/64 history windows、scene-agnostic goal prototypes、t50 gain/harm/failure、安全 conformal policy。 | 成功。 | external all +13.48%，t50 +8.46%，CI [+7.69,+9.15]，hard +15.54%，easy 0.041%，16/16 gates。 |
| Stage38 bounded correction | Stage37 保护下训练 bounded delta / correction head。 | 不部署。 | correction 没稳定超过 Stage37，普通 residual 容易伤 easy。 |
| Stage39/40 Transformer / JEPA / Hybrid neural | Causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 诊断为主，不替代 Stage37。 | 无保护 neural 不安全；受保护 neural 没有稳定脱离 Stage37 floor。 |
| Stage41/42 long research | composite-tail、full-waypoint、row cache、source-level split、ablation、source-CV、t100/source calibration、blocker matrix。 | 成功形成 protected evidence package，但仍有 blocker。 | 强机制主要是 baseline-family rollout context + validation-safe guard；t100/global metric/seconds 仍 blocked。 |

## 3. 成功结果

### 3.1 Stage26：SDD 上的第一个可靠 deployable selector

Stage24 证明“oracle headroom 大”不等于 selector 能成功。hard-class selector 会在 easy 样本上乱切换，造成大幅伤害。

Stage25/26 的修复：

- 不再只预测 `best baseline class`。
- 改成 expected-FDE / cost-aware / regret-aware selection。
- 加 confidence gate、predicted gain threshold、easy guard、harm guard。
- 使用 failure predictor 作为辅助。
- 低 margin / 低 confidence 样本 fallback。

结果：

| 指标 | 结果 |
| --- | ---: |
| benchmark | SDD official pixel-space raw-frame |
| t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |
| verdict | SDD best deployable selector |

边界：只支持 SDD pixel raw-frame claim，不支持 metric / true 3D / foundation claim。

### 3.2 Stage37：external t+50 deployable 修复

Stage35/36 前的状态：

```text
all improvement = +12.13%
hard/failure = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
```

这说明外部迁移有正信号，但长时程 t+50 不敢切换，不能部署。

Stage37 的关键改动：

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
- 训练 t50 failure / gain / harm predictor。
- 训练 t50 专用安全 selector。
- 使用 conformal safety 控制 easy degradation。
- 不用 future endpoint 做 input，不用 test endpoint 建 goals。

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

意义：Stage37 是 external dataset-local raw-frame 下第一个可部署正迁移点。

### 3.3 M3W-Neural v1 protected package

在 Stage37 safety floor 下，Stage41/42 形成了更完整的 protected neural / source-level evidence package。

| 指标 | 结果 |
| --- | ---: |
| gates | 41 / 41 |
| rows | 55,528 |
| all ADE improvement | +21.03% |
| t+50 ADE improvement | +13.65% |
| t+100 raw-frame diagnostic ADE improvement | +14.69% |
| hard/failure ADE improvement | +20.38% |
| easy degradation | 0.00% |
| positive external domains | 3 |
| all-agent composite FDE improvement | +19.82% |
| all-agent composite FDE@50 improvement | +17.39% |

Bootstrap lower bounds：

| slice | low | mid | high |
| --- | ---: | ---: | ---: |
| all | 20.67% | 21.02% | 21.39% |
| t+50 | 13.06% | 13.66% | 14.26% |
| t+100 raw-frame diagnostic | 13.96% | 14.69% | 15.37% |
| hard/failure | 19.99% | 20.39% | 20.76% |

边界：这是 protected deployment，不是 ungated neural。

### 3.4 Stage42 source-level / full-waypoint / blocker work

Stage42 做了很多“防止过度包装”的工作：从 report-level 平均数推进到 row-cache、source-level split、ablation、source-CV、source-family support。

| 子阶段 | 关键结果 | 结论 |
| --- | --- | --- |
| Stage42-C full-waypoint dynamics | protected full-waypoint ADE all +18.58%，t50 +14.80%，t100 raw diagnostic +22.86%，hard +19.52%，easy 0。 | full-waypoint world-state evidence 成立，但仍 protected。 |
| Stage42-H sequence ablation | 去掉 history tokens 后 t50/hard 明显下降。 | history 在 sequence encoder 下有用；flattened history + ridge 不足。 |
| Stage42-J static-gated full-waypoint | ADE all +3.62%，t50 +3.69%，hard +3.97%，FDE t50 +11.66%，easy 0。 | static context 不能全局硬混，要 validation-gated。 |
| Stage42-P t50 gain/harm selector | ADE all +5.15%，t50 +0.66%，hard +5.33%，easy 0.86%，FDE t50 +5.74%。 | t50 sign repaired，但 t50 CI low 仍负，不足以 paper-level 稳定 claim。 |
| Stage42-R row prediction cache combo | cached combo ADE all +5.24%，t50 +3.79%，t50 CI low +2.77%，hard +5.48%，easy 0.11%。 | row-level combo 修复 Stage42-Q report-level preflight 缺口。 |
| Stage42-S frozen row combo policy | policy/cache/schema hash frozen；positive domains ETH_UCY、TrajNet；UCY fallback-only。 | 可审计部署 artifact，但 UCY 仍不是这个 combo 的正迁移证据。 |
| Stage42-AU baseline-family mechanism | family-baseline context protected all/t50 很强。 | 当前 source-level 主机制是 baseline-family rollout context。 |
| Stage42-BV source acquisition matrix | gates 16/16；5 个 blocker active/actionable。 | 不把缺数据、terms、not_run 包装成完成。 |

## 4. 失败路线和原因

### 4.1 JEPA non-collapse 但 downstream 无效

失败表现：

- Stage18 / Stage19 / 后续 JEPA 分支可以 non-collapse。
- 但 selector、failure predictor、correction、official t+50 没有稳定 lift。

原因：

- JEPA 的 latent variance 正常不等于任务相关。
- 预训练目标和部署目标错位。
- t+50 / hard/failure 的真正关键是 gain/harm/easy 安全切换，而不是单纯 latent 表征。

结论：

```text
JEPA 只能写 representation diagnostic / auxiliary。
不能写成 latent generative world model，也不能写成当前主贡献。
```

### 4.2 hard-class selector 失败

失败表现：

- Stage24 selector oracle headroom 约 46.2%。
- 但 validation-selected hard selector t+50 improvement = -43.3%。
- easy degradation = 11.33%。

原因：

- oracle label 低 margin、噪声大。
- hard classification 强迫模型在本应 fallback 的 easy samples 上切换。
- 预测“哪个 baseline 最好”不等于最小化部署 regret。

修复：

- Stage25/26 改成 expected-FDE / cost-aware selector。
- 加 conservative fallback、gain/harm/easy guard。

### 4.3 SDD -> external zero-shot 失败

失败表现：

- Stage31 external zero-shot transfer 负迁移严重。
- Stage32 普通 normalization / adapted selector 仍不够。

原因：

- SDD 是 pixel-space，external 是 dataset-local / weak-metric diagnostic。
- 不同数据的 coordinate scale、horizon、track length、scene/goal、agent type 不一致。
- latent distribution alignment 缩小距离，不等于决策边界可迁移。

修复路径：

- Stage33/34 补 row geometry、relative targets、train-only goals。
- Stage35 selective transfer 修复 all/hard/easy。
- Stage37 history + goal prototype 修复 t+50。

### 4.4 普通 residual / correction 失败

失败表现：

- Stage38 bounded correction 没有稳定超过 Stage37。
- ordinary residual 容易伤 easy。

原因：

- residual 会在 baseline already-good 的样本上过度干预。
- hard/failure 有空间，但 easy preservation 是部署约束。
- 没有足够安全的 row-level harm gate 时，correction 不可靠。

结论：

```text
correction specialist 不能部署。
当前部署仍保持 Stage37 / teacher floor。
```

### 4.5 Transformer / Hybrid 无保护失败

失败表现：

- Stage39/40 训练 Transformer / JEPA / Hybrid。
- neural without fallback 不安全。
- neural with Stage37 fallback 没有稳定独立超过 floor。

原因：

- 模型容易学会复制 teacher/floor，而不是安全产生新 dynamics。
- t+50/hard 的可用信号需要 gain/harm/easy guard。
- full-waypoint / source-level 泛化下，静态 context、neighbor、goal 一旦无门控就会伤 easy。

结论：

```text
neural dynamics 目前只能作为 protected auxiliary / candidate。
不能替代 Stage37 safety floor。
```

### 4.6 t+100 / metric / seconds 仍失败或 blocked

失败表现：

- t100 有局部 diagnostic 正信号，但 source-CV / easy / data support 不足。
- global metric / seconds claim 不允许。

原因：

- 独立 t100-capable sources 不足。
- ETH_UCY / TrajNet / UCY source families 对 t100 支持不均。
- FPS、annotation stride、homography、meter-per-pixel 没有全局统一验证。

Stage42-BV 当前 blocker：

| blocker | 状态 | 下一步 |
| --- | --- | --- |
| `ETH_seq_t50_source_support` | blocked | 验证 ETH-Person terms 或提供 legal source-compatible ETH_seq long-track source。 |
| `UCY_students_t50_source_support` | blocked_narrowed | 还需要 1 个 independent t50-capable students-family source。 |
| `TrajNet_raw_long_t100_source_support` | blocked | 需要 legal raw long-track TrajNet-compatible sources。 |
| `ETH_UCY_global_t100_source_support` | blocked | terms/source support 不足，需重新 conversion + strict source-CV。 |
| `global_metric_seconds_claim` | blocked | 只能 source-specific calibrated subset，不能 global claim。 |

## 5. 当前最强可部署模型是谁

当前 best deployable：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

为什么不是裸 neural：

- 无保护 neural 会伤 easy。
- Stage38 correction 没有超过 Stage37。
- Stage39/40 Transformer / JEPA / Hybrid 没有稳定 floor-free lift。
- Stage42 source-level 证据显示主要有效机制是 baseline-family rollout context + validation-safe guard。

部署策略：

```text
if source/horizon/family not supported:
    fallback
elif confidence low:
    fallback
elif predicted gain small:
    fallback
elif easy/harm risk high:
    fallback
else:
    use protected selected baseline / bounded dynamics head
```

## 6. 现在可以写成论文 claim 的内容

可以写：

```text
We develop a protected dataset-local raw-frame 2.5D multi-agent world-state candidate.
The strongest deployable evidence comes from regret-aware baseline-family rollout selection under strict no-leakage and validation-only safety policies.
Stage37 repairs external t+50 transfer, and Stage42 source-level/full-waypoint evidence supports protected raw-frame gains.
```

可以写成弱一点的贡献：

- strict no-leakage external top-down transfer protocol。
- cost-aware / regret-aware / conservative fallback baseline policy。
- causal history + scene-agnostic goal prototype 对 external t+50 的作用。
- source-level / full-waypoint / validation-only policy evaluation package。
- safety floor necessity and blocker accounting。

不能写：

```text
We solved true 3D world modeling.
We built a foundation world model.
We achieved global metric prediction.
We achieved seconds-level t+50/t+100.
JEPA is a generative world model.
Ungated Transformer/Hybrid is deployable.
Stage5C or SMC is ready/executed.
History/goal/interaction are independently proven main contributions across all source-level ablations.
```

## 7. 仍然缺什么

### 7.1 数据缺口

- 还需要更多 legal independent external top-down sources。
- `UCY_students` 还缺 1 个 independent t50-capable students-family source。
- `ETH_seq` 需要 legal same-family/source-compatible support。
- TrajNet 需要 raw long-track source 才能修 t100 source-CV。

### 7.2 时间/几何缺口

- SDD 仍是 pixel-space。
- external 仍是 dataset-local / unverified weak-metric。
- global FPS / annotation stride / homography / scale 未完成。
- 不能把 raw-frame horizon 说成 seconds-level。

### 7.3 模型缺口

- JEPA downstream lift 未证明。
- Transformer / Hybrid floor-free lift 未证明。
- correction specialist 不可部署。
- 当前主机制偏 baseline-family rollout context，神经 world dynamics 独立贡献还不够强。

### 7.4 论文级缺口

- 需要更完整的 source-level retrained ablation。
- 需要更多 external domains。
- 需要 stronger scene/goal/interaction ablation。
- 需要 t100/source-CV 支持或诚实去掉 t100 主 claim。

## 8. 最短下一步

1. **先修 source blockers，不要继续包装 t100。**  
   补 `UCY_students` 第三个 independent t50 source、ETH_seq legal support、TrajNet raw long tracks；然后重新跑 conversion、no-leakage、source-CV。

2. **把 baseline-family rollout context 写成当前主机制。**  
   当前最强证据来自 causal baseline family + gain/harm/easy guard，而不是 JEPA/Transformer 独立学会 world dynamics。

3. **若继续追求真正 neural world model，要做 source-level graph/scene-rich protocol。**  
   需要 richer scene tokens、graph interaction、full-waypoint loss、multi-domain source split、multi-seed/bootstrap，并证明它在 protected policy 下超过 baseline-family-only。

## 9. 文件索引

关键总结/报告：

```text
README_RESULTS.md
README_M3W_LONG_GOAL_RETROSPECTIVE_ZH.md
README_M3W_RESEARCH_SUMMARY_ZH.md
README_M3W_DETAILED_RESULTS_ZH.md
outputs/stage37_t50_history/report_stage37_final.md
outputs/stage38_external_robustness/report_stage38_final.md
outputs/stage40_neural_optimization/report_stage40_final.md
outputs/stage42_long_research/report_stage42_final.md
outputs/stage42_long_research/source_acquisition_status_stage42.md
outputs/stage42_long_research/user_action_required_source_acquisition_stage42.md
```

本文件是当前目标级入口：

```text
README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md
```

## 10. 当前最终判定

```text
项目是否跑通：是，作为 protected dataset-local/raw-frame 2.5D world-state candidate。
是否 true 3D：否。
是否 foundation：否。
是否 metric/seconds-level：否。
是否 Stage5C ready/executed：否 / 未执行。
是否 SMC ready/enabled：否 / 未启用。
当前 best deployable：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。
当前最大 blocker：external source support、global metric/time calibration、t100 source-CV、floor-free neural dynamics。
```
