# M3W 长期目标复盘 README：路线、失败、成功证据与当前边界

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已有报告、gate、README、`research_state.json`，并纳入最近 Stage42-BR `fresh_calibrated_t50_source_support_gap_audit`。  

这份 README 回答你的问题：

1. 在“训练真正强的真实世界多模态多智能体世界模型 M3W”这个目标内，我做了什么。
2. 尝试过哪些路线。
3. 哪些失败了，失败原因是什么。
4. 哪些成功了，成功证据是什么。
5. 当前最强可部署模型是谁。
6. 目前距离真正 world model / A刊候选还差什么。

它不是宣传稿。凡是 `not_run`、technical dry-run、fallback-only、license-blocked、source-support 不足的结果，都不能写成完成或成功。

## 0. 当前必须诚实承认的事实

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

当前 M3W 更准确的表述是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

严格边界：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 仍不安全；当前强结果都依赖 safety floor / conservative fallback。

## 1. 当前最强可部署模型

当前 best deployable 不是裸 Transformer、裸 JEPA、裸 Hybrid，也不是 Stage5C，而是受保护策略：

```text
current best deployable:
  M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
  under Stage37 / teacher safety floor

dominant effective mechanism:
  causal baseline-family rollout context
  + validation-selected gain/harm/easy guard
  + conservative fallback
```

部署逻辑：

```text
如果 confidence 低：
  fallback

如果 predicted gain 小：
  fallback

如果 easy/harm risk 高：
  fallback

如果 source-family / horizon 支持不足：
  fallback

否则：
  只在 validation 证明安全的 slice 上切换或使用 protected correction / dynamics head
```

这条路线的核心不是让神经网络无条件替代 baseline，而是让神经网络或 selector 在安全 floor 保护下，只吃确定有收益的样本。

## 2. 一张总表：路线、结果、原因

| 路线 | 做了什么 | 结果 | 原因 / 解释 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建 per-agent multi-agent 2.5D world-state、causal baseline fallback、diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 true 3D / foundation。 |
| Stage18/19 JEPA / WAM-style data | SAM-JEPA-2.5D、WAM-style simulation / top-down / ego-video data strategy。 | JEPA non-collapse，但 downstream lift 未证明。 | non-collapse 不等于 selector/failure/correction/t50 改善；目标和部署任务不够对齐。 |
| Stage20/21 数据采集与 SDD 转换 | 合法数据登记、SDD/OpenTraj 路径、SDD world-state shards。 | 成功建立 SDD official pixel-space 数据基座。 | 数据合法性、no-leakage、causal velocity 都被显式审计。 |
| Stage22/23 SDD benchmark | scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、baselines。 | SDD benchmark 建立成功；quick-plus/medium 区分清楚。 | SDD 仍是 pixel raw-frame，不是 metric/seconds。 |
| Stage24 hard selector | medium SDD 上训练 validation-selected selector。 | 失败。 | oracle headroom 大，但 hard classification 过度切换，t+50 -43.3%，easy degradation 11.33%。 |
| Stage25/26 cost-aware selector | expected-FDE / regret-aware / confidence-gated fallback selector。 | 成功。 | 修复低 margin label 和 easy 过切换；SDD t+50 约 +14.58%，hard/failure 约 +11.23%，easy 约 1.81%。 |
| Stage31/32 external zero-shot / domain alignment | SDD -> external transfer，normalization、CORAL、latent adapter、mixed-domain selector。 | 失败。 | 坐标尺度、horizon、scene/goal、agent type、track length 不兼容；对齐分布不等于对齐决策目标。 |
| Stage33/34 external geometry | 坐标不变 features、relative targets、external row geometry、train-only goals。 | 局部正信号，不可部署。 | t50/hard 有提升，但 all/easy 不稳。 |
| Stage35 selective transfer | external hard/easy/failure labels、gain/harm/easy gate。 | 部分成功。 | all +12.13%、hard +13.98%、easy 0.041%，但 t+50 = 0，不能部署。 |
| Stage36 t50 repair | horizon-specific selector / t50 policy search / curriculum。 | t50 仍 0。 | t50 有 22.98% oracle headroom，但缺足够 past-only history / goal prototype / switchability signal。 |
| Stage37 causal history + goal prototype | K=8/16/32/64 history windows、scene-agnostic goal prototypes、t50 gain/harm/failure、安全 conformal policy。 | 成功。 | external all +13.48%，t50 +8.46%，CI [+7.69,+9.15]，hard +15.54%，easy 0.041%，16/16 gates。 |
| Stage38 bounded correction | Stage37 保护下训练 bounded delta / correction head。 | 不部署。 | correction 没稳定超过 Stage37，ordinary residual 容易伤 easy。 |
| Stage39/40 Transformer / JEPA / Hybrid neural dynamics | Causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 诊断为主，裸 neural 未部署。 | neural without fallback 不安全；JEPA downstream lift 未证明；受保护 neural 没稳定脱离 Stage37 floor。 |
| Stage41/42 source-level evidence | composite-tail、full-waypoint dynamics、row cache、source-level split、ablation、source-CV、t100/source calibration。 | 成功形成证据包，但有明确 blocker。 | 强机制主要是 baseline-family rollout context + validation-safe guard；t100/global metric/seconds 仍 blocked。 |

## 3. 成功路线详细总结

### 3.1 Stage26：SDD 上第一个可靠 deployable selector

Stage24 证明“oracle headroom 大”不等于训练 selector 能成功。Stage24 selector 把 easy 样本大量切错，t+50 反而大幅变差。

Stage25/26 的关键修复：

- 不再做 hard classification：`哪个 baseline 最好`。
- 改成 expected-FDE / cost-aware / regret-aware selection。
- 加 confidence gate。
- 加 predicted gain threshold。
- 加 easy guard / harm guard。
- failure predictor 作为辅助信号。
- 低 margin / 低 confidence 样本 fallback。

Stage26 结果：

| 指标 | 结果 |
| --- | ---: |
| benchmark | SDD official pixel-space raw-frame |
| t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |
| verdict | SDD best deployable selector |

意义：

- Stage26 是 SDD 上的当前 best deployable。
- 但它只支持 SDD pixel raw-frame claim，不能扩展成 metric/true 3D/foundation claim。

### 3.2 Stage37：external t+50 deployable 修复

Stage35/36 前的问题：

```text
all improvement = +12.13%
hard/failure = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
```

这说明外部迁移有正信号，但长时程 t+50 不敢切换，不能部署。

Stage37 的关键修复：

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

Stage37 结果：

| 指标 | 结果 |
| --- | ---: |
| external all improvement | +13.48% |
| external t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

意义：

- Stage37 是 external dataset-local raw-frame 下第一个可部署正迁移点。
- 它修复了“all/hard 有提升但 t50 为 0”的核心 blocker。

### 3.3 M3W-Neural v1 protected package

在 Stage37 safety floor 下，Stage41/42 形成了更完整的 protected neural / source-level evidence package。

核心结果：

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

意义：

- 这是目前最强 protected package。
- 但它仍依赖 safety floor，不能写成自由神经动力学模型已经成功。

### 3.4 Stage42 full-waypoint / source-level 关键证据

Stage42 做了很多“防止过度包装”的工作：从 report-level 平均数推进到 row-cache、source-level split、ablation、source-CV、source-family support。

| 阶段 | 核心结果 | 结论 |
| --- | --- | --- |
| Stage42-C | protected full-waypoint ADE all +18.58%，t50 +14.80%，t100 raw diag +22.86%，hard +19.52%，easy 0 | full-waypoint 有正信号 |
| Stage42-R | row-cache-backed combo gates 15/15，ADE all +0.052387，t50 +0.037934，easy degradation 0.001102 | 从 report-level 变成 row-cache-backed |
| Stage42-S | frozen row combo policy gates 13/13；positive domains ETH_UCY, TrajNet；UCY fallback-only | 冻结 policy；UCY 仍缺 candidate |
| Stage42-T | UCY rows 9540，all/t50/hard/easy 全 0 | UCY 不是 threshold 问题，而是缺 candidate prediction source |
| Stage42-U | UCY endpoint-to-full bridge t50 -49.21%，easy degradation 56.66% | endpoint 成功不能线性桥接成 full-waypoint 成功 |
| Stage42-V | strict pure-UCY full-waypoint gates 11/11，ADE all +22.08%，t50 +29.03%，t100 raw +14.75%，hard +22.95%，easy 0 | UCY full-waypoint candidate source 修复 |
| Stage42-AM | source-level full-waypoint test rows 47,458，ADE all +24.58%，t50 +22.02%，t100 raw +14.37%，hard +23.75% | source-level full-waypoint 正证据 |
| Stage42-AU | family_baseline_rel_only all +27.38%，t50 +23.73%；baseline_family_all all +28.78%，t50 +31.54% | 当前主机制是 baseline-family rollout context |
| Stage42-AW | UCY all +37.45%，t50 +24.53%，hard +35.51%，easy 为负 | UCY blocker 被 train-only internal validation 修复 |
| Stage42-AX | repaired protocol robustness gates 14/14；global all +35.31% low，t50 +28.54% low；h100 easy degradation 2.396% | all/t50/hard 稳；t100 easy 是弱点 |
| Stage42-AY | h100 easy 从 2.396% 修到 -0.650%；t100 gain 降为 +6.78% | t100 safety 修复，但收益变小 |
| Stage42-AZ | shadow holdout 下 AY h100 easy degradation 12.29%；source-support guard 后 t100 = 0 | AY t100 正收益不能作为独立稳健 claim |
| Stage42-BA | train-only t100 source-CV：ETH_UCY safe folds 0，TrajNet 1，UCY not_run；guard 后 all/t50/hard 保持正，t100 = 0 | t100 仍是 blocker/diagnostic |
| Stage42-BI | UCY independent-source t100 repair：mean +0.445914，min +0.425313，max easy 0.011340，gates 14/14 | UCY t100 easy blocker 修复；global t100 仍被 ETH_UCY/TrajNet 阻塞 |
| Stage42-BJ | UCY repaired；ETH_UCY 1 independent source、还差 2；TrajNet 0、还差 3；gates 14/14 | t100 进入 source acquisition / user-action 阶段 |
| Stage42-BK | ETH-Person XML 发现 5 个 ETH_UCY t100-capable 候选；TrajNet 本地文件为短 snippet，t100 files=0 | ETH_UCY 有本地 loader-gap 修复入口；TrajNet 仍需更长官方/用户 raw source |
| Stage42-BL | ETH-Person XML technical dry-run：5 strict independent sources，t100 windows 1485，mean +0.683549，min +0.496424，easy -0.014155 | 技术路径强正，但 terms 未确认，不能算 official/deployable/global t100 |
| Stage42-BM | ETH-Person terms audit：OpenTraj MIT 只能覆盖 toolkit/software，不能覆盖 ETH-Person 数据；local terms file 未找到 | 正确保留 technical evidence，同时阻止 official overclaim |
| Stage42-BN | source time/geometry calibration：ETH 2 源、UCY 4 源有 source-specific calibration evidence；global metric/seconds 仍 blocked | 可以做 source-specific calibrated subset，不能做 global metric claim |
| Stage42-BO | calibrated subset macro all +9.05%，t50 +7.07%，但 t50 min -10.78%，easy max +103.25% | 有正信号但过切换严重，不可部署 |
| Stage42-BP | source-family guard 后 easy max 0；macro all +5.76%，t50 +6.19%，hard +5.63%，但 t50 min -6.66% | 修 easy，但 t50 仍有负迁移 |
| Stage42-BQ | t50 同 family 至少 2 个 train+val support 才切换；all +4.24%，hard +4.03%，easy 0，t50 min 0，t50 macro 0 | 修 t50 负迁移为 non-harm，但没有正 t50 |
| Stage42-BR | `ETH_seq` 还差 1 个同族源，`UCY_students` 还差 2 个，`UCY_zara` 有源但 policy/model 无正 t50 | 把 t50 non-harm 的根因拆清楚 |

## 4. 失败路线详细总结

### 4.1 JEPA 为什么失败

现象：

- Stage18 JEPA non-collapse。
- 后续多轮 JEPA probe / auxiliary training 没有稳定 downstream lift。
- selector、failure predictor、correction、official t+50 没有因为 JEPA 稳定提升。

原因：

- latent variance 不等于部署收益。
- 当前部署任务的核心是“何时切换、何时 fallback、怎么不伤 easy”，不是单纯学一个 latent。
- JEPA 目标和 expected-FDE / gain/harm/easy guard 目标不够对齐。

结论：

```text
JEPA 保留为 auxiliary / diagnostic。
不能写成主贡献。
不能写成生成式 world model。
```

### 4.2 Stage24 hard selector 为什么失败

现象：

- selector oracle headroom 很大，约 46.2%。
- 训练出来的 selector t+50 improvement = -43.3%。
- easy degradation = 11.33%。

原因：

- oracle label 很多是低 margin / ambiguous。
- hard classification 强迫模型在差距很小的 baseline 之间切换。
- confidence calibration 不够。
- easy cases 被过度干预。

修复：

- Stage25/26 改为 expected-FDE prediction。
- 加 margin-aware filtering。
- 加 soft-label / regret-aware objective。
- 加 conservative fallback。
- 加 failure predictor / easy guard。

### 4.3 SDD -> external zero-shot 为什么失败

现象：

- Stage31 external zero-shot all/t50 大幅负。
- Stage32 normalization / CORAL / latent adapter 仍未修复。
- adapted selector 接近 0 improvement。

原因：

- SDD 是 pixel-space，external 是 dataset-local / weak-metric diagnostic。
- 坐标尺度、horizon、track length、scene/goal、agent type、数据域都不一致。
- 分布对齐不等于任务对齐；latent 距离变小不保证 selector 决策变好。

修复路径：

- Stage33/34 做 row geometry、relative-error target、train-only goals。
- Stage35 做 hard/easy/failure label。
- Stage37 做 causal history + scene-agnostic goal prototype。

### 4.4 external t+50 为什么长期卡住

现象：

```text
Stage35/36:
all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
```

但 Stage36 也证明：

```text
t+50 rows = 16263
t+50 oracle headroom = 22.98%
```

原因：

- 有可学空间，但原特征不足以支持安全切换。
- all-test objective 会淹没 t+50。
- 没有足够 history / goal prototype / switchability 信号时，安全策略只能 fallback。

修复：

- Stage37 加 K=8/16/32/64 past-only history。
- 加 scene-agnostic goal prototypes。
- 加 t50 failure/gain/harm predictors。
- 加 conformal safety。

### 4.5 ordinary residual / bounded correction 为什么不部署

现象：

- Stage38 correction 没有稳定超过 Stage37。
- residual 一旦放开容易伤 easy。

原因：

- 直接改 trajectory 比选择 baseline 风险更高。
- bounded residual 更安全，但提升不足。
- 在 external dataset-local 坐标下，残差模型更容易过拟合局部数据。

结论：

```text
Stage38 correction diagnostic only。
当前 deployable 仍是 Stage37 / teacher floor protected policy。
```

### 4.6 Transformer / Hybrid 为什么还不是主贡献

现象：

- Stage39/40 训练过 Transformer、JEPA、Hybrid。
- neural without fallback 不安全。
- neural with fallback 没有稳定独立超过 Stage37。

原因：

- Stage37 safety floor 很强。
- neural 容易学成复制 floor，而不是新增 dynamics。
- neural 输出一旦错，easy cases 更容易受伤。
- 当前 ridge/residual/tabular/sequence/graph context 多轮 ablation 表明：baseline-family rollout context 仍是最强机制。

结论：

```text
neural 必须在 Stage37 / teacher safety floor 下使用。
不能部署 ungated neural。
不能把 protected policy 成功包装成自由 neural world dynamics 成功。
```

### 4.7 t+100 为什么仍是 blocker

现象：

- Stage42-AY 一度修 h100 easy，但 AZ shadow holdout 不稳。
- BA source-CV guard 后 t100 回退为 0。
- BI 修复 UCY independent-source t100，但 ETH_UCY / TrajNet 仍不足。
- BL ETH-Person XML technical dry-run 强正，但 BM 证明 terms/license 未确认。

原因：

- 独立 source 支持不足。
- ETH_UCY / TrajNet 缺足够 t100-capable independent sources。
- local TrajNet 文件是短 snippet，不能支持 raw-frame t100。
- ETH-Person XML 技术上可行，但 terms/license 未确认。

结论：

```text
t100 可以报告为 raw-frame diagnostic。
不能写成 global stable positive transfer。
不能写成 seconds-level long-horizon。
```

### 4.8 calibrated subset t50 为什么现在只能 non-harm

现象：

- Stage42-BO macro all/t50 正，但 easy harm 和 t50 negative 让它不可部署。
- Stage42-BP 修 easy，仍有 ETH_seq_eth t50 negative。
- Stage42-BQ 用 source-family support guard 把 t50 negative 守到 0，但 positive_t50_fold_count 也变成 0。
- Stage42-BR 指出 source support 和 policy 两类 blocker。

原因：

- `ETH_seq` 只有 2 个 calibrated sources，leave-one-out 时同族 train+val support 不够。
- `UCY_students` 只有 1 个 calibrated source，根本不足以做同族 source-CV。
- `UCY_zara` 有 3 个 sources，但现有 policy/model 仍没有 validation-safe positive t50。

结论：

```text
calibrated subset 目前可以写 limited positive all/hard + t50 non-harm。
不能写 positive calibrated t50 transfer。
不能写 global metric/seconds-level M3W success。
```

## 5. 当前可以写的 claim

可以写：

```text
M3W 是一个受保护的 dataset-local / raw-frame 2.5D multi-agent world-state candidate。
Stage26 在 SDD pixel-space benchmark 上建立了 fallback-safe expected-FDE selector。
Stage37 修复了 external t+50 deployable transfer。
Stage41/42 source-level full-waypoint / baseline-family evidence 显示 protected raw-frame gains。
当前最强机制是 causal baseline-family rollout context + validation-safe gain/harm/easy guard。
```

可以写得更谨慎一点：

```text
M3W 当前最强结果来自安全选择和 baseline-family rollout context，
不是来自无保护端到端神经网络直接生成未来轨迹。
```

## 6. 当前不能写的 claim

不能写：

```text
M3W 是 true 3D world model。
M3W 是 foundation world model。
M3W 已解决 metric / seconds-level prediction。
M3W 已经 global t100 positive。
JEPA 是生成式 world model。
裸 Transformer / Hybrid 已可部署。
Stage5C 已执行或 ready。
SMC 已启用或 ready。
ETH-Person XML 已经 official converted/evaluated。
Stage42-BQ 是 positive t50 success。
fallback-only / t50=0 是 positive transfer。
```

## 7. 现在最值得做的下一步

### 下一步 1：UCY_zara family-specific t50 policy

Stage42-BR 已说明：

```text
UCY_zara has enough family sources,
but no validation-safe positive t50 policy yet.
```

这是当前最可执行的下一步，因为它不需要新数据许可：

- 只用 `UCY_zara01 / zara02 / zara03`。
- 做 family-only source-CV。
- 专门训练 t50 policy / gain-harm selector。
- 如果仍失败，说明是 policy/model/feature blocker，而不是 source-support blocker。

### 下一步 2：ETH-Person XML terms 确认后正式 conversion

Stage42-BL technical dry-run 很强：

```text
mean t100 improvement = +0.683549
min improvement = +0.496424
easy degradation = -0.014155
```

但 Stage42-BM 已经证明：

```text
license / terms 未确认。
OpenTraj MIT 不能自动覆盖 ETH-Person 数据。
```

所以必须先确认 terms，然后才能：

- official conversion
- no-leakage audit
- train-only source-CV
- 是否允许 ETH_UCY t100 claim

### 下一步 3：TrajNet 长 raw source

本地 TrajNet 文件是短 snippet：

```text
t100-capable files = 0
```

所以 TrajNet 需要合法官方或用户提供的更长 raw trajectories。否则 global t100 claim 永远被 TrajNet 卡住。

### 下一步 4：如果继续追 neural 主贡献，要换更强 protocol

当前多轮证据表明：

```text
baseline-family rollout context 是主机制。
history / goal / neighbor / graph / sequence / JEPA 独立贡献还没证明。
```

下一步神经路线应该不是重复小 MLP，而是：

- graph-neural interaction model
- source-level sequence model
- stronger scene token / route token
- full-waypoint loss
- multi-domain source split
- multi-seed/bootstrap
- 仍然保持 Stage37 / teacher safety floor

## 8. 最终当前 verdict

```text
项目是否跑通：是
是否 true 3D：否
是否 foundation world model：否
是否 metric：否
是否 seconds-level：否
是否 Stage5C：否
是否 SMC：否

SDD best deployable：
  Stage26 expected-FDE fallback-safe selector

external best deployable：
  Stage37 / teacher floor protected policy

overall best package：
  M3W-Neural v1 protected dataset-local raw-frame package

source-level strongest mechanism：
  baseline-family rollout context + validation-safe guard

t100 global claim：
  否，仍 blocked

calibrated-subset t50:
  non-harm yes
  positive transfer no

current verdict:
  M3W 是一个有实证进展的 protected 2.5D multi-agent world-state candidate。
  它已经有 SDD 与 external raw-frame 的正迁移证据，
  但仍不能被宣传成 true 3D / foundation / metric / seconds-level / ungated neural world model。
```

## 9. 我实际做出的主要工程资产

这个目标内已经形成的有意义资产包括：

- SDD world-state conversion、scene packs、episode index、baseline tables。
- SDD medium / true medium fast cache 与 selector training pipeline。
- Stage26 expected-FDE / regret-aware selector。
- external OpenTraj / ETH-UCY / UCY / TrajNet feature-store schema。
- external row geometry、train-only goals、relative-error targets。
- external hard/easy/failure labels。
- Stage37 causal history windows 和 scene-agnostic goal prototype。
- Stage37 t50 switchability / conformal safety selector。
- Stage38 bounded correction diagnostics。
- Stage39/40 neural dynamics diagnostics。
- Stage41/42 M3W-Neural v1 protected evidence package。
- row-cache-backed full-waypoint evaluations。
- source-level source-CV protocol。
- t100 source acquisition / support gap audit。
- ETH-Person XML technical dry-run + terms blocker audit。
- source-specific time/geometry calibration audit。
- calibrated-subset safety repair / t50 support repair / gap audit。

## 10. 简短结论

这条路线里真正成功的不是“直接训练一个大模型然后超过一切”，而是逐步发现：

```text
强 causal baseline + expected-FDE selection + gain/harm/easy safety + baseline-family rollout context
```

才是当前数据和任务下最可靠的世界状态建模方式。

神经网络、JEPA、Hybrid 并没有被隐藏失败；它们目前的结果说明：

```text
没有 Stage37 / teacher safety floor 的 neural dynamics 仍不可靠。
```

但项目也不是原地失败。它已经从 SDD-only 走到了 external deployable t50，再走到 source-level full-waypoint evidence 和 calibrated-subset support audit。当前最需要的不是夸大，而是继续补：

1. source support；
2. t100 independent sources；
3. verified terms/license；
4. source-specific calibration；
5. 真正能独立贡献的 graph/scene/sequence neural dynamics。
