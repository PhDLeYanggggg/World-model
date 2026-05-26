# M3W 完整目标复盘：路线、失败原因、成功证据与当前最强模型

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已有阶段报告、gate、README、`research_state.json`，并纳入 Stage42-BL `fresh_technical_dry_run_terms_unverified` 结果。本文是面向人的总账，不把 `not_run`、technical dry-run、fallback-only 或 license-blocked 结果包装成成功。

## 0. 一句话结论

M3W 已经从一个 SDD-only / baseline-selector scaffold，推进到一个 **受安全 floor 保护的 dataset-local raw-frame 2.5D 多智能体 world-state candidate**。  

当前最强可部署路线不是裸 Transformer、裸 JEPA、裸 Hybrid，也不是 Stage5C，而是：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
dominant effective mechanism:
  causal baseline-family rollout context
  + validation-selected gain/harm/easy guard
  + conservative fallback
```

它可以诚实写成：

```text
protected dataset-local raw-frame 2.5D multi-agent world-state candidate
```

它不能写成：

```text
true 3D world model
large-scale foundation world model
metric trajectory predictor
seconds-level long-horizon predictor
ungated neural world dynamics
latent generative world model
SMC-ready model
```

## 1. 必须保持的边界

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 不部署；当前强结果都需要 safety floor / conservative fallback。

## 2. 主要路线总览

| 路线 | 做了什么 | 结果 | 根因 / 解释 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建 per-agent multi-agent 2.5D world-state、causal baseline fallback、diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 true 3D / foundation。 |
| JEPA / WAM-style representation | Stage18/19 及后续多轮 JEPA non-collapse、probe、downstream lift。 | 失败为主。 | non-collapse 不等于 downstream lift；selector/failure/correction/t50 没有稳定改善。 |
| SDD official pixel benchmark | SDD 转 world-state shards，建 scene packs、episodes、GoalBench、HardBench、BaselineFailureBench。 | 成功。 | SDD 成为 official pixel-space raw-frame benchmark；但不是 metric。 |
| Stage24 hard selector | 在 medium SDD 上利用大 oracle headroom 训练 hard class selector。 | 失败。 | t+50 improvement = -43.3%，easy degradation = 11.33%；低 margin label 和过度切换伤 easy。 |
| Stage25/26 cost-aware selector | 改为 expected-FDE / regret-aware / fallback-safe selector。 | 成功。 | SDD t+50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 1.81%。 |
| Stage31/32 external zero-shot / alignment | SDD selector / latent 直接迁移到 OpenTraj/ETH-UCY/UCY/TrajNet，并做 normalization / CORAL / adapter。 | 失败。 | 坐标、scale、horizon、scene/goal、agent type 不兼容；分布对齐不等于目标对齐。 |
| Stage33/34 external geometry | 坐标不变 features、relative targets、row geometry、train-only goals、scene packs。 | 局部成功，不可部署。 | t50/hard 有正信号，但 all/easy 不稳。 |
| Stage35 selective transfer | 建 external hard/easy/failure label，训练 gain/harm/easy gate。 | 部分成功。 | all +12.13%、hard/failure +13.98%、easy 0.041%，但 t+50 = 0。 |
| Stage37 causal history + goal prototypes | past-only history windows、scene-agnostic goal prototypes、t50 switchability、conformal safety。 | 成功。 | external all +13.48%，t50 +8.46%，hard/failure +15.54%，easy 0.041%，16/16 gates。 |
| Stage38 bounded correction | 在 Stage37 保护下训练 bounded correction。 | 不部署。 | 没有稳定超过 Stage37；ordinary residual 容易伤 easy。 |
| Stage39/40 Transformer / JEPA / Hybrid | 训练 causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 诊断为主。 | 无保护 neural 不安全；受保护 neural 未稳定超过 Stage37；JEPA downstream lift 未证明。 |
| Stage41/42 protected source-level evidence | composite-tail、full-waypoint dynamics、row cache、source-level split、full-waypoint eval、ablation、baseline-family mechanism。 | 成功形成 evidence package。 | 强结果主要来自 safety floor + baseline-family rollout context + validation-only policy。 |

## 3. 成功证据

### 3.1 SDD 内部成功：Stage26

| 指标 | 结果 |
| --- | ---: |
| benchmark | SDD official pixel-space raw-frame |
| t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |
| Stage5C | 未执行 |
| SMC | 未启用 |

意义：Stage26 修复了 hard-class selector 的过度切换问题，成为 SDD 上当前可靠的 deployable selector 基座。

### 3.2 外部 t+50 修复：Stage37

| 指标 | 结果 |
| --- | ---: |
| external all improvement | +13.48% |
| external t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

意义：Stage35/36 有 all/hard 正信号但 t+50 为 0；Stage37 用 causal history、goal prototype、gain/harm/easy guard 修复了外部 t+50。

### 3.3 M3W-Neural v1 protected package

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

Bootstrap lower bounds:

| slice | low | mid | high |
| --- | ---: | ---: | ---: |
| all | 20.67% | 21.02% | 21.39% |
| t+50 | 13.06% | 13.66% | 14.26% |
| t+100 raw-frame diagnostic | 13.96% | 14.69% | 15.37% |
| hard/failure | 19.99% | 20.39% | 20.76% |

意义：这是当前最强 protected package，但它不是 ungated neural world dynamics 成功。

### 3.4 Stage42 source-level / full-waypoint 证据

| 阶段 | 关键结果 | 结论 |
| --- | --- | --- |
| Stage42-C | protected full-waypoint ADE all +18.58%，t50 +14.80%，t100 raw diagnostic +22.86%，hard +19.52%，easy 0 | full-waypoint 有正信号 |
| Stage42-R | row-cache-backed combo gates 15/15，ADE all +0.052387，t50 +0.037934，easy degradation 0.001102 | report-level combo 变成 row-cache-backed |
| Stage42-S | frozen row combo policy gates 13/13；positive domains ETH_UCY, TrajNet；UCY fallback-only | policy 冻结，UCY 仍缺 candidate |
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
| Stage42-BL | ETH-Person XML technical dry-run：5 strict independent sources，t100 windows 1485，mean +0.683549，min +0.496424，easy -0.014155，gates 13/13 | 技术路径强正，但 terms 未确认，不能算 official / deployable / global t100 |

## 4. 失败路线和原因

### 4.1 JEPA 为什么失败

现象：

- JEPA non-collapse。
- 但 selector、failure predictor、correction、official t+50 没有稳定 downstream lift。

原因：

- latent variance 只是表示没有 collapse，不代表能改善部署决策。
- 当前部署的关键是 gain/harm/easy/failure 的安全切换，不是单纯学一个未来 latent。
- JEPA 目标和 selector / failure / fallback policy 目标不够对齐。

结论：

```text
JEPA 保留为 auxiliary / diagnostic。
不能写成主贡献，也不能写成生成式 world model。
```

### 4.2 hard classification selector 为什么失败

现象：

- Stage24 oracle headroom 很大。
- validation-selected selector 反而 t+50 -43.3%。
- easy degradation 11.33%。

原因：

- oracle best label 很多是低 margin / ambiguous。
- hard classification 会强迫模型在差距很小的 baseline 间做过度切换。
- confidence calibration 不够，easy cases 被错切。

修复：

- 改成 expected-FDE / regret-aware / margin-aware / confidence-gated selector。
- 加 conservative fallback。
- 加 easy guard / harm predictor。

### 4.3 SDD -> external zero-shot 为什么失败

现象：

- Stage31/32 external zero-shot all/t50 大幅负。
- adapted selector 也接近 0。

原因：

- SDD 是 pixel-space；external 是 dataset-local / weak-metric diagnostic。
- 数据集间坐标尺度、frame/horizon、scene/goal、agent type、track length 都不同。
- normalization / CORAL / latent adapter 缩小分布距离，但没有解决“该不该切换”的任务目标。

修复：

- 建 external row geometry。
- 建 relative target。
- 建 train-only goals。
- 建 causal history window。
- 建 scene-agnostic goal prototypes。

### 4.4 t+50 为什么长期卡住

现象：

- Stage35 all +12.13%、hard +13.98%、easy 0.041%，但 t+50 = 0。
- Stage36 证明 t+50 有 16,263 rows 和约 22.98% oracle headroom。

原因：

- 有可学空间，但原特征不足以支持安全切换。
- all-test objective 会掩盖 t+50。
- 没有足够 history / goal prototype / switchability 信号时，安全策略只能 fallback。

修复：

- Stage37 加 K=8/16/32/64 past-only history。
- 加 scene-agnostic goal prototypes。
- 加 t50 failure/gain/harm predictors。
- 加 conformal safety。

### 4.5 ordinary residual / bounded correction 为什么不部署

现象：

- correction 无法稳定超过 Stage37。
- residual 一旦放开，容易伤 easy。

原因：

- 直接改 trajectory 比选择 baseline 风险更高。
- bounded 后更安全，但提升不足。

结论：

```text
Stage38 correction diagnostic only。
当前 deployable 仍是 Stage37 / teacher floor protected policy。
```

### 4.6 neural Transformer / Hybrid 为什么还不是主贡献

现象：

- Transformer / Hybrid 训练过，但没有稳定超过 Stage37。
- neural without fallback 不安全。

原因：

- Stage37 已经是强 safety selector。
- neural 很容易学到复制 safety floor，而不是稳定新增 dynamics。
- 当 neural 输出错时，easy cases 损伤更明显。

当前处理：

- neural 必须放在 Stage37 / teacher safety floor 下。
- 不能部署 ungated neural。
- 不能把 protected policy 成功包装成自由 neural world dynamics 成功。

### 4.7 t100 为什么仍是 blocker

现象：

- Stage42-AY 一度让 h100 easy 安全，但 AZ shadow holdout 显示不稳。
- Stage42-BA train-only source-CV 后，source-support guard 保 all/t50/hard/easy，但 t100 raw diagnostic 回退为 0。
- Stage42-BI 修复 UCY independent-source t100，但 ETH_UCY / TrajNet 仍不足。

原因：

- t100 独立 source 支持不足。
- ETH_UCY / TrajNet 缺足够 t100-capable independent sources。
- local TrajNet files 是短 snippet，不能支持 raw-frame t100。
- ETH-Person XML 技术上可行，但 terms/license 未确认。

结论：

```text
t100 可以报告为 raw-frame diagnostic。
不能写成 global stable positive transfer。
不能写成 seconds-level long-horizon。
```

## 5. 当前最强 claim 与禁止 claim

### 可以写

```text
M3W 是一个受保护的 dataset-local raw-frame 2.5D multi-agent world-state candidate。
Stage26 在 SDD pixel-space benchmark 上建立了 fallback-safe expected-FDE selector。
Stage37 修复了 external t+50 deployable transfer。
Stage42 source-level full-waypoint / baseline-family evidence 显示 protected raw-frame gains。
当前最强机制是 causal baseline-family rollout context + validation-safe gain/harm/easy guard。
```

### 不能写

```text
M3W 是 true 3D world model。
M3W 是 foundation world model。
M3W 已经解决 metric / seconds-level prediction。
JEPA 是生成式 world model。
裸 Transformer / Hybrid 已可部署。
Stage5C 已执行或 ready。
SMC 已启用或 ready。
t100 已经 global stable positive。
ETH-Person XML 已经 official converted/evaluated。
```

## 6. 当前最值得继续做的事

1. **先做 ETH-Person XML terms / license audit。**  
   Stage42-BL 技术 dry-run 很强，但 terms 未确认。确认前不能把它算成 official converted dataset，也不能把 t100 写成 deployable/global。

2. **补 TrajNet 原始长轨迹或合法官方 t100-capable source。**  
   本地 TrajNet 是短 snippet，不支持 raw-frame t100。TrajNet 是 global t100 claim 的硬 blocker。

3. **如果继续追神经世界模型主贡献，要重训 graph/scene-rich neural protocol。**  
   当前 source-level 主机制是 baseline-family rollout context。若要证明 neural dynamics，需要更强的 scene tokens、interaction graph、full-waypoint loss、multi-domain source split 和 multi-seed/bootstrap。

4. **继续保持 Stage37 / teacher safety floor。**  
   在 neural 真的稳定超过 Stage37 前，部署不能脱离 fallback floor。

## 7. 最终当前 verdict

```text
项目是否跑通：是
是否 true 3D：否
是否 foundation world model：否
是否 metric：否
是否 seconds-level：否
是否 Stage5C：否
是否 SMC：否

SDD best deployable：Stage26 expected-FDE fallback-safe selector
external best deployable：Stage37 / teacher floor protected policy
overall best package：M3W-Neural v1 protected dataset-local raw-frame package
source-level strongest mechanism：baseline-family rollout context + validation-safe guard
t100 global claim：否，仍 blocked
ETH-Person XML：technical dry-run positive，但 terms/license 未确认

current verdict:
  M3W 是一个有实证进展的 protected 2.5D multi-agent world-state candidate。
  它已经有 SDD 与 external raw-frame 的正迁移证据，
  但仍不能被宣传成 true 3D / foundation / metric / seconds-level / ungated neural world model。
```

