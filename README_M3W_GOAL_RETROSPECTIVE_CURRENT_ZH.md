# M3W 长期目标详细复盘：做过什么、失败了什么、成功了什么

更新时间：2026-05-26
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总 Stage18-Stage42 已生成报告、gate、README、`research_state.json`；最近已提交的 fresh 证据截至 Stage42-CB source robustness audit。
用途：这是当前 M3W 长期目标的单文件中文总览，用来回答“这个目标内到底做了什么、尝试了什么路线、哪些失败了、为什么失败、哪些成功了、当前最强模型是谁、下一步该怎么走”。

## 0. 当前必须诚实承认的事实

当前 M3W 不是以下东西：

```text
true 3D world model
large-scale foundation world model
global metric predictor
seconds-level long-horizon predictor
ungated neural world dynamics
latent generative world model
SMC-ready model
```

当前最准确的定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

严格边界：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down pedestrian 数据仍是 dataset-local / unverified weak-metric diagnostic，不是统一米制真实世界坐标。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 当前强结果都依赖 safety floor / conservative fallback；无保护 neural dynamics 不能部署。

## 1. 一句话总判定

这个长期目标已经从“SDD 上的 2.5D 轨迹 scaffold”推进到了“有外部数据正迁移证据的 protected raw-frame / dataset-local 多智能体 world-state candidate”。

最重要的成功：

- Stage26：在 SDD pixel-space 上训练出第一个可部署的 cost-aware selector。
- Stage37：修复 external t+50，形成 external deployable selector candidate。
- Stage41/42：把 Stage37 扩展成 protected neural / full-waypoint / source-level evidence package。
- Stage42-BY/BZ：修复并统计验证 TrajNet|50 与 UCY|50 protected t50 slices。

最重要的失败：

- JEPA non-collapse 但没有稳定 downstream lift。
- hard-class selector 因低 margin / easy 过切换严重失败。
- SDD -> external zero-shot 严重失败。
- 普通 normalization / latent alignment 不足以修复跨域。
- 普通 residual / correction 和 ungated Transformer/Hybrid 不能安全部署。
- t100 / metric / seconds-level / broad source generalization 仍有 blocker。

当前 best deployable：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

当前 honest claim：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

不能 claim：

```text
foundation world model
true 3D world model
global metric / seconds-level model
Stage5C generative model
SMC model
ungated neural dynamics model
broad source-level generalization without source-diversity caveat
```

## 2. 主要路线总表

| 路线 | 做了什么 | 结果 | 关键原因 |
| --- | --- | --- | --- |
| BPSG-MA / early scaffold | 建 per-agent multi-agent 2.5D world-state、causal baseline fallback、failure diagnostics。 | 成功作为稳定基座。 | 可运行、可审计、可 fallback，但不是 true 3D / foundation。 |
| Stage18/19 JEPA / WAM-style data | SAM-JEPA-2.5D、WAM-style simulation/top-down/ego-video 数据策略。 | JEPA non-collapse，但 downstream lift 未证明。 | non-collapse 不等于 selector/failure/correction/t50 改善。 |
| Stage20/21 web data / SDD conversion | 合法数据登记、OpenTraj/SDD 路径、SDD world-state shards。 | 成功建立 SDD official pixel-space 数据基座。 | SDD 数据、no-leakage、causal velocity 被显式审计。 |
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
| Stage41/42 long research | composite-tail、full-waypoint、row cache、source-level split、ablation、source-CV、t100/source calibration、safety-floor necessity、blocker matrix。 | 成功形成 protected evidence package，但仍有 blocker。 | 强机制主要是 baseline-family rollout context + validation-safe guard；t100/global metric/seconds 仍 blocked。 |

## 3. 成功路线详解

### 3.1 Stage26：SDD 上第一个可靠 deployable selector

Stage24 证明了一个重要负结果：selector oracle headroom 大，不代表训练出来的 hard-class selector 能部署。

Stage24 失败指标：

| 指标 | 结果 |
| --- | ---: |
| selector oracle headroom | 约 46.2% |
| trained selector t+50 improvement | -43.3% |
| easy degradation | 11.33% |
| 结论 | 不可部署 |

失败原因：

- 只做 hard classification：强迫模型预测“哪个 baseline 最好”。
- 很多样本 best 和 second-best margin 很小，oracle label 不稳定。
- easy 样本本来 fallback 就很好，但 selector 过度切换。
- 目标函数没有直接惩罚 harm over fallback。

Stage25/26 修复：

- 改成 expected-FDE / regret-aware / cost-aware 选择。
- 对每个候选 baseline 预测 expected FDE / risk。
- 如果 confidence low、predicted gain small、easy risk high，则 fallback。
- 使用 failure predictor 辅助判断是否允许切换。
- test set 只最终评估，不用 test 调 threshold。

Stage26 成功指标：

| 指标 | 结果 |
| --- | ---: |
| benchmark | SDD official pixel-space raw-frame |
| t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |
| verdict | SDD best deployable selector |

边界：这是 SDD pixel raw-frame selector 成功，不是 metric / true 3D / foundation。

### 3.2 Stage37：external t+50 deployable 修复

Stage35/36 前的状态：

```text
external all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
t+50 oracle headroom = 约 22.98%
```

这说明 external 不是没有可学空间，而是模型在 t+50 上不敢安全切换。

Stage37 做的关键工作：

- 构建 past-only history windows：
  - K=8
  - K=16
  - K=32
  - K=64
- 构建 history features：
  - history dx/dy
  - speed
  - acceleration
  - heading
  - curvature
  - turn angle
  - stop/go
  - dwell
  - path length
  - velocity decay
  - neighbor count
  - min neighbor distance
  - density
  - TTC
  - closing speed
- 构建 scene-agnostic goal prototypes：
  - straight_continue
  - slow_stop
  - left_turn
  - right_turn
  - reverse_or_u_turn
  - group_follow
  - density_avoid
  - exit_like_direction_from_past_motion
- 训练：
  - t50 failure predictor
  - t50 gain predictor
  - t50 harm predictor
  - t50 safe selector
  - conformal safety rule

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

意义：这是 external dataset-local raw-frame 下第一个可部署正迁移点。

### 3.3 M3W-Neural v1 protected package

Stage41/42 没有把 neural dynamics 包装成无保护成功，而是把它放在 Stage37/teacher safety floor 下做 protected evaluation。

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

结论：

```text
M3W-Neural v1 是 protected deployment candidate。
不是 ungated neural world dynamics。
```

### 3.4 Stage42 full-waypoint / source-level evidence

Stage42 的作用不是单纯提高一个数字，而是做论文级证据拆解：source-level split、full-waypoint dynamics、row cache、ablation、safety floor、blocker matrix。

关键结果：

| 子阶段 | 结果 | 解释 |
| --- | --- | --- |
| Stage42-C full-waypoint dynamics | protected full-waypoint ADE all +18.58%，t50 +14.80%，t100 raw diagnostic +22.86%，hard +19.52%，easy 0。 | full-waypoint world-state evidence 成立，但仍 protected。 |
| Stage42-H sequence ablation | 去掉 history tokens 后 t50/hard 明显下降。 | history 在 sequence encoder 下有用；flattened history + ridge 不足。 |
| Stage42-J static-gated full-waypoint | ADE all +3.62%，t50 +3.69%，hard +3.97%，FDE t50 +11.66%，easy 0。 | static context 不能全局硬混，要 validation-gated。 |
| Stage42-P t50 gain/harm selector | ADE all +5.15%，t50 +0.66%，hard +5.33%，easy 0.86%，FDE t50 +5.74%。 | t50 sign repaired，但 t50 CI low 仍负，不足以 paper-level 稳定 claim。 |
| Stage42-R row prediction cache combo | ADE all +5.24%，t50 +3.79%，t50 CI low +2.77%，hard +5.48%，easy 0.11%。 | row-level combo 修复 report-level preflight 缺口。 |
| Stage42-S frozen row combo policy | policy/cache/schema hash frozen；positive domains ETH_UCY、TrajNet；UCY fallback-only。 | 可审计部署 artifact，但 UCY 不是这个 combo 的正迁移证据。 |
| Stage42-AU baseline-family mechanism | family-baseline context protected all/t50 很强。 | 当前 source-level 主机制是 baseline-family rollout context。 |
| Stage42-BW safety-floor necessity | protected all/t50/hard = +21.03% / +13.65% / +20.38%，easy 0；ungated easy degradation = 124.59%。 | Stage37/teacher floor 是必要安全机制；无保护 neural 不可部署。 |
| Stage42-BY t50 floor-relaxability repair | repaired slices = TrajNet|50, UCY|50；global t50 +28.97%，global easy -37.05%。 | 修复 t50 slices，但不是 floor-free neural。 |
| Stage42-BZ bootstrap evidence | target union t50 CI [28.52%,29.45%]；hard/failure CI low +28.51%；easy CI high -25.16%。 | BY 的 t50 repair 有 bootstrap 支持。 |
| Stage42-CB source robustness audit | TrajNet|50 与 UCY|50 在 available major source 上 robust，但 source concentration limited。 | 可以说 available major-source robust，不能说 broad source generalization。 |

## 4. 失败路线与根因

### 4.1 JEPA：non-collapse 不等于 downstream lift

做过：

- Stage18 SAM-JEPA-2.5D。
- Stage19 WAM-style data strategy。
- 后续 JEPA-only / JEPA auxiliary / Hybrid 分支。

结果：

- JEPA latent non-collapse。
- selector / failure / correction / official t+50 没有稳定提升。

失败原因：

- JEPA 的 variance 正常只说明 latent 没塌，不说明它对决策有用。
- 表征目标和部署目标错位。
- t+50/hard/failure 真正关键是 gain/harm/easy safety switching。
- 当前数据规模和多模态 scene 信息不足以让 JEPA 成为主贡献。

当前结论：

```text
JEPA 只能写 representation diagnostic / auxiliary。
不能写成 latent generative world model。
不能写成当前主方法贡献。
```

### 4.2 hard-class selector：oracle headroom 大但训练失败

失败表现：

- Stage24 oracle headroom 约 46.2%。
- trained selector t+50 improvement = -43.3%。
- easy degradation = 11.33%。

根因：

- hard one-hot oracle best baseline label 有大量低-margin 噪声。
- selector 在 easy samples 上过度切换。
- 训练目标不是 regret-minimizing。
- 没有 confidence / gain / harm / easy fallback gate。

修复：

- Stage25/26 改成 expected-FDE / regret-aware / fallback-safe selector。

### 4.3 SDD -> external zero-shot：跨域直接迁移失败

失败表现：

- Stage31 zero-shot external transfer 严重负迁移。
- Stage32 normalization / CORAL / latent adapter / adapted selector 仍不能修复。

根因：

- SDD 是 pixel-space。
- external 是 dataset-local / unverified weak metric diagnostic。
- horizon、track length、scene/goal、agent type、scale/homography 不一致。
- latent distribution 更近，不代表 baseline decision 更正确。

修复路径：

- Stage33/34 补 row geometry、relative targets、train-only goals。
- Stage35 selective transfer 修 all/hard/easy。
- Stage37 history window + goal prototype 修 t+50。

### 4.4 普通 normalization / latent adapter：缩小距离但不带来预测价值

做过：

- per-scene zscore。
- velocity-scale normalized。
- path-length / speed normalized。
- robust quantile normalization。
- CORAL / whitening / linear latent adapter。

结果：

- 可以缩小某些 feature/latent distribution gap。
- 但没有稳定 positive transfer。

根因：

- 跨域问题不是单纯分布距离，而是：
  - 哪些 baseline family 在该 source/horizon 上可靠；
  - 哪些样本可切换；
  - 切换是否会伤 easy；
  - t50/hard/failure 是否有足够历史证据。

结论：

```text
domain alignment 只能作为辅助诊断。
不能替代 source/horizon/family-specific safety policy。
```

### 4.5 普通 residual / correction：容易伤 easy

失败表现：

- Stage38 bounded correction 没稳定超过 Stage37。
- ordinary residual / correction 在 easy 上不安全。

根因：

- baseline already-good 样本很多。
- residual 一旦过度干预，会损害 easy preservation。
- 没有足够可靠的 harm gate 时，hard improvement 换不来部署安全。

当前结论：

```text
correction specialist 不可部署。
部署仍保持 Stage37 / teacher floor。
```

### 4.6 Transformer / Hybrid：无保护不能部署

做过：

- Causal Temporal Transformer。
- JEPA auxiliary。
- JEPA + Transformer Hybrid。
- teacher distillation。
- hard/failure oversampling。
- t50-focused curriculum。
- Stage37 fallback gate。

结果：

- neural without fallback 不安全。
- neural with Stage37 fallback 没有稳定独立脱离 Stage37 floor。
- protected neural / full-waypoint evidence 有用，但部署仍依赖 safety floor。

根因：

- 神经网络容易学会复制 teacher/floor，而不是在所有 source/horizon 上安全改进。
- 无保护输出会在 easy samples 上产生过大 harm。
- scene/goal/interaction/static context 一旦没有 validation gate，容易变成负贡献。

当前结论：

```text
neural dynamics 目前是 protected auxiliary / candidate。
不是 floor-free deployable world dynamics。
```

### 4.7 t100 / metric / seconds：仍然 blocked

当前状态：

- t100 有 raw-frame diagnostic 局部正信号。
- 但 global t100 positive claim 仍 blocked。
- metric / seconds-level claim 仍 blocked。

原因：

- 独立 t100-capable sources 不足。
- ETH_UCY / TrajNet / UCY source families 支持不均。
- FPS、annotation stride、homography、meter-per-pixel 没有全局统一验证。
- dataset-local coordinates 不能混写成真实米制。

Stage42-BV blocker：

| blocker | 状态 | 下一步 |
| --- | --- | --- |
| `ETH_seq_t50_source_support` | blocked | 验证 ETH-Person terms 或提供 legal source-compatible ETH_seq long-track source。 |
| `UCY_students_t50_source_support` | blocked_narrowed | 还需要 1 个 independent t50-capable students-family source。 |
| `TrajNet_raw_long_t100_source_support` | blocked | 需要 legal raw long-track TrajNet-compatible sources。 |
| `ETH_UCY_global_t100_source_support` | blocked | terms/source support 不足，需重新 conversion + strict source-CV。 |
| `global_metric_seconds_claim` | blocked | 只能 source-specific calibrated subset，不能 global claim。 |

## 5. 为什么现在没有把 loss 当主指标

这个项目目前的关键不是“训练 loss 降了没有”，而是“部署决策有没有超过最强 causal fallback，并且不伤 easy”。

原因：

- selector / policy 的真实目标是 regret、harm、easy preservation、hard/failure gain，不是单一 supervised loss。
- 很多模型 train loss 可以下降，但部署时会在 easy samples 上乱切换。
- JEPA loss / representation loss 下降，也不保证 downstream lift。
- correction loss 下降，也可能损坏 fallback already-good 的样本。

所以当前核心指标是：

- improvement over strongest baseline / Stage37。
- t+50 improvement。
- hard/failure improvement。
- easy degradation。
- harm over fallback。
- selector regret。
- bootstrap CI。
- no-leakage gate。

loss 仍可记录，但不能作为部署成功标准。

## 6. 为什么当前没有直接做端到端神经网络替代

端到端神经网络已经尝试过，但当前不能直接替代 Stage37/teacher floor。

主要原因：

- 数据仍是 dataset-local / raw-frame / weak geometry，不是统一 metric 真实世界数据。
- external source/horizon 差异很大，端到端模型容易过拟合某个 source。
- easy preservation 是硬约束，端到端 residual 容易伤 easy。
- Stage37 的成功机制是“只在高 gain、低 harm、高 confidence 时切换”，不是每个样本都预测一条新轨迹。
- 无保护 neural without fallback 已经被 safety-floor audit 拒绝。

因此当前最稳路线是：

```text
causal baseline family rollout context
+ gain/harm/easy/failure safety gate
+ protected bounded neural / full-waypoint auxiliary
+ Stage37/teacher floor
```

端到端神经网络可以继续做，但必须在 protected gate 下证明：

- all/t50/hard 至少一个超过 Stage37。
- easy degradation <= 2%。
- no leakage pass。
- SDD safety 不破坏。

## 7. 当前 best deployable 是什么

当前 best deployable：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

部署逻辑：

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

为什么不是裸 neural：

- Stage38 correction 没有稳定超过 Stage37。
- Stage39/40 Transformer / JEPA / Hybrid 没有 floor-free 稳定 lift。
- Stage42-BW 证明 safety floor 必要：ungated endpoint/full-waypoint easy degradation 约 124.59%。
- 去掉 floor/safe rollout context 后 protected t50 下降约 9%。

## 8. 当前可以写成论文 claim 的内容

可以写：

```text
We develop a protected dataset-local raw-frame 2.5D multi-agent world-state candidate.
The strongest deployable evidence comes from regret-aware baseline-family rollout selection under strict no-leakage and validation-only safety policies.
Stage37 repairs external t+50 transfer, and Stage42 source-level/full-waypoint evidence supports protected raw-frame gains.
```

可以作为贡献的内容：

- strict no-leakage external top-down transfer protocol。
- cost-aware / regret-aware / conservative fallback baseline policy。
- causal history + scene-agnostic goal prototypes for external t+50。
- source-level / full-waypoint / validation-only policy evaluation package。
- safety floor necessity audit。
- blocker matrix / source-diversity caveat。

不能写：

```text
We solved true 3D world modeling.
We built a foundation world model.
We achieved global metric prediction.
We achieved seconds-level t+50/t+100.
JEPA is a generative world model.
Ungated Transformer/Hybrid is deployable.
Stage5C or SMC is ready/executed.
Broad source-level generalization is proven without caveats.
```

## 9. 当前最有意义的结果清单

| 项目 | 最有意义结果 | 是否可作为主 claim |
| --- | --- | --- |
| SDD Stage26 | t+50 +14.58%，hard/failure +11.23%，easy 1.81%。 | 可作为 SDD pixel-space selector claim。 |
| External Stage37 | all +13.48%，t50 +8.46%，CI [+7.69,+9.15]，hard +15.54%，easy 0.041%。 | 可作为 external dataset-local raw-frame protected selector claim。 |
| M3W-Neural v1 | all +21.03%，t50 +13.65%，t100 raw diagnostic +14.69%，hard +20.38%，easy 0。 | 可作为 protected world-state candidate claim。 |
| Stage42 full-waypoint | all +18.58%，t50 +14.80%，t100 raw diagnostic +22.86%，hard +19.52%，easy 0。 | 可作为 protected full-waypoint evidence。 |
| Stage42-BY/BZ t50 repair | target union t50 CI [28.52,29.45]，hard CI low +28.51%，easy CI high -25.16%。 | 可作为 protected t50 slice repair evidence。 |
| Stage42-CB source robustness | available major-source robust，但 source concentration limited。 | 可作为 caveated source robustness claim，不可作为 broad source generalization。 |
| JEPA | non-collapse。 | 只能 auxiliary / diagnostic，不是主 claim。 |
| Transformer/Hybrid | protected auxiliary / ablation evidence。 | 不能作为 floor-free deployable neural claim。 |
| t100 | raw-frame diagnostic 局部正信号。 | 不能作为 global deployable t100 claim。 |
| metric/seconds | 未验证。 | 不能 claim。 |

## 10. 现在最重要的 blocker

### 10.1 Source diversity blocker

Stage42-CB 显示：

- TrajNet|50 最大 source 占比 99.08%。
- UCY|50 最大 source 占比 100.00%。
- Stage42-CC 本地 inventory 扫描 93 个文件、10 个 t50-capable 文件，但 unused independent ready-to-claim t50 sources = 0。
- Stage42-CC 找到的 4 个 t50-capable 文件只是当前 source 的 alternate representation；1 个是 synthetic/diagnostic。
- Stage42-CD 已把 blocker 转成官方/manual acquisition package：UCY、ETH/BIWI、TrajNet++、OpenTraj toolkit、additional top-down target；auto-download=0，converted datasets=0。
- Stage42-CE 进一步检查这些 target 的本地路径：4 个 target 有 local path 且 schema_possible，3 个 target 有 t50/t100 文件，但 independent_t50_candidates=0，source_cv_ready=0。
- Stage42-CF 把 CE 结果接入 legal/source-identity gate：conversion_allowed_now=0，converted=0，evaluated=0；只有显式官方 terms/path confirmation 和 independent source identity 后，未来 conversion 才允许继续。

结论：

```text
available major-source robust: yes
broad source-level generalization: no
local source-diversity repair ready: no
official acquisition package ready: yes
conversion preflight ready: yes
source-CV repair ready: no
```

下一步：

- 需要更多 independent t50-capable external top-down sources。
- 不能把 source-concentrated success 写成 broad source generalization。
- 不能把 registry-only、alternate representation、synthetic/diagnostic inventory 写成 converted/evaluated dataset。
- Stage42-CD 的官方源包是 next-action evidence，不是外部泛化成功证据。
- Stage42-CE 的 local parseability 也不是 legal permission、conversion success 或 final-test evidence。
- Stage42-CF 的 terms confirmation template 是 checklist，不是 permission；不能把 template 或 local path 写成 legal/converted evidence。

### 10.2 t100 blocker

当前 t100 是 raw-frame diagnostic。

仍缺：

- legal raw long-track TrajNet-compatible sources。
- 足够 independent t100-capable sources。
- source-CV 下稳定 easy-safe positive t100。

### 10.3 metric / seconds blocker

仍缺：

- FPS / annotation stride 全局审计。
- homography / scale / meter-per-pixel 验证。
- source-specific calibration 的可复现路径。

### 10.4 floor-free neural blocker

仍缺：

- 无 Stage37/teacher floor 时仍能 all/t50/hard 提升。
- easy degradation <= 2%。
- source-level bootstrap / multi-seed 证据。

## 11. 下一步最短路径

1. **先补 source diversity，不要再包装 source-concentrated t50。**
   重点是获取或合法转换更多 independent t50-capable external top-down pedestrian sources，特别是 UCY_students、ETH_seq、TrajNet long-track。

2. **把 baseline-family rollout context 写成当前主机制。**  
   当前证据最强的是 causal baseline family + gain/harm/easy guard，不是 JEPA/Transformer 独立学会完整 world dynamics。

3. **继续做 protected neural，但目标要清楚。**
   下一个有意义 neural 目标不是“跑更大模型”，而是证明在 source-level split 下，neural/graph/scene/full-waypoint context 在 Stage37 floor 上有稳定增益，并且 easy 不被破坏。

4. **如果要追 metric/seconds，需要另开时间几何审计线。**
   不完成 FPS/stride/homography/scale，就不能把 raw-frame 结果说成真实秒级或米制。

## 12. 文件索引

当前入口文件：

```text
README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md
```

其他总结：

```text
README_RESULTS.md
README_M3W_LONG_GOAL_RETROSPECTIVE_ZH.md
README_M3W_FULL_GOAL_REVIEW_ZH.md
README_M3W_RESEARCH_SUMMARY_ZH.md
README_M3W_DETAILED_RESULTS_ZH.md
README_M3W_EXECUTION_SUMMARY_ZH.md
```

关键报告：

```text
outputs/stage37_t50_history/report_stage37_final.md
outputs/stage38_external_robustness/report_stage38_final.md
outputs/stage40_neural_optimization/report_stage40_final.md
outputs/stage42_long_research/report_stage42_final.md
outputs/stage42_long_research/model_card_stage42.md
outputs/stage42_long_research/a_journal_gap_stage42.md
outputs/stage42_long_research/safety_floor_necessity_audit_stage42.md
outputs/stage42_long_research/t50_repair_statistical_evidence_stage42.md
outputs/stage42_long_research/t50_source_robustness_audit_stage42.md
outputs/stage42_long_research/independent_t50_source_inventory_stage42.md
outputs/stage42_long_research/user_action_required_independent_t50_sources_stage42.md
outputs/stage42_long_research/source_diversity_acquisition_package_stage42.md
outputs/stage42_long_research/user_action_required_source_diversity_stage42.md
outputs/stage42_long_research/source_diversity_conversion_preflight_stage42.md
outputs/stage42_long_research/user_action_required_source_conversion_preflight_stage42.md
outputs/stage42_long_research/source_acquisition_status_stage42.md
```

## 13. 最终结论

```text
项目是否跑通：是，作为 protected dataset-local/raw-frame 2.5D world-state candidate。
是否 true 3D：否。
是否 foundation：否。
是否 metric/seconds-level：否。
是否 Stage5C ready/executed：否 / 未执行。
是否 SMC ready/enabled：否 / 未启用。
当前 best deployable：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。
当前最大 blocker：external source diversity、global metric/time calibration、t100 source-CV、floor-free neural dynamics。
```

最诚实的一句话：

```text
M3W 目前已经不是单纯 demo，它有 protected external raw-frame 正迁移和 source-level 证据；但它仍然是受安全地板保护的 2.5D world-state candidate，不是 true 3D / foundation / metric / seconds-level world model。
```
