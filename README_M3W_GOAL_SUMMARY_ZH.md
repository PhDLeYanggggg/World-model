# M3W 长期目标总总结：尝试路线、失败原因、成功证据与当前边界

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
总结来源：`cached_verified` 汇总已有报告、gate、README、`research_state.json`，并纳入最近 Stage42-BI 的 `fresh_source_robust_easy_guard_repair` 结果。

这份 README 是给“训练一个真正强的真实世界多模态多智能体世界模型 M3W”这个长期目标看的总账。它不是论文包装稿，也不是成功宣传稿；它把我在这个目标内尝试过的路线、失败原因、成功证据、当前最强可部署模型和仍然不能宣称的边界集中写清楚。

## 0. 必须保持的诚实边界

当前 M3W 仍然必须这样描述：

```text
不是 true 3D world model。
不是 large-scale foundation world model。
仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
SDD 是 pixel-space benchmark，不是 metric benchmark。
external 是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
homography、metric scale、effective seconds 没有全局验证。
self-audited / visual-prior / auto-silver 不是 human gold。
Stage5C latent generative 没有执行。
SMC 没有启用。
JEPA 是 representation / auxiliary，不是 latent generative rollout。
无保护 neural dynamics 仍不安全，部署必须经过 Stage37 / teacher safety floor。
```

当前最准确的总 claim 是：

```text
M3W 目前是一个 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
它已经有外部 top-down 数据上的 protected positive transfer 证据，
但还不是 true 3D、不是 foundation、不是 metric/seconds-level world model。
```

## 1. 当前最强可部署模型

当前 best deployable 不是裸 Transformer / JEPA / Hybrid，也不是 Stage5C，而是受保护策略：

```text
current best deployable:
  M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
  safety floor = Stage37 selector / teacher floor
  deployment = protected / conservative / validation-selected

fallback rule:
  如果 confidence 低、predicted gain 小、easy risk 高、source support 不足，
  回退到 causal strongest baseline / Stage37 floor。
```

这条路的核心不是“神经网络无条件替代 baseline”，而是：

1. 用因果历史、baseline-family rollout、hard/easy/failure、gain/harm 信号找可安全切换的样本。
2. 只在 validation 证明安全的 slice 上切换。
3. 对 unsupported source/horizon/domain 回退。
4. test 只最终评估一次，不用 test 调阈值。

## 2. 关键结果总表

### 2.1 SDD 内部：Stage26 成为 SDD best deployable

| 项 | 结果 |
| --- | ---: |
| benchmark | SDD official pixel-space raw-frame |
| Stage26 selector t+50 improvement | 约 +14.58% |
| hard/failure improvement | 约 +11.23% |
| easy degradation | 约 1.81% |
| Stage5C | 未执行 |
| SMC | 未启用 |

意义：

- Stage26 修复了 Stage24/25 selector 的核心问题：不要 hard classification，而要 expected-FDE / regret-aware / fallback-safe selection。
- 这是 SDD 上可靠的部署基座，但只限 SDD pixel raw-frame，不是 metric，不是 true 3D。

### 2.2 外部 t+50 修复：Stage37 第一次达到 external deployable

| 项 | 结果 |
| --- | ---: |
| external all improvement | +13.48% |
| external t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

意义：

- Stage35/36 已有 all/hard 正信号，但 t+50 = 0，不能部署。
- Stage37 通过 past-only history windows、scene-agnostic goal prototypes、t+50 switchability、gain/harm/easy guard 和 conformal safety 修复了 t+50。
- 这是外部 dataset-local raw-frame 下第一个可部署正迁移点。

### 2.3 Protected M3W-Neural v1 package

相对于 Stage37 / teacher floor：

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

- 这是目前最强的 protected package。
- 但它仍依赖 safety floor，不能写成 ungated neural world dynamics 成功。

### 2.4 Stage42 full-waypoint / row-cache / source-level 证据

Stage42 把 Stage37 的 selector-level 成功继续往 full-waypoint 和 source-level 审计推进。

| 阶段 | 核心结果 | 结论 |
| --- | --- | --- |
| Stage42-C | protected full-waypoint ADE all +18.58%，t50 +14.80%，t100 raw diagnostic +22.86%，hard +19.52%，easy 0 | full-waypoint 有正信号 |
| Stage42-R | row-cache-backed combo gates 15/15，ADE all +0.052387，t50 +0.037934，easy degradation 0.001102 | report-level combo 变成 row-cache-backed |
| Stage42-S | frozen row combo policy gates 13/13；positive domains ETH_UCY, TrajNet；UCY fallback-only | policy 冻结，UCY 仍缺 candidate |
| Stage42-T | UCY rows 9540，all/t50/hard/easy 全 0 | UCY 不是 threshold 问题，而是缺 candidate prediction source |
| Stage42-U | UCY endpoint-to-full bridge t50 -49.21%，easy degradation 56.66% | endpoint 成功不能线性桥接成 full-waypoint 成功 |
| Stage42-V | strict pure-UCY full-waypoint gates 11/11，ADE all +22.08%，t50 +29.03%，t100 raw +14.75%，hard +22.95%，easy 0 | UCY full-waypoint candidate source 修复 |
| Stage42-AM | source-level full-waypoint test rows 47458，ADE all +24.58%，t50 +22.02%，t100 raw +14.37%，hard +23.75% | source-level full-waypoint 正证据 |
| Stage42-AU | family_baseline_rel_only all +27.38%，t50 +23.73%；baseline_family_all all +28.78%，t50 +31.54% | 当前主要有效机制是 baseline-family rollout context |
| Stage42-AW | UCY validation support repair：UCY all +37.45%，t50 +24.53%，hard +35.51%，easy 为负 | UCY blocker 被 train-only internal validation 修复 |
| Stage42-AX | repaired protocol robustness gates 14/14；global all +35.31% low，t50 +28.54% low；h100 easy degradation 2.396% | all/t50/hard 稳；t100 easy 是弱点 |
| Stage42-AY | strict t100 easy guard gates 17/17；h100 easy 从 2.396% 修到 -0.650%；t100 gain 降为 +6.78% | t100 safety 修复，但收益变小 |
| Stage42-AZ | shadow holdout 显示 AY t100 easy degradation 12.29%；source-support guard 后 t100 = 0 | AY t100 正收益不能作为独立稳健 claim |
| Stage42-BA | train-only t100 source-CV：ETH_UCY safe folds 0，TrajNet 1，UCY not_run；guard 后 all/t50/hard 保持正，t100 = 0 | t100 仍是 blocker/diagnostic |
| Stage42-BG | local t100 protected policy：UCY mean +0.440938，easy 0.01134，global t100 blocked | UCY local t100 有正 evidence，但不能 global claim |
| Stage42-BH | independent-source audit：UCY mean +0.483414，但 easy 0.063323 > 2% | 独立源协议下 UCY t100 仍不 deployable |
| Stage42-BI | source-robust easy guard repair：UCY mean +0.445914，min +0.425313，max easy 0.011340，gates 14/14 | UCY independent-source t100 easy blocker 修复；global t100 仍被 ETH_UCY/TrajNet 数据不足阻塞 |
| Stage42-BJ | post-BI t100 source package：UCY repaired 保留；ETH_UCY 1 个 independent source、还差 2 个；TrajNet 0 个、还差 3 个；gates 14/14 | 明确下一步必须合法补 ETH_UCY/TrajNet independent t100 source；没有自动下载，没有 overclaim |
| Stage42-BK | post-BJ local source verification：ETH-Person XML 发现 5 个 ETH_UCY t100-capable 候选；TrajNet 59 个本地文件均为短 snippet、t100 files=0；gates 11/11 | ETH_UCY 有本地 loader-gap 修复入口但需 license/terms + conversion/source-CV；TrajNet 仍需更长官方/用户 raw source |
| Stage42-BL | ETH-Person XML technical dry-run：5 个 strict independent sources，t100 windows 1485，t100 all folds safe-positive，mean +0.683549，min +0.496424，easy -0.014155，gates 13/13 | 技术路径强正，但 terms 未确认，所以不能算 official converted/evaluated，也不能宣称 deployable/global t100 |

## 3. 路线复盘：试了什么，结果是什么

### 路线 A：BPSG-MA / 2.5D world-state scaffold

做了什么：

- 建立 per-agent multi-agent trajectory world-state scaffold。
- 建立 causal baseline fallback。
- 建立 failure diagnostics / hard bench / goal bench。

结果：

- 成功成为稳定工程基座。
- 但不是 true 3D，不是 foundation，不是神经世界动力学主贡献。

原因：

- 它偏 deterministic / baseline fallback。
- 解决的是可运行、可审计、无泄露，不是大规模 representation / dynamics learning。

### 路线 B：JEPA / WAM-style representation pretraining

做了什么：

- Stage18 SAM-JEPA-2.5D。
- Stage19 WAM-style data registry / simulation / top-down / ego-video pretraining 数据策略。
- 后续在 Stage22/23/24/39/40/42 多次做 JEPA non-collapse 和 downstream probe。

结果：

```text
JEPA non-collapse = yes
downstream lift = not proven
deployable contribution = no
```

失败原因：

- latent variance 不等于 selector/failure/correction/t50 有 lift。
- JEPA 训练目标与部署目标不对齐：部署需要“何时切换、何时回退、easy 不伤、hard/failure 有收益”。
- 普通 JEPA 没有稳定改善 downstream heads。
- JEPA 不能被包装成 latent generative rollout。

当前处理：

- JEPA 保留为 auxiliary / diagnostic，不作为主 claim。

### 路线 C：SDD official pixel-space benchmark

做了什么：

- 解压 SDD 并转换为 world-state shards。
- 构建 60 个 SDD scene packs。
- 构建 lazy per-agent multi-agent episodes。
- 建立 t+10/t+25/t+50/t+100 raw-frame horizon audit。
- 建立 strongest causal baselines、GoalBench、HardBench、BaselineFailureBench。

结果：

- SDD official pixel-space benchmark 建立成功。
- no-leakage pass。
- strongest causal baseline 多数情况下是 damped velocity / scene_clamped。

限制：

- SDD 是 pixel-space，不是 metric。
- t+50/t+100 是 raw annotation-frame，不是 seconds-level。
- homography / scale / FPS stride 未全局验证。

### 路线 D：Stage24/25 selector 失败取证

做了什么：

- Stage24 修复 I/O，建立 true medium index 600,000 windows。
- 发现 selector oracle headroom 很大：约 46.2%。
- 但 validation-selected hard-class selector t+50 improvement = -43.3%，easy degradation = 11.33%。
- Stage25 分析 confusion、regret、margin、fallback harm、easy degradation。

失败原因：

1. oracle label low-margin 太多，best baseline 和 second best 差距小。
2. hard classification 把 ambiguous 样本强行 one-hot。
3. selector 不知道“切错的代价”，只学“哪个类最好”。
4. easy case 被过度切换。
5. 没有 conservative fallback / confidence / gain margin。

结论：

- oracle headroom 大不等于 trained selector 成功。
- 必须做 cost-aware / regret-aware / fallback-safe policy。

### 路线 E：Stage26 expected-FDE selector

做了什么：

- 构建 causal feature store。
- 不再预测 best baseline class，而是预测每个 baseline 的 expected FDE / risk。
- 加 conservative fallback。
- 使用 failure predictor 辅助。
- validation 选 threshold，test 只评一次。

结果：

- SDD t+50 约 +14.58%。
- hard/failure 约 +11.23%。
- easy degradation 约 1.81%。

成功原因：

- 选择器从 hard classification 改为 expected-cost selection。
- 只有 predicted gain 足够且 confidence 足够时才切换。
- easy risk 高时回退。

结论：

- Stage26 是 SDD best deployable base。

### 路线 F：SDD -> external zero-shot transfer

做了什么：

- Stage31 把 OpenTraj / ETH-UCY / UCY / TrajNet 等非 SDD top-down 数据转到外部 feature store。
- 建 external no-leakage、baseline、latent cache。
- 评估 SDD-trained / M3W-LAS zero-shot transfer。

结果：

- zero-shot external transfer 崩。
- all improvement 约 -92.67%，t50 约 -278.57%。
- adapted selector 接近 0。

失败原因：

1. SDD 是 pixel-space，external 是 dataset-local / weak metric diagnostic。
2. 坐标尺度不兼容。
3. horizon 定义不一致。
4. scene/goal/interaction 缺失。
5. agent type schema 不一致。
6. scale / homography 未验证。

结论：

- SDD-only 不能写成跨数据集 world model。

### 路线 G：domain normalization / latent adapter

做了什么：

- Stage32/33 做 raw、zscore、velocity-scale、path-length/speed、quantile normalization。
- 做 CORAL、whitening、linear latent adapter、domain-conditioned adapter。
- 做 coordinate-invariant features 和 relative-error target。

结果：

- 部分减少 distribution gap。
- 但不能稳定带来 predictive lift。
- mixed-domain 容易破坏 SDD easy preservation。

失败原因：

- 分布距离缩小不等于预测任务对齐。
- 目标、horizon、goal、agent type、scene context 没同步解决。
- 对 easy case 没有足够安全约束。

结论：

- normalization / adapter 是辅助，不是核心解决方案。

### 路线 H：external row geometry / train-only goals / relative baseline

做了什么：

- Stage34 补外部逐行几何。
- 建 train-only goals、scene packs、goal distances、route/density proxy。
- 重建 relative-error baselines。

结果：

- t+50 / hard 出现局部正信号。
- 但 all-test 为负、easy degradation 高，不可部署。

失败原因：

- 目标和特征开始有用，但选择策略不够 selective。
- 全量切换会伤 easy。
- held-out scene 的 goal context 不稳定。

结论：

- 外部 row geometry 是必要条件，但不是充分条件。

### 路线 I：selective transfer

做了什么：

- Stage35 构建 external hard/easy/failure labels。
- 训练 external hard detector、failure predictor、gain predictor、harm predictor。
- 做 selective transfer policy。

结果：

| 指标 | Stage35 |
| --- | ---: |
| all improvement | +12.13% |
| hard/failure improvement | +13.98% |
| easy degradation | 0.041% |
| t+50 improvement | 0.0 |

失败原因：

- all/hard/easy 过了，但 t+50 不切换。
- t+50 objective 被 all-test objective 淹没。
- 长 horizon 需要更完整的历史窗口和目标原型。

结论：

- selective transfer 保护了 easy，但 t+50 仍未修好。

### 路线 J：t+50 专用修复

做了什么：

- Stage36 做 t+50 forensics、horizon-specific selector、t+50 policy search。
- 发现 t+50 rows 有 16,263，oracle headroom 约 22.98%，不是没有空间。
- Stage37 构建 past-only history windows K=8/16/32/64。
- 构建 scene-agnostic goal prototypes：
  - straight_continue
  - slow_stop
  - left_turn
  - right_turn
  - reverse_or_u_turn
  - group_follow
  - density_avoid
  - exit_like_direction_from_past_motion
- 训练 t50_failure / gain / harm / safe selector。
- 加 conformal safety。

结果：

- Stage37 external t+50 修复成功：t50 +8.46%，CI [+7.69%, +9.15%]。

成功原因：

- t+50 需要 past-only temporal context，而不是单帧 eval metadata。
- scene held-out 时不能用 test endpoints，所以 goal prototype 必须 scene-agnostic。
- switch 必须受 gain/harm/easy/confidence 共同控制。

### 路线 K：bounded correction / residual

做了什么：

- Stage38 在 Stage37 保护下训练 bounded correction / dynamics head。
- 形式：selected baseline + bounded delta。
- 比较 correction without fallback、with fallback、hard-only、t50-only。

结果：

- correction 没有稳定超过 Stage37。
- 不部署。

失败原因：

- residual 容易在 easy case 上制造无必要偏移。
- hard/failure gain 不稳定。
- bounded residual 不是当前主要贡献。

结论：

- 当前不训练普通 residual 作为 deployable。

### 路线 L：Transformer / JEPA / Hybrid neural dynamics

做了什么：

- Stage39/40 构建 neural dataset。
- 训练 causal temporal Transformer。
- 训练 JEPA auxiliary。
- 训练 Hybrid。
- 加 Stage37 fallback。
- 做 small/optimization/trial。

结果：

- 无保护 neural 通常不安全。
- 有保护 neural 没有稳定超过 Stage37，直到后续 protected composite / row-cache / source-level 路线变强。
- JEPA downstream lift 仍未证明。

失败原因：

1. 神经网络容易复制 Stage37 floor，而不是学出更强 dynamics。
2. 没有 safety gate 时会伤 easy。
3. JEPA non-collapse 不等于 downstream lift。
4. 一般 Transformer/Hybrid 没抓住当前有效机制：baseline-family rollout context。

结论：

- 不能把裸 neural 写成成功。
- 当前 deployable neural 必须是 protected / teacher-floor / baseline-family context aware。

### 路线 M：Stage42 source-level / full-waypoint / mechanism audit

做了什么：

- 把 report-level combo 推到 row-cache。
- 做 frozen policy。
- 做 full-waypoint，而不是 endpoint shortcut。
- 做 source-level evaluation。
- 做 retrained ablation：
  - all_latent
  - no JEPA
  - no Transformer
  - no scene
  - no goal
  - no interaction
  - no fallback
  - latent only
  - baseline-family only
  - residual-context / neural-context / sequence-context / graph-context
- 做 weak-slice repair、source-CV、t100 source audit。

结果：

- full-waypoint 和 source-level protected policy 有强正证据。
- 但 ablation 显示当前最可靠贡献主要来自 baseline-family rollout context，不是 JEPA/goal/neighbor/graph 的独立贡献。
- t100 仍需要更多 independent sources。

结论：

- 当前最稳的论文贡献方向应写成：

```text
strict no-leakage protected baseline-family world-state policy
with source-level validation, full-waypoint evaluation, and safe fallback
```

而不是写成：

```text
ungated neural foundation world model
```

## 4. 失败路线与失败原因总表

| 失败/partial 路线 | 表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| JEPA downstream | non-collapse 但 selector/failure/t50 无 lift | 表征目标与部署目标不匹配 | diagnostic only |
| Stage24 hard selector | t50 -43.3%，easy degradation 11.33% | hard one-hot label、low margin、过度切换 | 改 expected-FDE / fallback |
| SDD -> external zero-shot | all -92.67%，t50 -278.57% | coordinate / horizon / scene / agent mismatch | 外部 schema、relative target、selective transfer |
| normalization / CORAL | gap 变小但预测不升 | distribution alignment != task alignment | auxiliary only |
| Stage34 external all/easy | t50/hard 局部正，all/easy 不稳 | 全量乱切、easy guard 不够 | selective transfer |
| Stage35 t50 | all/hard/easy 过，t50=0 | long-horizon 特征不足，policy 不切 | Stage37 history + prototypes |
| bounded correction | 不稳定超过 Stage37 | residual 伤 easy，hard gain 不稳 | not deployable |
| naked Transformer/Hybrid | easy 风险高或被 fallback 吃掉 | 没有学到可部署增量 | protected only |
| endpoint-to-full bridge | UCY t50 -49.21%，easy 56.66% | endpoint 成功不能线性代表 full waypoint shape | 直接训练 full-waypoint source |
| t100 global positive | 多次被 source-CV / shadow holdout 打回 | independent source 不足，easy safety 不稳 | t100 diagnostic/blocker |

## 5. 成功路线与成功原因总表

| 成功路线 | 成功原因 | 当前用途 |
| --- | --- | --- |
| SDD official pixel benchmark | 数据转换、no-leakage、baseline 审计完整 | SDD internal benchmark |
| Stage26 expected-FDE selector | cost-aware、regret-aware、fallback-safe | SDD best deployable |
| Stage37 t50 repair | past-only history + goal prototypes + gain/harm/easy guard | external best deployable floor |
| Protected composite M3W-Neural v1 | teacher floor + safety switch + bootstrap evidence | current best protected package |
| Full-waypoint direct training | 不再用 endpoint 插值偷换目标 | world-state shape evidence |
| Baseline-family rollout context | 因果 baseline 候选本身携带强动态先验 | 当前最强机制 |
| Source-level validation / source-CV | 防止同源泄露和 threshold overfit | claim safety |
| T100 easy-guard repair for UCY | 候选必须在所有 non-holdout source 上 positive/easy-safe | UCY local t100 blocker 修复 |

## 6. 现在可以写进论文的东西

可以写：

```text
1. 严格 no-leakage 的 top-down 2.5D multi-agent world-state benchmark/evaluation pipeline。
2. Strong causal baseline 不是弱对照，必须作为 deployment floor。
3. Hard-class selector 会因为 low-margin oracle labels 和 easy over-switching 失败。
4. Expected-FDE / regret-aware / confidence-gated fallback policy 能显著改善 SDD。
5. External transfer 需要 past-only history、scene-agnostic goal prototypes、hard/easy/failure switchability。
6. Stage37 在 external t+50 上实现 deployable positive transfer。
7. Source-level / full-waypoint / row-cache 证据显示 protected baseline-family world-state policy 有稳定 raw-frame 正收益。
8. t100 目前是 diagnostic/blocker，需要更多 independent t100 sources。
```

不能写：

```text
1. 已经是 true 3D world model。
2. 已经是 large-scale foundation model。
3. SDD/external 是 metric 或 seconds-level。
4. JEPA 已经证明主贡献。
5. Transformer/Hybrid 裸模型已经可部署。
6. Stage5C latent generative 已执行。
7. SMC 已启用。
8. t100 已经全局稳定正迁移。
9. simulation success 等于 real-world success。
10. auto-silver / visual-prior label 是 human gold。
```

## 7. 当前 t+100 状态

t+100 现在最容易被误读，所以单独写清楚。

已有事实：

- Stage37/early M3W package 曾有 t+100 raw-frame diagnostic positive。
- Stage42-AZ shadow holdout 显示 AY strict t100 guard 在独立 shadow 上 easy degradation 高。
- Stage42-BA train-only source-CV 后，把 unsupported `domain|100` slice 回退，t100 positive 变成 0。
- Stage42-BG local UCY t100 source-CV positive/easy-safe。
- Stage42-BH independent-source audit 发现 UCY mean positive，但 easy degradation 6.33% > 2%。
- Stage42-BI 用 source-robust easy guard 修复 UCY independent-source t100 easy：mean +0.445914、min +0.425313、max easy 0.011340。
- Stage42-BJ 把 BI 后的 blocker 转成 source package：ETH_UCY 只有 1 个 independent t100 source、还差 2 个；TrajNet 0 个、还差 3 个；当前 local inventory 对这些独立源需求已经耗尽。
- Stage42-BK 进一步检查 loader gap，发现 `ETH-Person/data/*.xml` 本地文件有 5 个 ETH_UCY t100-capable 独立候选；但它们仍只是 conversion candidates，必须先确认 license/terms，再转换、跑 no-leakage 和 train-only source-CV。
- Stage42-BK 同时解释 TrajNet blocker：本地 TrajNet 文件都是 8/20-step challenge snippets，不是 raw long-track source，所以不能修 raw-frame t100。
- Stage42-BL 对 ETH-Person XML 做技术 dry-run：严格去重后 5 个 independent sources，t100 source-CV 全 fold safe-positive，mean +0.683549，min +0.496424，easy -0.014155；但 license/terms 仍未确认，因此不能写成 official/deployable/global t100 success。
- 所以 global t100 positive claim 仍不允许。

当前 t+100 结论：

```text
UCY local independent-source t100 support = repaired positive/easy-safe。
Global t100 deployable positive claim = false。
t100 remains raw-frame diagnostic / source-limited blocker。
```

## 8. 当前应该继续做什么

最短路径不是继续堆模型，而是分两条：

### 路径 1：补 t100 source evidence

目标：

- ETH_UCY 至少补到可做 source-CV 的 independent t100 sources。
- TrajNet 至少补到可做 source-CV 的 independent t100 sources。
- UCY 保持 Stage42-BI source-robust guard，并做更大 source/fold 验证。

下一步建议：

```text
Stage42-BN:
  Stage42-BM 已确认 ETH-Person XML 仍是 terms/license blocker。
  如果用户确认 ETH-Person XML research-use terms，正式转换 ETH-Person XML 到 Stage42 external source rows，并 rerun official no-leakage + train-only source-CV。
  对 TrajNet 继续寻找官方/用户提供的 raw long-track source。
  不做 metric/seconds claim。
  不把 registry-only 写成 converted。
  不把 failed download 写成接入。
```

### 路径 2：把 baseline-family mechanism 写清楚并做更强 ablation

当前最强机制是 baseline-family rollout context。下一步应该：

- 固化 baseline-family 的数学定义。
- 把 `family_baseline_rel_only`、`baseline_family_all`、`no_family`、`history_only`、`goal_only`、`neighbor_only` 做同源 source-level ablation。
- 报告每个 domain / horizon / hard/easy 的 CI。
- 不把 JEPA/Transformer 写成主贡献，除非重新训练后真的有 lift。

## 9. 本次 README 更新记录

本次用户要求：

```text
总结在这个目标内你做了什么、尝试了什么路线、哪些失败、失败原因、哪些成功，
并把有意义的结果总结到一个 README 文件里。
```

本文件就是该总 README：

```text
README_M3W_GOAL_SUMMARY_ZH.md
```

同时保留相关更详细的长期账本：

```text
README_M3W_RESEARCH_SUMMARY_ZH.md
README_M3W_DETAILED_RESULTS_ZH.md
README_M3W_EXECUTION_SUMMARY_ZH.md
README_RESULTS.md
```

## 10. 最近验证状态

最近 Stage42-BM 已完成：

```text
runner = pass
focused pytest = 9 passed
full pytest = deferred for the ongoing long-running Stage42 goal
verdict = stage42_bm_eth_person_terms_audit_pass_claim_blocked
gate = 14 / 14
official_converted_dataset_claim_allowed = false
deployable_t100_claim_allowed = false
global_t100_positive_claim_allowed = false
```

随后 Stage42-BN 已完成 source-level time/geometry calibration audit：

```text
runner = pass
focused pytest = 11 passed
verdict = stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked
gate = 13 / 13
ETH source-specific metric/time sources = 2
UCY source-specific metric/time sources = 4
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
M3W official metric/seconds claim allowed = false
```

解释：ETH `seq_eth` / `seq_hotel` 和 UCY `zara01/02/03/students03` 有局部 H + 2.5fps / 0.4s annotation-step evidence，可作为未来 source-specific calibrated subset 的候选；但不能把整个 M3W 写成 metric 或 seconds-level。

随后 Stage42-BO/BP 已完成 calibrated-subset source-CV 与安全修复：

```text
Stage42-BO:
  source = fresh_calibrated_subset_source_cv
  verdict = stage42_bo_calibrated_subset_eval_partial
  gates = 10 / 13
  calibrated_sources = 6
  rows_total = 160338
  all_macro = +0.090510
  t50_macro = +0.070729
  t50_min = -0.107784
  easy_max = +1.032550

Stage42-BP:
  source = fresh_calibrated_subset_safety_repair
  verdict = stage42_bp_calibrated_subset_safety_repair_pass_limited_positive
  gates = 11 / 11
  all_macro = +0.057580
  t50_macro = +0.061868
  t50_min = -0.066609
  hard_macro = +0.056282
  easy_max = 0.0
  positive_fold_count = 3
  positive_t50_fold_count = 2

Stage42-BQ:
  source = fresh_calibrated_subset_t50_support_repair
  verdict = stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive
  gates = 12 / 12
  all_macro = +0.042380
  t50_macro = 0.0
  t50_min = 0.0
  hard_macro = +0.040266
  easy_max = 0.0
  positive_fold_count = 3
  positive_t50_fold_count = 0

Stage42-BR:
  source = fresh_calibrated_t50_source_support_gap_audit
  verdict = stage42_br_calibrated_t50_source_support_gap_audit_pass
  gates = 12 / 12
  families_audited = 3
  unsupported_family_holdout_count = 3
  families_with_additional_sources_needed = ETH_seq, UCY_students
  families_with_support_but_no_positive_t50 = UCY_zara
  ETH-Person XML local candidates = 5
  ETH-Person terms verified = false
```

解释：BO 证明 calibrated-subset 有 macro 正信号，但 `UCY_students03` easy degradation 爆到 103%，`ETH_seq_eth` t50 为负，所以不能部署。BP 加入 train+val source/source-family support guard 后，`UCY_students03` 被安全回退，easy harm 修复为 0；BQ 进一步要求 t50 同 family 至少两个 train+val 支持源，把 `ETH_seq_eth` t50 负迁移守到 0。代价是 t50 正迁移也被守没了：positive_t50_fold_count = 0。BR 将原因拆开：`ETH_seq` 少 1 个同族 calibrated source，`UCY_students` 少 2 个，`UCY_zara` 源数量够但 validation-safe t50 policy 没有正收益。因此只能写 limited positive all/hard + t50 non-harm repair，不能写 global calibrated-subset / metric / seconds-level / t50 positive success。

验证：

```text
python3 run_stage42_calibrated_subset_eval.py -> completed, BO partial
python3 run_stage42_calibrated_subset_safety_repair.py -> 11 / 11
python3 run_stage42_calibrated_subset_t50_support_repair.py -> 12 / 12
python3 run_stage42_calibrated_t50_source_support_gap_audit.py -> 12 / 12
focused pytest -> 6 passed for BQ/BR
python3 -m pytest tests -> 490 passed
```

已知 runtime 注意事项：

```text
Apple Silicon 上不要默认用 x86_64 Conda 跑训练。
训练应使用 .venv-pytorch/bin/python 的 arm64/universal 环境。
num_workers = 0。
torch_threads 可设 4 或 8。
入口脚本需要提前挡住 x86_64/Rosetta，避免 OpenMP/SHM 卡死。
```

## 11. 一句话总结

M3W 目前的真实质量是：

```text
强于早期 SDD-only selector scaffold，已经有 external dataset-local raw-frame protected positive transfer；
Stage37/Stage42 的 safety-floor / baseline-family / source-level full-waypoint 路线是当前有效方向；
但它仍不是 true 3D，不是 foundation，不是 metric/seconds-level，JEPA/裸 Transformer 也还不是主贡献。
```
