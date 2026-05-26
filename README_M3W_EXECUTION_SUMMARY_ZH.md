# M3W 长期目标执行总结

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总已有 README、阶段报告、gate report、`research_state.json`；其中 Stage37、Stage42-AM/AW/AZ/BA 等关键结论来自对应 `fresh_run` 报告。  

这份文件回答一个问题：在“训练真正强的真实世界多模态多智能体世界模型 M3W”这个长期目标里，到底做了什么、试过哪些路线、哪些失败了、为什么失败、哪些成功了，以及现在可以诚实 claim 什么。

## 0. 先说边界

当前必须继续诚实承认：

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External 数据是 dataset-local / unverified weak-metric diagnostic，不是统一米制真实世界坐标。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 仍没有全局验证。
- self-audited / visual-prior / auto-silver 标签不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 不能部署；当前部署仍依赖 safety floor / fallback。

一句话结论：

```text
M3W 已经从 SDD-only selector scaffold 推进到 protected dataset-local raw-frame 2.5D multi-agent world-state candidate。
最强可部署证据来自 Stage37/teacher safety floor 保护下的 baseline-family / row-level full-waypoint policy。
但它还不是 true 3D、不是 foundation、不是 metric、不是 seconds-level，也不能执行 Stage5C 或 SMC。
```

## 1. 当前 best deployable

当前最强可部署候选：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
deployment protection = Stage37 selector / teacher safety floor
policy type = protected / conservative / validation-selected
Stage5C executed = false
SMC enabled = false
```

它的准确定位是：

```text
protected dataset-local raw-frame 2.5D multi-agent world-state candidate
```

它不能被描述为：

```text
true 3D world model
foundation world model
metric trajectory predictor
seconds-level long-horizon predictor
ungated neural dynamics model
latent generative world model
SMC-ready model
```

## 2. 我实际尝试过的路线

### 路线 A：BPSG-MA / 早期 2.5D scaffold

做了什么：

- 建立 per-agent multi-agent trajectory world-state scaffold。
- 建立 strongest causal baseline fallback。
- 建立 failure diagnostics、hard/failure subsets、goal/failure/correction 的早期评估。

结果：

- 成功作为稳定基座。
- BPSG-MA v1 可运行，但部署策略仍是 strongest causal baseline fallback + diagnostics。
- 它不是 true 3D，也不是 foundation world model。

为什么重要：

- 后面所有 selector / failure / world-state 模型都需要这个可审计基座。
- 如果没有 fallback 和 no-leakage audit，后面的神经模型很容易把失败包装成成功。

### 路线 B：JEPA / WAM-style representation

做了什么：

- Stage18 训练 SAM-JEPA-2.5D。
- Stage19 建 WAM-style data registry，区分 simulation、top-down、egocentric video。
- 后续 Stage22/23/24/39/40/42 多次检查 JEPA non-collapse 和 downstream lift。

成功点：

- JEPA 没有 collapse。
- 表征训练流程、数据角色和限制被建立起来。

失败点：

```text
JEPA downstream lift = not proven
selector lift = no
failure predictor lift = no
correction lift = no
official t+50 lift = no
```

失败原因：

- non-collapse 只说明 latent 有方差，不说明它能改善决策。
- 下游任务需要 gain / harm / easy preservation / fallback-aware signal，普通 JEPA target 不直接优化这些。
- JEPA 不能被当作 latent rollout 或生成式 world model。

当前结论：

- JEPA 只能保留为 diagnostic / auxiliary。
- 不能把 JEPA 写成当前主贡献。

### 路线 C：SDD official pixel-space benchmark

做了什么：

- Stage21 将 SDD 解压、转换为 per-video world-state shards。
- Stage22 构建 SDD official pixel-space benchmark、scene packs、episodes、GoalBench、HardBench、BaselineFailureBench。
- Stage23 因本地 NPZ I/O 慢只跑 quick-plus，明确没有包装成 full medium。
- Stage24 修复 SDD I/O，建立 fast cache 和 true medium index。

关键数据：

```text
SDD scenes = 8
SDD videos = 60
SDD tracks = 10300
SDD world-state rows = 10616256
raw-frame t+50 samples ~= 10009005
raw-frame t+100 samples ~= 9497463
split = train 40 videos / val 4 videos / test 16 videos
velocity = causal finite difference
coordinate status = pixel-space
metric status = no verified homography / scale
```

成功点：

- SDD 成为 official pixel-space raw-frame benchmark。
- no-leakage audit pass。
- strongest causal baseline 被稳定建立。

边界：

- SDD 不是 metric benchmark。
- t+50 / t+100 是 raw annotation-frame horizon，不是 seconds-level。

### 路线 D：SDD selector 从失败到 Stage26 成功

失败阶段：

- Stage24 selector oracle headroom 很大，约 `46.2%`。
- 但 validation-selected selector 失败：t+50 improvement `-43.3%`。
- easy degradation `11.33%`，严重伤害 easy cases。

失败原因：

- 直接 hard classification “哪个 baseline 最好”是错的。
- oracle best label 很多 low-margin / ambiguous 样本。
- selector 学会了过度切换，而不是最小化 regret。
- easy 样本本来 strongest baseline 很好，错误切换会造成大伤害。

修复路线：

- Stage25 改成 regret-minimizing / confidence-gated / fallback-safe policy。
- Stage26 建 feature-complete cost-aware expected-FDE selector。
- 输入只允许 past-only causal features。
- 预测每个 baseline 的 expected FDE / risk，而不是只预测 best class。
- 如果 confidence low、predicted gain 小、easy case、switch risk high，就 fallback。

Stage26 成功结果：

```text
t+50 improvement ~= 14.58%
hard/failure improvement ~= 11.23%
easy degradation ~= 1.81%
Stage5C = false
SMC = false
```

结论：

- Stage26 是 SDD 上第一个稳定 selector 基座。
- 但仍是 SDD pixel-space，不是 cross-domain / metric / true 3D。

### 路线 E：External zero-shot / domain alignment

做了什么：

- Stage31 建 external feature store、no-leakage、strongest baseline、latent cache、zero-shot transfer eval。
- Stage32 做 normalization、CORAL、feature whitening、domain adapter、domain-conditioned selector。
- Stage33 做 coordinate-invariant features、relative-error targets、external scene packs / train-only goals。
- Stage34 补 external row geometry、horizon/split、relative baselines v2、latent adapter v2。

最初失败：

```text
Stage31 SDD -> external zero-shot:
all improvement ~= -92.67%
t+50 improvement ~= -278.57%
```

失败原因：

- SDD 是 pixel-space，external 是 dataset-local / weak metric diagnostic。
- 坐标尺度、frame step、horizon、agent type、scene/goal context 不一致。
- external 缺 scene packs / train-only goals / interaction context。
- latent adapter 缩小分布距离，但没有带来 predictive lift。

Stage34 局部正信号：

```text
external t+50 diagnostic lift ~= +6.6%
hard/failure ~= +18% 到 +25%
all-test = negative
easy degradation = high
```

结论：

- 普通 domain alignment 不够。
- 必须做 selective transfer：只有 hard/failure 且 gain 高、harm 低时才切换。

### 路线 F：Stage35 selective transfer

做了什么：

- 扩容 external 数据。
- 建 external hard/easy/failure labels。
- 训练 hard detector、failure predictor、gain predictor、harm predictor。
- 建 selective transfer policy。

结果：

```text
external all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
```

成功：

- all / hard / easy 三个关键指标首次同时稳住。

失败：

- t+50 仍为 0，不能部署为 external t+50 policy。

失败原因：

- t+50 样本虽然有 oracle headroom，但 policy 不知道什么时候安全切换。
- all-test objective 淹没了 t+50。
- 现有 feature 不足以判断 long-horizon drift / goal direction / history shape。

### 路线 G：Stage37 causal history + scene-agnostic goal prototypes

做了什么：

- 构建 past-only history windows：K=8/16/32/64。
- 构建 history speed、accel、heading、curvature、density、TTC、neighbor features。
- 构建 scene-agnostic goal prototypes：
  - straight_continue
  - slow_stop
  - left_turn
  - right_turn
  - reverse_or_u_turn
  - group_follow
  - density_avoid
  - exit_like_direction_from_past_motion
- 训练 t+50 failure / gain / harm / switchability。
- 用 conformal safety 控制 easy degradation。

Stage37 成功结果：

```text
all improvement = +13.48%
t+50 improvement = +8.46%
t+50 bootstrap CI = [+7.69%, +9.15%]
hard/failure improvement = +15.54%
easy degradation = 0.041%
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

为什么成功：

- t+50 不再只靠阈值，而是有完整 past-only history window。
- 目标原型提供了跨 scene 的方向先验，不需要 test endpoints。
- gain/harm/easy guard 避免在 easy cases 上乱切换。

结论：

- Stage37 是 external t+50 修复的第一个 deployable 成功点。
- 但它主要还是 protected selector / policy success，不是无保护 neural world dynamics success。

### 路线 H：Stage38 correction / bounded dynamics head

做了什么：

- 冻结 Stage37 deployable policy。
- 训练 bounded correction / dynamics head：

```text
prediction = selected_baseline + alpha * bounded_delta
```

结果：

- correction 没有安全超过 Stage37。
- correction 不部署。

失败原因：

- residual / correction 容易伤害 easy cases。
- 直接改 trajectory 比安全选择 baseline 更危险。
- bounded residual 对 hard/failure 有局部信号，但不足以超过 Stage37 protected policy。

结论：

- 当前 external best 仍是 Stage37 selector。

### 路线 I：Stage39/40 neural world dynamics

做了什么：

- 在 Stage37 safety floor 下训练：
  - Causal Transformer
  - JEPA auxiliary
  - Hybrid JEPA + Transformer
  - neural with fallback
  - neural without fallback
- 后续 Stage40 做 failure diagnosis、teacher distillation、multi-task loss、hard/failure oversampling、t50 curriculum。

结果：

- neural 没有稳定超过 Stage37。
- 无保护 neural 不能部署。
- JEPA downstream lift 仍不成立。

失败原因：

- neural without fallback 会在 easy cases 上造成灾难性 harm。
- neural with fallback 往往被 safety floor 吃掉，真实增益不足。
- JEPA non-collapse 不等于 gain/harm/easy-safe downstream lift。
- Transformer/Hybrid 需要更强的 source-level full-waypoint supervision，而不是只在 selector-level 证明。

结论：

- Stage39/40 是重要负结果：不要把神经网络失败包装成成功。
- 当前 best deployable 仍是 Stage37/teacher-protected policy。

### 路线 J：Stage41/42 protected full-waypoint / row-level evidence

做了什么：

- Stage41/42 把 selector-level external success 推到 full-waypoint / row-level / source-level evidence：
  - composite-tail safe-switch
  - full-waypoint dynamics
  - static-gated full-waypoint
  - row-level prediction cache
  - unified external full-waypoint policy
  - source-level validation repair
  - UCY full-waypoint source
  - ablation matrix
  - source-level full-waypoint evaluation
  - source-level residual / neural / sequence / graph context ablations
  - baseline-family mechanism audit
  - t100 easy-safety / shadow-holdout / source-CV audits

M3W-Neural v1 package 关键结果：

```text
gates = 41 / 41
rows = 55,528
all ADE improvement = 21.03%
t+50 ADE improvement = 13.65%
t+100 raw-frame diagnostic ADE improvement = 14.69%
hard/failure ADE improvement = 20.38%
easy degradation = 0.00%
positive external domains = 3
all-agent composite FDE improvement = 19.82%
all-agent composite FDE@50 improvement = 17.39%
```

bootstrap lower bounds：

```text
all low = 20.67%
t+50 low = 13.06%
t+100 raw-frame diagnostic low = 13.96%
hard/failure low = 19.99%
```

Stage42-AM source-level full-waypoint result：

```text
test rows = 47,458
ADE all = +24.58%
ADE t+50 = +22.02%
ADE t+100 raw-frame diagnostic = +14.37%
ADE hard/failure = +23.75%
easy degradation = -25.66%
gates = 12 / 12
```

Stage42-AU baseline-family mechanism：

```text
family_baseline_rel_only protected all = +27.38%
family_baseline_rel_only protected t+50 = +23.73%
baseline_family_all protected all = +28.78%
baseline_family_all protected t+50 = +31.54%
```

结论：

- 当前最强机制不是“JEPA 单独成功”或“Transformer 无保护成功”。
- 当前最强机制是：

```text
多候选因果 baseline family rollout context
+ validation-selected expected-FDE / gain / harm / easy guard
+ Stage37 / teacher safety floor
+ source-level full-waypoint row policy
```

## 3. 失败路线总表

| 失败路线 | 失败表现 | 主要原因 | 后续修复 |
| --- | --- | --- | --- |
| JEPA-only | non-collapse 但 downstream lift 无 | target 与 selector/failure/gain/harm 不对齐 | 保留为 auxiliary，不作为主贡献 |
| Stage24 hard selector | t50 -43.3%，easy degradation 11.33% | hard label 低 margin，过度切换 | Stage26 expected-FDE / regret / fallback |
| SDD->external zero-shot | all -92.67%，t50 -278.57% | 坐标、horizon、scene/goal、agent type mismatch | Stage33-37 重建 external geometry / history / goal prototype |
| latent adapter | 分布距离缩小但无 predictive lift | alignment 不等于任务目标对齐 | 转向 row geometry / relative target / source-level policy |
| Stage35 selective transfer | all/hard/easy 过，t50=0 | t50 缺 history/goal/switchability 信息 | Stage37 history windows + goal prototypes |
| bounded correction | 没超过 Stage37 | residual 容易伤 easy，hard 局部不够稳定 | 不部署 correction |
| ungated neural | easy harm / 不安全 | 没有 safety floor | 保留 Stage37 floor |
| JEPA/Transformer/Hybrid | 没稳定超过 Stage37 | neural target 与安全部署目标仍不够一致 | Stage42 转向 protected full-waypoint/source-level evidence |
| endpoint-to-full bridge | UCY endpoint 成功不能线性变 full-waypoint | endpoint 不等于 trajectory shape | Stage42-V 直接训练 UCY full-waypoint |
| goal/neighbor/graph 独立贡献 | 多个 ablation 下不稳定 | 当前 protocol 被 baseline-family context 主导 | 不能过 claim；需更强 graph/scene-rich protocol |
| t100 robust claim | AY 正收益在 shadow/source-CV 下不稳 | 独立 t100 source 支持不足，easy safety 不稳 | BA source-CV guard 后 t100 回退 0，保 all/t50/hard |

## 4. 成功路线总表

| 成功路线 | 关键结果 | 说明 |
| --- | ---: | --- |
| SDD official pixel-space benchmark | 10.6M rows，no-leakage pass | official raw-frame pixel benchmark 建立 |
| Stage26 SDD selector | t50 +14.58%，hard/failure +11.23%，easy +1.81% | SDD best deployable selector |
| Stage37 external t50 repair | all +13.48%，t50 +8.46%，hard +15.54%，easy 0.041% | external deployable selector candidate |
| M3W-Neural v1 protected package | all +21.03%，t50 +13.65%，hard +20.38%，easy 0 | protected external 2.5D candidate |
| Stage42-AM source-level full-waypoint | all +24.58%，t50 +22.02%，hard +23.75% | fresh source-level full-waypoint evidence |
| Stage42-AU mechanism audit | baseline_family_all all +28.78%，t50 +31.54% | 当前主机制是 baseline-family rollout context |
| Stage42-AW UCY repair | UCY all +37.45%，t50 +24.53%，hard +35.51% | train-only internal validation 修复 UCY support |
| Stage42-AX robustness | all CI low +35.31%，t50 CI low +28.54%，hard CI low +33.52% | repaired protocol robust，但 t100/easy 有弱切片 |
| Stage42-AY t100 easy repair | h100 easy CI high 修到 0.983%，all +30.55%，t50 +28.97% | t100 safety 修复，但 t100 claim 仍弱 |
| Stage42-BA source-CV repair | all +28.10%，t50 +28.97%，hard +25.16%，t100=0 | 证明 t100 正收益源支持不足，安全回退 |
| Stage42-BB t100 data gap audit | gates 14/14；ETH_UCY 缺 2 个安全 t100 source，TrajNet 缺 1 个，UCY 缺 1 个 t100-capable source | 把 t100 blocker 转成数据/标定行动清单 |

## 5. 为什么 t100 现在仍是 blocker

Stage42-AY 一度让 t100 raw-frame diagnostic 有正收益，但 Stage42-AZ / BA 做了更严格验证：

- Stage42-AZ shadow-holdout 发现 AY strict t100 guard 在独立 shadow holdout 上不够稳，`h100 easy degradation = 12.29%`。
- 加 source-support guard 后，all/t50/hard/easy 仍为正或安全，但 t100 变成 `0.0`。
- Stage42-BA 用 train-only source-CV 再次检查：
  - ETH_UCY safe-positive t100 folds = `0 / 4`
  - TrajNet safe-positive t100 folds = `1 / 3`
  - UCY 因 t100 train sources 少于 3 个而 `not_run`
  - 因此没有任何 domain 有足够独立 t100 source support。

Stage42-BA source-CV guard 后：

```text
ADE all = +28.10%
ADE t+50 = +28.97%
ADE t+100 raw-frame diagnostic = 0.0
ADE hard/failure = +25.16%
easy degradation = -37.24%
```

这不是坏消息被隐藏，而是安全策略在起作用：

```text
t100 positive gain lacks train-only source-CV support,
so deployment-safe policy must guard unsupported t100 slices to floor.
```

所以当前可以说：

- all / t50 / hard/failure 仍有强 protected 正证据。
- t100 只能写 raw-frame diagnostic blocker。
- 不能把 t100 写成 stable long-horizon success，更不能写 seconds-level。

Stage42-BB 把这个 blocker 进一步转成可执行的数据缺口清单：

```text
verdict = stage42_bb_t100_data_gap_audit_pass_with_data_blocker
gates = 14 / 14
unsupported_t100_domains = ETH_UCY, TrajNet, UCY
ETH_UCY additional safe t100-capable train sources needed = 2
TrajNet additional safe t100-capable train sources needed = 1
UCY additional t100-capable train sources needed = 1
metric/seconds claim allowed = false
Stage5C = false
SMC = false
```

对应行动文件是：

```text
outputs/stage42_long_research/user_action_required_t100_stage42.md
```

这一步没有训练新模型；它的价值是把“t100 不能 claim”从一句 limitation 变成明确的数据需求：需要更多合法、独立、t100-capable 的 top-down pedestrian sources，或 source-specific t100 safety repair，并且需要官方 FPS/stride/homography/scale 证据才能升级时间/metric 口径。

## 6. 哪些东西现在可以写进论文

可以写：

- 受保护的 dataset-local raw-frame 2.5D multi-agent world-state candidate。
- SDD pixel-space official benchmark 和 external dataset-local top-down transfer 证据。
- Stage26/37 证明 cost-aware / regret-aware / fallback-safe selector 是必要机制。
- Stage37 证明 past-only history windows + scene-agnostic goal prototypes 能修复 external t+50。
- Stage42 source-level full-waypoint / row-level cache 证明 protected full-waypoint dynamics 有正证据。
- Baseline-family rollout context 是当前最强可解释机制。
- No-leakage、validation-only threshold、train-only normalization、future-as-label-only 是核心实验贡献。
- 负结果：JEPA、ungated neural、plain latent adapter、hard classification selector、endpoint-to-full linear bridge 都不能包装成成功。

不能写：

- M3W 已经是 true 3D world model。
- M3W 已经是 foundation world model。
- SDD / external 结果是 metric 或 seconds-level。
- JEPA 是生成式 world model。
- Stage5C 已执行或 ready。
- SMC 已启用或 ready。
- 无保护 Transformer/Hybrid 可以部署。
- goal/scene/neighbor/graph 在所有 source-level ablation 下都有统一正贡献。
- t100 已经稳定成功。

## 7. 当前最短下一步

1. **不要继续 overclaim t100。**  
   要么补更多独立 t100-capable top-down pedestrian sources，要么训练真正 source/domain-specific t100 policy。当前 BA 证明 t100 positive gain 缺独立 source-CV 支持。

2. **把 baseline-family rollout context 正式写成当前主机制。**  
   现在最强证据不是“JEPA/Transformer 独立学出世界动力学”，而是“多候选因果 baseline family + validation-safe selector/policy 形成 protected world-state dynamics candidate”。

3. **若继续追求神经 world model 主贡献，需要更强 graph/scene-rich protocol。**  
   不能再把简单 JEPA/Transformer/MLP 负结果包装成成功。下一步要用 richer scene tokens、graph neural interaction、full-waypoint loss、multi-domain source-level split、multi-seed/bootstrap 重新证明 history/goal/interaction 的独立贡献。

4. **若要升级 metric / seconds claim，必须做时间几何审计。**  
   需要官方 homography、FPS、annotation stride、scale direction、meter-per-pixel 验证；否则所有结果继续写 raw-frame / dataset-local。

## 8. 给你的直接回答

### 我做了什么？

我把项目从一个 2.5D trajectory scaffold 推进到一个受保护的 external raw-frame multi-agent world-state candidate：包括 SDD benchmark、external feature store、selector/failure/gain/harm、安全 fallback、history windows、goal prototypes、full-waypoint row cache、source-level split、bootstrap/gates、paper package 和多轮失败审计。

### 我试了哪些路线？

我试了 JEPA、WAM-style data、SDD official benchmark、hard selector、expected-FDE selector、domain normalization、latent adapter、external row geometry、selective transfer、t50 history/goal prototype、bounded correction、Transformer/JEPA/Hybrid neural、full-waypoint dynamics、static-gated dynamics、row-level gain/harm selector、source-level ablation、graph/sequence/neural residual context、baseline-family mechanism 和 t100 source-CV repair。

### 哪些失败了？

失败最明确的是：

- JEPA downstream lift。
- hard classification selector。
- SDD->external zero-shot。
- plain latent/domain alignment。
- correction/residual 直接部署。
- ungated neural dynamics。
- endpoint-to-full linear bridge。
- goal/neighbor/graph 的统一独立贡献。
- t100 stable positive claim。

### 为什么失败？

核心原因是这个任务不是“预测一个平均轨迹”这么简单，而是要在强因果 baseline 很强、easy case 很容易被伤害、跨数据坐标不统一、horizon 不匹配、scene/goal 不完整的情况下，决定什么时候切换、什么时候不切换。只要模型不知道 gain/harm/easy risk，就会过度切换或在 external 上崩。

### 哪些成功了？

成功最明确的是：

- SDD pixel-space official benchmark 建立。
- Stage26 SDD selector 成功。
- Stage37 external t+50 修复成功并可部署。
- M3W-Neural v1 protected package 有强 positive evidence。
- Stage42-AM source-level full-waypoint fresh evaluation 成功。
- Stage42-AU 证明 baseline-family rollout context 是当前主机制。
- Stage42-AW 修复 UCY validation support。
- Stage42-BA 明确 t100 blocker，同时保护 all/t50/hard/easy。
- Stage42-BB 把 t100 blocker 转成 ETH_UCY / TrajNet / UCY 的具体 source-support 和 calibration 行动清单。

### 当前最好模型是谁？

当前最好部署路径不是无保护神经网络，而是：

```text
M3W-Neural v1 protected policy
with Stage37 / teacher safety floor
and baseline-family / row-level full-waypoint evidence
```

### 当前最诚实 verdict

```text
项目是否跑通：是
是否 true 3D：否
是否 foundation：否
是否 metric：否
是否 seconds-level：否
是否 Stage5C ready/executed：否
是否 SMC ready/enabled：否
是否有 SDD 成功：是
是否有 external t50 成功：是
是否有 protected full-waypoint source-level 成功：是
是否有 t100 稳定成功：否，当前是 blocker / diagnostic
当前 best deployable：Stage37/teacher-floor protected M3W-Neural v1 policy
当前研究定位：protected dataset-local raw-frame 2.5D multi-agent world-state candidate
```

## 9. 主要验证命令记录

最近累计通过的关键验证包括：

```text
python3 run_stage37_cross_domain_eval.py = pass
python3 run_stage37_gates.py = pass
python3 run_stage42_unified_ablation_evidence.py = pass
python3 run_stage42_paper_claim_evidence_audit.py = pass
python3 run_stage42_retrained_ablation.py = pass
python3 run_stage42_source_level_full_waypoint_eval.py = pass
python3 run_stage42_source_level_ablation.py = pass
python3 run_stage42_source_level_incremental_ablation.py = pass
python3 run_stage42_source_level_residual_context.py = pass
python3 run_stage42_source_level_neural_context.py = pass
python3 run_stage42_source_level_sequence_context.py = pass
python3 run_stage42_source_level_graph_context.py = pass
python3 run_stage42_source_level_baseline_family_mechanism.py = pass
python3 run_stage42_ucy_validation_support_repair.py = pass
python3 run_stage42_repaired_protocol_robustness.py = pass
python3 run_stage42_aw_t100_easy_safety_repair.py = pass
python3 run_stage42_ay_shadow_holdout_robustness.py = pass
python3 run_stage42_t100_source_cv_repair.py = pass
python3 run_stage42_t100_data_gap_audit.py = pass
python3 -m pytest tests/test_stage42_t100_data_gap_audit.py = 4 passed
python3 -m pytest tests = 426 passed
```

## 10. 总结成一句话

```text
M3W 现在最有价值的成果，不是“已经做成 true 3D/foundation 神经世界模型”，而是严谨地证明了：
在真实 top-down 多智能体 raw-frame 数据上，只有把因果 baseline family、past-only history、目标原型、gain/harm/easy guard、validation-only policy 和 safety floor 结合起来，才能获得可部署的 external t50 / full-waypoint 正迁移；同时，JEPA、无保护 neural、plain domain alignment 和不稳健 t100 claim 都必须被明确限制。
```
