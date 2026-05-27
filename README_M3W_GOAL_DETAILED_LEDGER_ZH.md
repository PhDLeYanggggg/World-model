# M3W 长期目标详细总账：路线、失败原因、成功证据与当前质量

更新时间：2026-05-27
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总既有报告、gate、README、`research_state.json`。本文件是总结，不是新训练、不新调参、不把 cached 结果写成 fresh。
最新纳入证据：Stage26、Stage37、Stage38、Stage39、Stage40、Stage41、Stage42-DY/DZ/EA/EB/EC/ED。

## 0. 总结先说结论

M3W 现在已经不是最早的 demo，也不是单纯的 baseline 表格。它已经形成了一条可复现的 protected 2.5D multi-agent world-state 研究轨道：

```text
当前定位：
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate

当前不能声称：
true 3D world model
large-scale foundation world model
global metric predictor
seconds-level long-horizon predictor
ungated neural dynamics deployable model
Stage5C latent generative model
SMC-ready model
```

当前最强可部署/可报告证据分层：

| 场景 | 当前最强 | 关键结果 | 是否可部署/可主张 |
| --- | --- | --- | --- |
| SDD pixel raw-frame official benchmark | Stage26 cost-aware selector | t+50 约 `+14.58%`，hard/failure 约 `+11.23%`，easy degradation 约 `1.81%` | SDD 内可作为 best deployable selector，但不是 metric/3D |
| External dataset-local t+50 transfer | Stage37 history + goal-prototype safe selector | all `+13.48%`，t50 `+8.46%`，t50 CI `[+7.69%, +9.15%]`，hard `+15.54%`，easy `0.041%` | external t50 修复成功，可作为 dataset-local deployable selector |
| Protected neural/world-state candidate | M3W-Neural v1 / Stage41 protected policy family | all ADE `+21.03%`，t50 ADE `+13.65%`，t100 raw diagnostic `+14.69%`，hard `+20.38%`，easy `0.00%` | 有 protected neural contribution，但依赖 safety floor，不是 ungated neural |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware composer guard | all `+1.77%`，t50 `+1.07%`，t100 raw `+3.48%`，hard `+1.93%`，near@0.05 改善 | 安全优先策略成立，但牺牲部分准确率 |
| Source-level full-waypoint runtime policy | Stage42-DL/DQ group-consistency full-waypoint runtime | rows `47458`，all `+24.72%`，t50 `+22.36%`，t100 raw `+14.35%`，hard `+23.89%`，exact replay pass | source-level protected runtime 证据成立，不是 global ungated replacement |
| 论文证据包 | Stage42-EB/EC/ED | group-consistency 贡献可写；source conversion legal blocker 仍在 | 可以写 protected 2.5D 候选论文包，不能写 foundation/metric/Stage5C |

## 1. 永久边界和不能越线的说法

下面这些边界贯穿所有阶段：

- SDD 是 `pixel-space benchmark`，不是 metric benchmark。
- External top-down 数据是 `dataset-local / unverified weak-metric diagnostic`，不是统一真实世界米制。
- t+50 / t+100 是 `raw-frame horizon`，不能写成秒级 long horizon。
- homography、metric scale、effective seconds 没有全局闭环验证。
- self-audited / visual-prior / auto-silver 标签不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- future endpoint / future waypoint 只能作为 supervised label 或 evaluation label，不能作为 inference input。
- 不使用 central velocity official input。
- 不用 test endpoints 构建 goals。
- 不用 test metrics 调 threshold。
- 无保护 neural dynamics 不部署。

## 2. 做过哪些路线

### 2.1 早期 BPSG-MA / 2.5D scaffold

做了：

- per-agent multi-agent trajectory world-state scaffold。
- causal baseline fallback。
- failure diagnostics。
- basic selector / goal / correction heads。

结果：

- 成功形成可运行、可审计、可 fallback 的稳定基座。
- 但它不是 true 3D，也不是 foundation world model。

结论：

```text
BPSG-MA v1 是稳定基座，不是终点。
部署策略长期都是 strongest causal fallback + diagnostics，直到后续 selector/floor 过 gate。
```

### 2.2 Stage18 / Stage19 JEPA 与 WAM-style data

做了：

- SAM-JEPA-2.5D representation pretraining。
- WAM-style data registry。
- 区分 simulation data、top-down trajectory、human/egocentric video 的不同用途。

结果：

- JEPA non-collapse。
- 但 selector、failure predictor、goal predictor、hard/failure correction、official t+50 都没有 downstream lift。

失败原因：

- non-collapse 只说明 latent 没坍塌，不说明 latent 对部署头有用。
- JEPA target 和 cost-aware selector / failure / correction 目标错位。
- 当时缺真实 raw scene/video + trajectory + long-horizon top-down data。

后续处理：

- JEPA 保留为 auxiliary / diagnostic。
- 不把 JEPA 写成生成式 world model。
- 不继续盲目加 JEPA，而转向数据策略、强 baseline、selector cost。

### 2.3 Stage20 / Stage21 数据采集与 SDD 转换

做了：

- 联网搜索并审计 SDD、OpenTraj、ETH/UCY、TrajNet 等数据源。
- 明确 license / manual terms / local path / not downloaded / not converted 的状态。
- 用户提供 SDD archive 后，转换成 SDD per-video world-state shards。

关键 SDD 状态：

```text
scenes = 8
videos = 60
tracks = 10300
world-state rows = 10616256
raw-frame t+50 samples = 10009005
raw-frame t+100 samples = 9497463
split = train 40 videos / val 4 videos / test 16 videos
no-leakage = pass
velocity = causal finite difference
coordinate status = pixel-space
metric status = no verified homography / scale
```

结果：

- SDD 成为第一个 official pixel-space top-down benchmark。
- raw data / large derived shards 没提交 Git。

### 2.4 Stage22 / Stage23 / Stage24 SDD benchmark 与 IO 修复

做了：

- SDD scene packs、GoalBench、HardBench、BaselineFailureBench。
- SDD lazy episodes。
- strongest causal baselines。
- Stage23 quick-plus 后没有把 quick-plus 包装成 medium。
- Stage24 修复 NPZ/shard I/O，建 fast cache 和 true medium index。

Stage24 关键结果：

```text
I/O speedup: about 12.66x
true medium index:
  cross_scene: train 200k / val 50k / test 50k
  within_scene: train 200k / val 50k / test 50k
total indexed windows = 600000
no-leakage = pass
strongest baseline = damped_velocity / scene_clamped depending split/horizon
selector oracle headroom = 46.2%
```

失败点：

```text
Stage24 validation-selected hard-class selector:
  t+50 improvement = -43.3%
  easy degradation = 11.33%

Stage24 failure predictor:
  AUROC = 0.8715
  passed

Stage24 JEPA:
  non-collapse
  no downstream lift
```

失败原因：

- selector oracle headroom 大，但 hard classification 过度切换。
- best-baseline label 有低 margin/歧义。
- selector 没有 cost/regret/easy safety 约束。
- easy samples 被错误切换，导致部署不可接受。

### 2.5 Stage25 / Stage26 cost-aware selector

做了：

- 把 selector 任务从 hard classification 改成 expected-FDE / regret-aware / confidence-gated policy。
- 预测每个 candidate baseline 的 expected FDE / risk。
- 加 conservative fallback：低置信、低预测增益、easy、高 harm 风险都回退 strongest baseline。
- 使用 Stage24 passed failure predictor 作为辅助。

Stage26 成功结果：

```text
t+50 improvement: about +14.58%
hard/failure improvement: about +11.23%
easy degradation: about 1.81%
```

成功原因：

- 不再强制每个样本选择 oracle hard label。
- 避免 low-margin 样本过度切换。
- 通过 gain/harm/failure 和 fallback 保护 easy。

结论：

```text
Stage26 是 SDD pixel raw-frame 上当前最强 deployable selector。
```

### 2.6 Stage31 / Stage32 external zero-shot 与 domain alignment

做了：

- 构建 external feature store。
- external no-leakage。
- external strongest baselines。
- external latent cache。
- SDD->external zero-shot transfer。
- normalization、CORAL、feature whitening、linear adapter、mixed-domain selector。

结果：

```text
external strongest baseline = constant_velocity_causal_fd
zero-shot M3W-LAS external transfer failed:
  all improvement = -92.67%
  t50 improvement = -278.57%
adapted selector about 0 improvement
```

失败原因：

- coordinate incompatibility。
- scene/goal/interaction missing。
- agent type mismatch。
- scale/homography 影响大。
- horizon/step definition 不一致。

结论：

```text
不能把 SDD-only 写成 cross-domain success。
普通 normalization / latent alignment 不足以修复 transfer。
```

### 2.7 Stage33 / Stage34 row geometry 与 relative target

做了：

- coordinate-invariant features。
- relative-error targets。
- external scene packs / train-only goals。
- external row geometry。
- per-row goal distance / angle。
- horizon / split rebuild。

结果：

- 出现 t+50 / hard 局部正信号。
- 但 all-test 为负、easy degradation 高。
- 最终只能 fallback 0.0，不是 positive transfer。

失败原因：

- 目标和特征还不足以判断哪些外部样本可安全切换。
- held-out scene goal 不稳定。
- t+50 受 horizon / track-length / goal ambiguity 影响更强。

### 2.8 Stage35 / Stage36 selective transfer 与 t50 专项修复

Stage35 做了：

- external 数据扩容。
- hard/easy/failure 标签。
- selective transfer policy。
- external selector v3。

Stage35 结果：

```text
test rows = 66303
t+50 rows = 16263
t+100 rows = 10008
all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t+50 improvement = 0.0
```

Stage36 做了：

- t50 forensics。
- horizon-specific selector。
- t50-specific policy search。
- t50 curriculum。

Stage36 仍失败：

```text
t+50 improvement = 0.0
t+50 oracle headroom = about 22.98%
```

失败原因：

- 不是没有 t50 可学空间，而是已有特征不能支持安全切换。
- all-test objective 淹没 t50。
- t50 需要更完整的 past-only history 和 goal prototype。

### 2.9 Stage37 causal history + scene-agnostic goal prototypes

做了：

- 构建 K=8/16/32/64 past-only history windows。
- history_dx/dy、speed、accel、heading、curvature、turn、stop/go、dwell、path length。
- neighbor count、min neighbor distance、density、TTC、closing speed。
- scene-agnostic goal prototypes：straight_continue、slow_stop、left_turn、right_turn、u-turn、group_follow、density_avoid、exit-like direction。
- t50 failure / gain / harm predictors。
- conformal safe policy。

Stage37 成功结果：

```text
all improvement: +13.48%
t+50 improvement: +8.46%
t+50 bootstrap CI: [+7.69%, +9.15%]
hard/failure improvement: +15.54%
easy degradation: 0.041%
gates: 16 / 16
verdict: stage37_t50_transfer_repaired_deployable
```

成功原因：

- past-only history 给了长时程趋势。
- goal prototypes 给了 scene-agnostic route intent。
- gain/harm/failure 让 policy 只在安全高增益样本切换。
- conformal safety 控制 easy degradation。

结论：

```text
Stage37 是 external dataset-local t+50 transfer 的关键成功点。
```

### 2.10 Stage38 bounded correction

做了：

- 冻结 Stage37 deployable policy。
- 补 external multi-domain audit。
- 在 Stage37 保护下训练 bounded trajectory correction / dynamics head。

结果：

- correction 没有安全超过 Stage37。
- 不部署 correction。

失败原因：

- selected baseline / Stage37 已经很强。
- residual delta 容易伤 easy。
- bounded 后收益不足。

结论：

```text
Stage38 correction 是 diagnostic，不是 deployable。
```

### 2.11 Stage39 / Stage40 neural dynamics

做了：

- Causal Temporal Transformer。
- JEPA auxiliary。
- JEPA + Transformer Hybrid。
- teacher distillation。
- horizon-specific heads。
- gain/harm/failure multi-task loss。
- hard/failure oversampling。
- Stage37 fallback protection。

结果：

- Transformer / JEPA / Hybrid 都训练过。
- 但无保护 neural 没有超过 Stage37，甚至会灾难性失败。
- with fallback 后也没有稳定超过 Stage37，因此不部署。

失败原因：

- neural without fallback 会伤 easy。
- JEPA non-collapse 仍不代表 downstream lift。
- Hybrid 很容易复制 Stage37 或被 fallback 吃掉。
- 现有数据/特征下，Stage37 safety policy 是强 teacher/floor。

结论：

```text
不要把 Stage39/40 写成 neural world model 成功。
当前 best external deployable 仍是 Stage37。
```

### 2.12 Stage41 protected neural breakthrough

做了：

- composite-tail safe-switch bounded neural dynamics。
- teacher floor。
- multi-seed / bootstrap。
- source-heldout checks。
- external-domain robust candidate。

关键结果：

```text
M3W-Neural v1 gates: 41 / 41
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

成功原因：

- 不再让 neural 无保护替代 Stage37。
- neural 做 bounded / protected / tail-aware 增益。
- teacher floor 保护 easy。
- safe-switch 决定是否使用 neural。

结论：

```text
这是 protected neural/world-state candidate，不是 ungated neural dynamics。
```

### 2.13 Stage42 long research evidence package

Stage42 做了很多补证据和负结果关闭，核心分成几类：

1. paper package / reproducibility / model/data card。
2. full-waypoint dynamics / bridge-shape composer。
3. proximity guard。
4. source-level group-consistency。
5. context/goal/neighbor contribution audits。
6. source conversion / legal/time/metric blockers。

重要成功：

```text
Stage42-CQ proximity-aware composer guard:
  all +1.77%
  t50 +1.07%
  t100 raw +3.48%
  hard/failure +1.93%
  near_collision@0.05 improved vs endpoint-linear
  gates 19 / 19

Stage42-DL/DQ group-consistency full-waypoint runtime:
  rows = 47458
  switch exact match = true
  selected_xy max abs diff = 0.0
  selected ADE/FDE max abs diff = 0.0
  all +24.72%
  t50 +22.36%
  t100 raw +14.35%
  hard/failure +23.89%
  near@0.05 from 1.94% to 1.38%

Stage42-DZ / EA dual-domain group-consistency:
  UCY all/t50/hard = +35.58% / +22.72% / +33.78%
  TrajNet all/t50/hard = +32.07% / +28.18% / +31.29%
  global all/t50/hard CI lows = +32.56% / +26.53% / +31.51%
```

重要负结果：

```text
Stage42-DP:
  current sequence/graph residual context protocol closed.
  best context delta vs baseline-family:
    all -2.30%
    t50 -8.31%
    hard -2.62%

Stage42-EC:
  supported: explicit group-consistency full-waypoint source-level repair.
  blocked: scalar loss-family primary, current sequence/graph residual context,
           goal/scene main claim, neighbor/interaction main claim.

Stage42-ED:
  conversion_ready_now = 0
  conversion_allowed_now = 0
  converted/evaluated now = 0
  technical_ready_after_terms_targets = 2
  estimated t50/t100 windows after terms = 10060 / 5696
```

Stage42 总结：

```text
可以写：protected source-level group-consistency full-waypoint dynamics evidence.
不能写：global ungated full-waypoint replacement, foundation, true 3D, metric/seconds, Stage5C, SMC.
```

## 3. 失败路线总表

| 路线 | 失败表现 | 根因 | 处理 |
| --- | --- | --- | --- |
| JEPA-only / SAM-JEPA | non-collapse 但 downstream 无 lift | 表征目标和 selector/failure/correction 目标错位 | 保留辅助，不写主贡献 |
| Stage24 hard-class selector | t50 `-43.3%`，easy `11.33%` | low-margin labels、过度切换、calibration 差 | 改 Stage26 cost-aware selector |
| SDD -> external zero-shot | all `-92.67%`，t50 `-278.57%` | 坐标/horizon/goal/agent/scale 不兼容 | 做 row geometry、relative target、history |
| normalization / CORAL / latent adapter | 分布距离变小但预测无 lift | distribution alignment 不等于 decision alignment | 转向 selective transfer |
| Stage35/36 t50 | all/hard/easy 过，t50=0 | t50 缺 history/prototype/switchability | Stage37 修复 |
| ordinary residual / bounded correction | 不稳定超过 Stage37 | residual 容易伤 easy，收益不足 | 不部署，只 diagnostic |
| ungated Transformer / Hybrid | 会灾难性失败 | neural 无 safety floor 会伤 easy | 必须 protected |
| goal/scene gated expert | 没超过 baseline-family control | 当前 scene/goal proxy 不足 | 不写独立主贡献 |
| neighbor/interaction gated expert | 没超过 baseline-family control | 当前 kNN/hand-built graph 不够 | 不写独立主贡献 |
| current sequence/graph residual context | 低于 baseline-family control | residual target 没抽出独立 value | Stage42-DP 关闭当前 protocol |
| t100 global deployable | source-CV/easy 不稳 | independent t100 source support 不足 | 保持 diagnostic |
| metric/seconds claim | 未闭环 | H/FPS/stride/scale/legal/source 不完整 | 禁止 claim |

## 4. 成功路线总表

| 成功路线 | 关键机制 | 证据 | 结论 |
| --- | --- | --- | --- |
| Stage26 SDD cost-aware selector | expected-FDE + regret + confidence fallback | t50 `+14.58%`，hard `+11.23%`，easy `1.81%` | SDD best deployable selector |
| Stage37 external t50 repair | past history + goal prototype + gain/harm/failure + conformal safety | all `+13.48%`，t50 `+8.46%`，hard `+15.54%`，easy `0.041%` | external t50 deployable selector |
| Stage41 M3W-Neural v1 | protected bounded neural under teacher floor | all `+21.03%`，t50 `+13.65%`，hard `+20.38%`，easy `0.00%` | protected neural candidate |
| Stage42-CQ proximity guard | predicted rollout geometry guard | near@0.05 repaired while all/t50/hard positive | safety-sensitive composer |
| Stage42-DL/DQ runtime replay | frozen policy exact replay | selected_xy/ADE/FDE diff `0.0` | reproducible runtime policy |
| Stage42-DZ/EA dual-domain group consistency | explicit group-consistency source-level repair | UCY+TrajNet positive, bootstrap lows positive | source-level paper evidence |

## 5. 现在模型大概是什么质量

当前质量可以这样说：

```text
M3W 已经达到“protected 2.5D multi-agent world-state paper-candidate evidence package”的质量。
它有强 baseline、严格 no-leakage、external positive transfer、protected neural / full-waypoint / runtime replay / bootstrap 证据。
但它还没达到 true 3D、foundation、global metric/time、ungated neural world model 的质量。
```

更具体地：

- **工程质量**：较高。已有大量 runner、reports、gates、tests、runtime replay、frozen policy、hash/schema/README/state。
- **实验质量**：中高。SDD medium、external transfer、bootstrap、多阶段 ablation、negative result closure 都做了。
- **论文候选质量**：有 protected 2.5D world-state candidate 的论文包雏形；还不是完整 A刊/foundation claim。
- **部署质量**：只能部署 protected selector / guarded policy，不部署无保护 neural / Stage5C / SMC。
- **泛化质量**：external dataset-local 正迁移已出现，但 source/legal/time/metric closure 仍未完全完成。
- **物理世界质量**：仍是 raw-frame / dataset-local；缺 metric/time/homography 全闭环。

## 6. 当前 best deployable 是谁

分场景：

```text
SDD:
  Stage26 cost-aware selector

external t50 selector:
  Stage37 history + goal-prototype safe transfer policy

protected neural/world-state:
  M3W-Neural v1 / Stage41 protected policy family under Stage37/teacher floor

safety-sensitive bridge/shape:
  Stage42-CQ proximity-aware composer guard

source-level full-waypoint runtime:
  Stage42-DL/DQ group-consistency full-waypoint runtime policy
```

最保守总说法：

```text
current best deployable family =
Stage37 / teacher-floor protected M3W policy family,
with Stage42 source-level group-consistency full-waypoint runtime evidence as protected source-level candidate.
```

## 7. 现在最不能犯的错误

不能写：

1. M3W 是 true 3D world model。
2. M3W 是 foundation world model。
3. SDD/external 是统一 metric result。
4. raw-frame t50/t100 是 seconds-level horizon。
5. JEPA 是生成式 world model。
6. Transformer/Hybrid 无保护超过 Stage37。
7. goal/scene 或 neighbor/interaction 当前是独立主贡献。
8. t100 是 global deployable success。
9. Stage5C 已执行或 ready。
10. SMC 已启用或 ready。
11. source conversion ready now。
12. legal/path hint 等于合法 converted dataset。

## 8. 下一步最短路径

最值得做的三件事：

1. **source/legal/time closure**
   按 Stage42-ED 的 user-action unblocker，先让 UCY / ETH-BIWI 的 terms/source identity/path 变成 legally conversion-ready，再转换和 source-CV eval。不要绕过条款。

2. **把 group-consistency runtime package 固化成 reviewer-ready artifact**
   保留 frozen policy、schema hash、cache hash、runtime replay commands、pytest 状态，继续做 reviewer replay 和 minimal dependency path。

3. **如果继续做 neural/context，不要重复当前 residual protocol**
   Stage42-DP/EC 已经关闭当前 sequence/graph residual route。下一轮必须换 target 或架构，比如 explicit physical consistency / graph-neural interaction target / scene-rich source-level protocol，而不是继续标量 loss 或同一 residual。

## 9. 最终总判定

```text
项目是否跑通：是
是否有真实可复现的正结果：是
是否有 external positive transfer：是，Stage37 起成立
是否有 protected neural/world-state evidence：是，Stage41/42 有
是否有 source-level full-waypoint runtime evidence：是，Stage42-DL/DQ/DZ/EA 有
是否 true 3D：否
是否 foundation：否
是否 metric/seconds-level：否
是否 Stage5C executed：否
是否 SMC enabled：否
当前 best deployable：Stage37/teacher-floor protected M3W policy family；source-level runtime 用 Stage42 group-consistency full-waypoint policy
当前 verdict：protected dataset-local/raw-frame 2.5D multi-agent world-state candidate, with source-level full-waypoint and group-consistency evidence, but not a true 3D/foundation/metric world model
```
