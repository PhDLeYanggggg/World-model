# M3W 目标内工作总账：尝试路线、失败原因、成功证据与当前质量

更新时间：2026-05-27  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README 与 `research_state.json`；最新已纳入 Stage42-EL/EM/EN/EO/EP/EQ/ER 的 `fresh_run` 审计与 claim-refresh 结果。  
用途：这是给用户阅读的统一单文件中文总账，回答“这个长期目标里到底做了什么、试过什么、哪些失败、为什么失败、哪些成功、当前大概是什么质量”。它不是新训练结果，也不把 cached 结果写成 fresh。

## 0. 最短结论

M3W 已经从早期的 SDD-only 2.5D trajectory scaffold，推进到一个有 SDD 和 external top-down dataset-local raw-frame 证据的 protected multi-agent world-state candidate。

但当前仍然不是：

- true 3D world model
- large-scale foundation world model
- global metric predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative model
- SMC-ready model

当前最诚实定位：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

当前最强部署/报告分层：

| 用途 | 当前最强结果 | 状态 |
| --- | --- | --- |
| SDD pixel raw-frame official benchmark | Stage26 cost-aware selector | SDD t+50 和 hard/failure 正提升；仍是 pixel raw-frame，不是 metric。 |
| External t+50 selector | Stage37 history + goal-prototype safe selector | external all/t50/hard/easy 同时过 gate，是 external selector best deployable。 |
| Protected neural/world-state candidate | M3W-Neural v1 / Stage41-42 protected policy family | 有 protected neural/full-waypoint/runtime evidence，但仍依赖 Stage37/teacher safety floor。 |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware composer guard | 用一部分 ADE 增益换 near-collision 安全修复。 |
| Source-level runtime policy | Stage42-DL/DQ group-consistency full-waypoint runtime | runtime exact replay 通过，source-level protected full-waypoint 证据成立。 |
| Paper package claim | Stage42-EG/EL/EM/EN/EO/EP/EQ/ER 后的受限 claim | 只能写 protected source-level group-consistency full-waypoint 2.5D evidence；source conversion、floor-free neural、当前 shallow sequence/graph context main claim 仍 blocked。 |

## 1. 永久边界

所有阶段都必须遵守：

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
- 不用 test endpoints 构建 goals。
- 不用 test metrics 调 threshold。
- 无保护 neural dynamics 不部署。

## 2. 做过的路线和结果

### 2.1 BPSG-MA / early 2.5D scaffold

做了：

- per-agent multi-agent trajectory world-state scaffold。
- causal baseline fallback。
- failure diagnostics。
- early selector / goal / correction heads。

结果：

- 成功形成可运行、可审计、可 fallback 的稳定基座。
- 但它不是 true 3D，也不是 foundation world model。

结论：

```text
BPSG-MA v1 是稳定基座，不是终点。
部署策略长期是 strongest causal baseline fallback + diagnostics，直到后续 selector/floor 过 gate。
```

### 2.2 Stage18 / Stage19 JEPA 与 WAM-style data

做了：

- SAM-JEPA-2.5D representation pretraining。
- WAM-style data registry。
- 区分 simulation、real top-down trajectory、human/egocentric video 的数据角色。

结果：

- JEPA non-collapse。
- 但 selector、failure predictor、goal predictor、hard/failure correction、official t+50 没有 downstream lift。

失败原因：

- non-collapse 只说明 latent 没坍塌，不说明 latent 对部署头有用。
- JEPA target 和 cost-aware selector / failure / correction 目标错位。
- 当时缺真实 raw scene/video + trajectory + long-horizon top-down data。

后续处理：

- JEPA 保留为 auxiliary / diagnostic。
- 不把 JEPA 写成生成式 world model。
- 不继续盲目加 JEPA，而转向数据策略、strong baseline、cost-aware selector。

### 2.3 Stage20 / Stage21 数据采集与 SDD 转换

做了：

- 审计 SDD、OpenTraj、ETH/UCY、TrajNet 等数据源。
- 明确 license / manual terms / local path / not downloaded / not converted 状态。
- 用户提供 SDD 后，转换成 SDD per-video world-state shards。

SDD 基础状态：

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
- raw data / large derived shards 未提交 Git。

### 2.4 Stage22-24 SDD benchmark、IO 修复与 hard-class selector 失败

做了：

- SDD scene packs、GoalBench、HardBench、BaselineFailureBench。
- SDD lazy episodes。
- strongest causal baselines。
- Stage23 quick-plus 后没有包装成 full medium。
- Stage24 修复 NPZ/shard I/O，建立 fast cache 和 true medium index。

Stage24 关键结果：

```text
I/O speedup: about 12.66x
true medium index:
  cross_scene: train 200k / val 50k / test 50k
  within_scene: train 200k / val 50k / test 50k
total indexed windows = 600000
no-leakage = pass
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
- best-baseline label 有 low-margin / ambiguity。
- selector 没有 cost/regret/easy-safety 约束。
- easy samples 被错误切换，导致不可部署。

### 2.5 Stage25 / Stage26 cost-aware selector 成功

做了：

- 从 hard best-baseline classification 改为 expected-FDE / regret-aware / confidence-gated policy。
- 加 conservative fallback。
- 用 Stage24 passed failure predictor 作为辅助。
- validation 选 threshold 和模型，test 只最终评一次。

结果：

```text
Stage26 selector:
  t+50 improvement: about +14.58%
  hard/failure improvement: about +11.23%
  easy degradation: about 1.81%
```

意义：

- Stage26 是 SDD pixel raw-frame 上的 best deployable selector。
- 它修复了 Stage24 的低 margin label 和 easy over-switch。
- 它仍然不是 metric / 3D / foundation。

### 2.6 Stage31 / Stage32 external zero-shot 和普通 domain alignment 失败

做了：

- 把 OpenTraj / ETH-UCY / TrajNet / UCY 等 external top-down pedestrian 数据转换到 M3W schema。
- 做 external strongest baseline、latent cache、zero-shot transfer、normalization、CORAL、latent adapter、mixed-domain selector。

结果：

```text
Stage31 external strongest baseline = constant_velocity_causal_fd
SDD -> external zero-shot:
  all improvement = -92.67%
  t50 improvement = -278.57%
external adapted selector:
  about 0 improvement
```

失败原因：

- SDD pixel-space 和 external dataset-local coordinates 不兼容。
- horizon definition、scene/goal availability、agent type、scale/homography 都不一致。
- latent adapter 缩小分布距离，但没有 predictive lift。

结论：

```text
普通 normalization / CORAL / latent adapter 不足以解决跨域。
必须做 row-level geometry、relative targets、train-only goals 和 selective transfer。
```

### 2.7 Stage33 / Stage34 row geometry 与 train-only goals：局部正信号但不可部署

做了：

- coordinate-invariant features。
- relative-error targets。
- external row geometry。
- train-only endpoint goals。
- external scene packs。
- domain-conditioned selector。

结果：

- t+50 / hard 有局部正信号。
- 但 all-test 为负，easy degradation 高。
- 最终只能 fallback 0.0。

失败原因：

- 外部 t+50 / hard 样本可学，但 easy 判别不足。
- selector 全量切换会伤害 easy。
- goal/scene context 不够可靠，held-out scene 下尤其弱。

### 2.8 Stage35 / Stage36 selective transfer 与 t+50 修复失败

Stage35 做了：

- external data expansion。
- held-out scenes。
- external hard/easy/failure labels。
- selective transfer policy。

结果：

```text
Stage35 selective transfer:
  all improvement = +12.13%
  hard/failure improvement = +13.98%
  easy degradation = 0.041%
  t+50 improvement = 0.0
```

Stage36 做了：

- t+50 forensics。
- horizon-specific selector。
- t+50 conservative policy search。
- t+50 curriculum adaptation。

结果：

- t+50 仍未修好。

失败原因：

- t+50 rows 足够，oracle headroom 也足够，但现有特征不足以支持安全切换。
- policy 不敢切，或切换样本不稳定。
- all-test objective 淹没 t+50 objective。
- 缺完整 past-only history window 和 scene-agnostic goal prototype。

### 2.9 Stage37 causal history + goal prototype 成功修复 external t+50

做了：

- past-only history windows：K=8/16/32/64。
- history speed、acceleration、heading、curvature、stop/go、dwell、path length、neighbor density、TTC 等。
- scene-agnostic goal prototypes：straight_continue、slow_stop、left_turn、right_turn、group_follow、density_avoid 等。
- t50 failure/gain/harm predictors。
- conformal safety / conservative fallback。

结果：

```text
all improvement: +13.48%
t+50 improvement: +8.46%
t+50 bootstrap CI: [+7.69%, +9.15%]
hard/failure improvement: +15.54%
easy degradation: 0.041%
gates: 16 / 16
verdict: stage37_t50_transfer_repaired_deployable
```

意义：

- Stage37 是 external t50 selector 的第一条可部署成功线。
- 成功机制是 causal history + goal prototype + switchability + conservative fallback。
- 仍然是 dataset-local raw-frame，不是 metric/seconds。

### 2.10 Stage38 bounded correction 失败，不部署

做了：

- 冻结 Stage37 policy。
- 训练 bounded delta / correction head。
- 比较 Stage35、Stage37、Stage38 correction with/without fallback。

结果：

- Stage38 correction 没有安全稳定超过 Stage37。
- 当前 external best 仍是 Stage37 selector。

失败原因：

- Stage37 selected baseline 已经很强。
- residual / correction 很容易伤 easy。
- bounded 后提升不足，不值得部署。

### 2.11 Stage39 / Stage40 neural dynamics：训练了，但没有超过 Stage37

做了：

- Causal Transformer。
- JEPA auxiliary。
- Hybrid。
- teacher distillation。
- gain/harm/failure/long-horizon drift targets。
- Stage37-protected fallback evaluation。

结果：

- 神经网络确实训练了。
- 但 neural with fallback 没有稳定超过 Stage37。
- neural without fallback 灾难性失败。

失败原因：

- 无保护 neural 容易伤 easy。
- JEPA non-collapse 仍不等于 downstream lift。
- Transformer/Hybrid 容易复制 Stage37 或被 fallback 吃掉。
- 轨迹 dynamics gain 很小，但安全成本高。

结论：

```text
Stage37 selector 仍是当时 external best deployable。
神经网络路线不能包装成成功，必须换 protected / bounded / teacher-floor 机制。
```

### 2.12 Stage41 protected neural breakthrough

做了：

- composite-tail safe-switch bounded neural dynamics。
- teacher safety floor。
- candidate distillation。
- t50/hard curriculum。
- multi-seed 和 bootstrap。
- pure UCY source-heldout。

结果：

```text
M3W-Neural v1 / Stage41 protected candidate:
  gates: 41 / 41
  all ADE improvement vs Stage37 floor: +21.03%
  t+50 ADE improvement: +13.65%
  t+100 raw-frame diagnostic ADE improvement: +14.69%
  hard/failure ADE improvement: +20.38%
  easy degradation: 0.00%
  positive external domains: 3
  bootstrap evidence: pass
  multiseed replication: pass
  pure UCY source-heldout: pass
```

意义：

- 这是当前 protected neural/world-state candidate 的核心成功。
- 成功成立在 Stage37/teacher safety floor 保护下。
- 不是 floor-free neural deployment。
- JEPA deployable path 被关闭，保留 diagnostic。

### 2.13 Stage42 long research：full-waypoint、runtime、proximity、安全地板、claim 边界

Stage42 做了很多细化工作，核心是把 Stage41 的 candidate 推到更像 paper evidence package 的状态，同时不越界。

#### Stage42-CQ / CR：proximity guard

结果：

```text
no proximity guard:
  all +3.02%
  t50 +1.50%
  t100 raw +6.12%
  hard +3.28%
  near@0.05 worsens +0.34%

proximity guard:
  all +1.77%
  t50 +1.07%
  t100 raw +3.48%
  hard +1.93%
  near@0.05 improves -0.06%
```

结论：

- no-guard 是 accuracy-priority diagnostic。
- proximity guard 是 safety-sensitive policy。

#### Stage42-DL / DQ：group-consistency full-waypoint runtime

结果：

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
```

意义：

- frozen policy 可通过 runtime API 精确复现。
- 支持 protected source-level group-consistency full-waypoint runtime policy。
- 不支持 ungated/global primary full-waypoint replacement。

#### Stage42-EL：context gain router

做了：

- 不再重复 residual trajectory prediction。
- 改为 deployment-aligned context gain/harm router。
- 测试 history、motion-goal、baseline+history+goal+neighbor 是否能安全提供 context increment。

结果：

```text
best router = baseline_plus_history_goal_neighbor
all delta = +0.000278
t50 delta = -0.000019
hard/failure delta = +0.000321
positive_context_gain_routers = []
gate = 10 / 10
verdict = context gain router pass, but context increment not supported
```

结论：

- 当前 protocol 下，scene/goal/neighbor/context 仍不能写成独立主贡献。
- 它们可能未来有用，但当前证据不足。

#### Stage42-EM：official source link audit

做了：

- 记录 UCY、ETH/BIWI、TrajNet++、OpenTraj 等官方/工具链来源。
- 只做 source-link / legal blocker 审计，不下载、不转换、不训练。

结果：

```text
targets audited = 5
official/toolkit source candidates = 4
manual terms required = 5
auto_download_allowed_now = 0
conversion_ready_now = 0
converted/evaluated = 0 / 0
estimated t50/t100 after terms = 10060 / 5696
gate = 14 / 14
```

结论：

- local parseability / GitHub toolkit license 不等于数据使用许可。
- source expansion 必须等用户确认 official terms、allowed use、local path、source identity。

#### Stage42-EN：floor removability decision map

做了：

- 汇总 safety-floor、t50 floor relaxability、proximity guard、source legal blocker。
- 明确哪些 floor 可放松、哪些不可移除。

结果：

```text
floor_free_neural_deployable = false
global_floor_removal_allowed = false
teacher_floor_rollout_context_removal_allowed = false
safe_partial_floor_relaxation_available = true
proximity_guard_required_for_safety_claim = true
gate = 13 / 13
```

关键决策：

| component | decision |
| --- | --- |
| ungated neural endpoint/full-waypoint | blocked，easy degradation 124.59% > 2% |
| teacher/floor rollout context | required |
| deployment fallback floor | required globally，允许 validation-backed t50 slice partial relaxation |
| proximity guard | safety-sensitive reporting required |
| source expansion without terms | blocked |
| TrajNet\|50 / UCY\|50 t50 relaxation | partial supported, slice-only |

## 3. 失败路线总表

| 失败路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| JEPA-only | non-collapse 但无 downstream lift | 表征目标和部署目标错位 | auxiliary/diagnostic，不写主贡献 |
| Stage24 hard-class selector | t50 -43.3%，easy 11.33% | hard labels 低 margin + 过度切换 | 改成 expected-FDE/regret/fallback |
| SDD->external zero-shot | all -92.67%，t50 -278.57% | coordinate/horizon/scale/goal/domain gap | row geometry + relative target + selective transfer |
| 普通 normalization/CORAL | latent gap 降但无 predictive lift | distribution alignment 不等于 decision alignment | 改成 causal row-level policy |
| Stage35/36 t50 | all/hard 正但 t50=0 | history/goal/switchability 不足 | Stage37 history+goal prototype 修复 |
| bounded correction | 没稳定超过 Stage37 | residual 易伤 easy，bounded 后收益小 | 不部署 |
| ungated Transformer/Hybrid | easy 不安全 | floor-free neural 不可控 | 所有 neural 必须过 Stage37/teacher floor |
| goal/scene gated expert | 低于 baseline-family control | proxy context 不够强 | diagnostic，不写独立 claim |
| neighbor/interaction gated expert | 低于 baseline-family control | hand-built kNN/graph 特征不够 | diagnostic，等待更强 graph/scene protocol |
| sequence/graph residual context | 不能超过 baseline-family first-stage | residual target 未抽出独立 context value | 当前 protocol 关闭 |
| t100 global claim | 局部正但 source/easy 不稳 | independent t100 source 不足 | t100 保持 raw-frame diagnostic |
| metric/seconds claim | evidence 不全局闭环 | H/FPS/stride/scale/source terms 不完整 | 禁止 global metric/seconds claim |
| floor-free neural | easy harm 过大 | ungated endpoint/full-waypoint unsafe | blocked |

## 4. 成功路线总表

| 成功路线 | 关键结果 | 为什么重要 |
| --- | --- | --- |
| Stage26 cost-aware selector | SDD t50 +14.58%，hard +11.23%，easy 1.81% | 证明 cost-aware/fallback-safe selector 比 hard-class selector 可部署。 |
| Stage37 t50 repair | external all +13.48%，t50 +8.46%，t50 CI [+7.69%, +9.15%]，hard +15.54%，easy 0.041% | 第一次 external t50/all/hard/easy 同时过 gate。 |
| Stage41 protected neural | all ADE +21.03%，t50 +13.65%，hard +20.38%，easy 0 | 证明 neural 在 Stage37/teacher floor 保护下可提供正贡献。 |
| Stage42-CQ proximity guard | near@0.05 从 caveat 修成 -0.06%，仍保留 all/t50/t100/hard 正增益 | 证明 safety-sensitive policy 不是单纯追求 ADE 最大。 |
| Stage42-DL/DQ runtime replay | rows 47458，exact replay，all +24.72%，t50 +22.36%，hard +23.89% | 支持 source-level protected group-consistency full-waypoint runtime policy。 |
| Stage42-EL gain router | gate 10/10，明确 context increment not supported | 负结果也有意义：收窄 claim，不再虚夸 context。 |
| Stage42-EM source audit | official links recorded，conversion_ready_now=0 | 防止把 local parseability / mirror 当合法接入。 |
| Stage42-EN floor map | global floor removal blocked，partial t50 relaxation allowed | 明确可部署边界：floor 不能全局移除。 |

## 5. 当前模型大概是什么质量

如果按研究成熟度分层：

```text
工程可复现性：较强
SDD pixel raw-frame selector：较强
external dataset-local t50 selector：较强
protected neural/world-state evidence：中强
full-waypoint runtime evidence：中强
scene/goal/interaction independent claim：弱/diagnostic
JEPA independent contribution：弱/negative
ungated neural deployment：失败
metric/seconds-level claim：未满足
true 3D / foundation claim：未满足
```

一句话：

```text
M3W 当前是一个强的 protected 2.5D multi-agent world-state candidate，
已经有 SDD 和 external raw-frame 正证据，
但还不是 true 3D / metric / foundation world model，
也不能脱离 Stage37/teacher safety floor 部署神经动力学。
```

## 6. 当前 best deployable 是什么

当前不应该说“一个无保护 neural model 已经成功”。更准确是分层 deployment：

```text
SDD:
  Stage26 cost-aware selector

External t50 selector:
  Stage37 history + goal prototype + switchability + conservative fallback

Protected neural / world-state:
  M3W-Neural v1 / Stage41-42 protected policy family
  with Stage37/teacher safety floor

Safety-sensitive full-waypoint/bridge:
  Stage42-CQ proximity-aware guarded composer

Source-level runtime policy:
  Stage42-DL/DQ group-consistency full-waypoint runtime policy
```

关键点：

- Stage37 / teacher floor 仍是核心安全地板。
- global floor-free neural 不可部署。
- partial t50 slice relaxation 只允许在 validation-backed slices 中使用。
- proximity guard 在 safety-sensitive claim 中必须保留。

## 7. 当前最短下一步

最短有效路径不是“继续堆模型”，而是：

1. **关闭 source legal / source diversity blocker**  
   按 Stage42-EM 记录的 official links 和 intake 模板，让用户确认 official terms、allowed use、local path、source identity。没有这个，不能把新 external source 写成 converted/evaluated。

2. **补 independent external top-down sources**  
   当前 broad source-level generalization 仍受 source concentration / terms blocker 限制。需要 legal new top-down pedestrian/drone sources，再做 conversion、no-leakage、source-CV、final test。

3. **减少但不硬移除 safety floor**  
   Stage42-EN 说明 global floor removal blocked。下一步只能做 slice-level / validation-backed floor relaxation，不能全局移除。

4. **改进 context/scene/interaction 的真实贡献**  
   Stage42-EL 仍未证明 context 独立增益。下一步若要把 context 写成主贡献，需要更强 scene/graph model 或新的数据，不是重复当前 residual/gain-router protocol。

5. **保持 t100 raw-frame diagnostic 边界**  
   t100 不是秒级，也不是 global deployable claim。要改变这个，需要更多 independent t100 sources 和 time/geometry calibration。

## 8. 不能写成论文主 claim 的内容

当前绝对不能写：

- M3W 是 true 3D world model。
- M3W 是 foundation world model。
- SDD/external 已经是 metric/seconds-level long-horizon world model。
- JEPA 是生成式 world model。
- Stage5C 已执行或 ready。
- SMC 已启用或 ready。
- ungated neural dynamics 可部署。
- goal/scene/neighbor/interaction 已经是独立主贡献。
- source expansion 已经合法转换完成。
- t100 已经是全局部署成功。

当前可以写但要加边界：

- protected source-level group-consistency full-waypoint raw-frame 2.5D evidence。
- Stage37/teacher-floor protected neural policy family。
- SDD pixel raw-frame selector success。
- external dataset-local t50 selector success。
- proximity-aware safety guard tradeoff。
- source legal / metric / seconds / floor-free neural 仍是 gap。

## 9. Git / 数据安全状态

已遵守的安全原则：

- 不提交 SDD raw data。
- 不提交 external raw data。
- 不提交 fast cache、feature store、latent cache、history cache、大 episodes。
- 不提交 checkpoint 大文件。
- 不提交 videos/images/third-party data。
- 不提交 `.venv-pytorch`。

本 README 是轻量文档，可以提交。

<!-- STAGE42_EO_POST_EM_EN_PAPER_REFRESH:START -->
## Stage42-EO Post-EM/EN Paper Package Refresh

- source: `fresh_paper_refresh_from_stage42_eg_em_en`
- role: propagate official-source/manual-terms blockers and floor-removability decisions into the paper package.
- This is a paper-package refresh, not new training, conversion, download, or threshold tuning.

### Source / Legal Boundary

- official/toolkit source candidates: `4` / `5`.
- manual terms required targets: `5`.
- auto_download_allowed_now: `0`; conversion_ready_now: `0`; converted/evaluated now: `0` / `0`.
- after-terms potential t50/t100 windows: `10060` / `5696`.
- Official links are not license acceptance; user must confirm terms, allowed use, local path, and source identity before conversion.

### Safety Floor Boundary

- floor_free_neural_deployable: `False`.
- global_floor_removal_allowed: `False`.
- teacher_floor_rollout_context_removal_allowed: `False`.
- safe_partial_floor_relaxation_available: `True` on `['t50_slice_relaxation::TrajNet|50', 't50_slice_relaxation::UCY|50']`.
- proximity_guard_required_for_safety_claim: `True`.

### Updated Paper Claim Boundary

- Supported: protected source-level group-consistency full-waypoint raw-frame 2.5D evidence.
- Supported only as narrow slice evidence: validation-backed t50 floor relaxation on mapped slices.
- Required: Stage37/teacher floor rollout context, deployment fallback floor, and proximity guard for safety-sensitive reporting.
- Blocked: source conversion without user terms/path/source identity; global floor-free neural; teacher-floor rollout context removal.
- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.
<!-- STAGE42_EO_POST_EM_EN_PAPER_REFRESH:END -->

<!-- STAGE42_EP_DEPLOYMENT_CONTRACT_GUARD:START -->
## Stage42-EP Deployment Contract Guard

- source: `fresh_stage42_deployment_contract_guard`
- verdict: `stage42_ep_deployment_contract_guard_pass`
- gates: `16 / 16`
- role: machine-readable guard for deployment and paper-claim requests after Stage42-DN/EM/EN/EO.
- safety_sensitive_default: `proximity_guard`.
- source_level_runtime_candidate: `group_consistency_full_waypoint_runtime`.
- allowed only as diagnostic: `no_proximity_guard` accuracy-priority reporting.
- blocked: global floor-free neural deployment, teacher-floor rollout context removal, source conversion without user terms, metric/seconds/foundation claims, Stage5C execution, and SMC.
- unknown future policy requests are denied by default until explicitly added to the contract.
<!-- STAGE42_EP_DEPLOYMENT_CONTRACT_GUARD:END -->

<!-- STAGE42_EQ_SEQUENCE_GRAPH_CONTEXT_ROUTER:START -->
## Stage42-EQ Sequence+Graph Context Router

- source: `fresh_stage42_sequence_graph_context_router`
- role: tests whether past-only sequence summary + current-frame graph summary can improve context gain routing over baseline-family protected control.
- gate: `12 / 12`; verdict `stage42_eq_sequence_graph_context_router_pass`.
- positive_sequence_graph_context_routers: `[]`; best router `baseline_plus_history_goal_neighbor`.
- best all/t50/t100raw/hard delta vs baseline-family: `0.000118` / `-0.000197` / `0.000083` / `0.000169`; easy `-0.001971`.
- sequence_graph_increment_verdict: `stage42_eq_sequence_graph_context_router_not_supported`.
- Boundary: fresh router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EQ_SEQUENCE_GRAPH_CONTEXT_ROUTER:END -->

<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:START -->
## Stage42-ER Post-EQ Context Claim Refresh

- source: `fresh_post_eq_context_claim_refresh`
- role: updates paper/action boundaries after the fresh Stage42-EQ sequence+graph router result.
- gate: `14 / 14`; verdict `stage42_er_post_eq_context_claim_refresh_pass`.
- Stage42-EQ best all/t50/t100raw/hard delta: `0.01%` / `-0.02%` / `0.01%` / `0.02%`.
- context decision: `close_current_shallow_sequence_graph_context_protocol`; independent context main claim allowed `False`.
- DA-2 is closed negative under the current shallow sequence/graph residual/router protocols.
- New priority: source/legal/time conversion plus stronger joint occupancy or interaction-constraint targets.
- Boundary: raw-frame/dataset-local 2.5D only; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:END -->

<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:START -->
## Stage42-ES Interaction / Occupancy Target Selection

- source: `fresh_stage42_interaction_occupancy_target_selection`
- role: fresh-reruns DH scalar proximity/occupancy target and DI explicit group-consistency target to choose the next interaction/occupancy training route.
- gate: `17 / 17`; verdict `stage42_es_interaction_occupancy_target_selection_pass`.
- selected target family: `explicit_group_consistency_repair`; decision `continue_with_explicit_group_consistency_interaction_target`.
- selected group-consistency all/t50/t100raw/hard/easy: `24.72%` / `22.36%` / `14.35%` / `23.89%` / `-25.63%`.
- near@0.05 base/final: `1.94%` / `1.38%`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:END -->
