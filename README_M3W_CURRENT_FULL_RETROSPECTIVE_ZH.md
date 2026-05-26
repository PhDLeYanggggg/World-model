# M3W 当前完整复盘：路线、失败原因、成功证据与当前结论

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总 Stage18-Stage42 已生成报告、gate、README 与 `research_state.json`；最近 Stage42-DC/DD 结果来自 fresh reports。  

这份 README 回答一个具体问题：在“训练一个真正强的真实世界多模态多智能体世界模型 M3W”这个长期目标内，已经做了什么、尝试了哪些路线、哪些失败了、为什么失败、哪些成功了、当前最强可部署模型是谁、还不能 claim 什么。

它不是宣传稿。所有 `not_run`、fallback-only、diagnostic-only、license/source blocked、metric/time blocked、source support 不足的结果，都不能写成成功。

## 0. 当前一句话结论

M3W 已经从一个 SDD pixel-space 轨迹 scaffold，推进到一个有外部 top-down dataset-local raw-frame 正迁移证据的 protected 2.5D multi-agent world-state candidate。

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
ungated neural dynamics
latent generative Stage5C execution
SMC-ready model
```

当前最强可部署分层：

| 用途 | 当前 best deployable / best evidence | 解释 |
| --- | --- | --- |
| SDD pixel-space | Stage26 cost-aware selector | SDD 内 t+50 与 hard/failure 有稳定提升，仍是 pixel raw-frame。 |
| External dataset-local selector floor | Stage37 safety-selected t+50 transfer policy | 第一次让 external all / t+50 / hard 同时为正且 easy 安全。 |
| Protected neural/world-state package | M3W-Neural v1 / Stage41-42 protected composer family | 在 Stage37/teacher floor 保护下有强证据，但不是无保护 neural，也不是 foundation。 |
| Safety-sensitive bridge/shape policy | Stage42-CQ proximity-aware guarded composer | 修复 proximity caveat，保留 all/t50/t100 raw-frame/hard 正提升。 |
| 当前最新 blocker | Stage42-DD source support closure | ETH_UCY、TrajNet、UCY 的 legal/source/time/t100 closure 仍未完全关闭。 |

## 1. 必须保留的边界

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据是 dataset-local / unverified weak-metric diagnostic，不是统一真实世界米制。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- homography、metric scale、effective seconds 没有全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- future endpoint / future waypoint 只能作为 supervised label 或 evaluation label，不能作为 inference input。
- 不使用 central velocity official input。
- 不用 test endpoints 建 goals。
- 不用 test metrics 调 threshold。
- 无保护 neural dynamics 不部署。

## 2. 大路线总览

| 阶段/路线 | 做了什么 | 结果 | 当前解释 |
| --- | --- | --- | --- |
| BPSG-MA / 早期 2.5D scaffold | 建 per-agent multi-agent world-state、causal baseline fallback、diagnostics。 | 成功作为稳定基座。 | 可运行、可审计，但不是 3D/foundation。 |
| Stage18/19 JEPA / WAM-style data | 做 SAM-JEPA-2.5D、WAM-style registry、simulation / top-down / ego video 分角色。 | JEPA non-collapse，但下游 lift 未证明。 | JEPA 目标和部署目标错位，不能作为主贡献。 |
| Stage20/21 SDD 数据 | 采集、解压、转换 SDD，建立 pixel raw-frame world-state。 | 成功。 | SDD 成为 official pixel-space benchmark。 |
| Stage22-24 SDD medium | scene packs、episodes、HardBench、GoalBench、fast cache、true medium index。 | benchmark 成立；Stage24 selector 失败。 | oracle headroom 大，但 hard-class selector 过度切 easy。 |
| Stage25/26 cost-aware selector | expected-FDE / regret / gain-harm / fallback-safe selector。 | SDD 成功。 | Stage26 成为 SDD best deployable。 |
| Stage31/32 external zero-shot/domain alignment | OpenTraj/ETH-UCY/TrajNet/UCY external transfer、normalization、latent adapter。 | 失败。 | 坐标、horizon、goal/scene、agent type、scale 不兼容。 |
| Stage33/34 coordinate-invariant / row geometry | relative features、relative target、train-only goals、scene packs。 | 局部正信号，不可部署。 | all/easy 不稳，t50/hard 有信号。 |
| Stage35 selective transfer | hard/easy/failure labels，选择性迁移策略。 | all/hard/easy 过，但 t50=0。 | 长时程安全切换缺少上下文。 |
| Stage36 t50 repair | horizon-specific selector、t50 policy search/curriculum。 | 仍失败。 | t50 有 22.98% oracle headroom，但特征不足。 |
| Stage37 history + goal prototypes | past-only history windows、scene-agnostic goal prototypes、switchability/gain/harm/conformal safety。 | 成功。 | external t50 被修复为可部署正迁移。 |
| Stage38 bounded correction | 在 Stage37 保护下训练 bounded correction/dynamics head。 | 不部署。 | correction 没有安全超过 Stage37。 |
| Stage39/40 Transformer/JEPA/Hybrid | 训练 Causal Transformer、JEPA auxiliary、Hybrid、多任务/teacher distillation。 | 诊断为主，没超过 Stage37。 | neural without fallback 不安全；with fallback 未稳定胜出。 |
| Stage41 protected neural breakthrough | self-gated endpoint / composite-tail / all-agent policy。 | 强 positive evidence。 | 证明 protected neural candidate 有用，但仍需 safety floor。 |
| Stage42 long research | full-waypoint、source-level、ablation、paper package、runtime replay、proximity guard、source closure。 | 证据包增强，边界更清楚。 | 形成 protected 2.5D paper candidate evidence，不是 foundation/metric。 |

## 3. 关键成功结果

### 3.1 Stage26：SDD cost-aware selector 成功

Stage24 直接 hard-class selector 失败后，Stage25/26 把任务改成 cost-aware / regret-minimizing / fallback-safe selection。

关键结果：

```text
SDD t+50 improvement: about +14.58%
SDD hard/failure improvement: about +11.23%
SDD easy degradation: about 1.81%
```

成功原因：

- 不再硬预测“哪个 baseline 最好”。
- 对每个 candidate baseline 预测 expected FDE/risk。
- 使用 confidence gate、predicted gain margin、easy guard、harm guard。
- failure predictor 作为辅助信号。
- 低置信度或低收益样本回退 strongest causal baseline。

边界：

- 这是 SDD pixel raw-frame 内成功。
- 不是 metric。
- 不是 3D。
- 不是 external cross-domain 成功。

### 3.2 Stage37：external t+50 transfer 被修复

Stage35/36 最大 blocker 是 external t+50 improvement = 0.0。Stage37 增加完整 past-only history windows 和 scene-agnostic goal prototypes 后修复。

关键结果：

```text
rows: 66,303
all improvement: +13.48%
t+50 improvement: +8.46%
t+50 bootstrap CI: [+7.69%, +9.15%]
hard/failure improvement: +15.54%
easy degradation: 0.041%
gates: 16 / 16
verdict: stage37_t50_transfer_repaired_deployable
```

成功原因：

- t+50 不缺 headroom，缺 causal context。
- 加入 K=8/16/32/64 past-only history。
- 加入 goal prototypes：straight_continue、slow_stop、left/right turn、group_follow、density_avoid 等。
- 训练 failure/gain/harm switchability，而不是只调 threshold。
- conformal / conservative safety rule 保护 easy。

边界：

- external 是 dataset-local / unverified weak metric diagnostic。
- t+100 仍没有安全正提升。
- 仍不是 true world dynamics；更像 safe selector-level world-state policy。

### 3.3 Stage41：protected neural candidate 形成强证据

Stage39/40 神经模型没有超过 Stage37 后，Stage41 通过 self-gated endpoint、candidate distillation、all-agent policy、composite-tail 等方式形成 protected neural candidate。

关键 Stage41 best result：

```text
best: fresh_self_gated_endpoint::binary_fde_neural_dynamics
rows: 55,528
all improvement: +41.96%
t+50 improvement: +40.62%
t+100 raw-frame diagnostic improvement: +45.73%
hard/failure improvement: +43.61%
easy degradation: 0.0%
domains positive: ETH_UCY, TrajNet, UCY
gates: 41 / 41
```

重要解释：

- 这是 protected neural candidate，不是无保护 rollout。
- 一些后续 locked / stricter protocols 会更保守，因此不能只拿最高单次数值当最终无条件部署 claim。
- Stage41 证明“神经候选 + safety floor”能产生明显正证据，但也暴露了 full-waypoint、source support、runtime replay、proximity safety 等后续需要补的缺口。

### 3.4 M3W-Neural v1 / Stage42 paper-package 证据

M3W-Neural v1 composite-tail safe-switch bounded neural dynamics 的 package 指标：

```text
all ADE improvement: +21.03%
t+50 ADE improvement: +13.65%
t+100 raw-frame diagnostic ADE improvement: +14.69%
hard/failure ADE improvement: +20.38%
easy degradation: 0.00%
gates: 41 / 41
```

bootstrap 支持：

```text
all CI: [+20.67%, +21.39%]
t+50 CI: [+13.06%, +14.26%]
t+100 raw-frame diagnostic CI: [+13.96%, +15.37%]
hard/failure CI: [+19.99%, +20.76%]
```

解释：

- 这是 strong protected 2.5D manuscript package。
- 不是 full A-journal-ready foundation result。
- 仍依赖 safety floor / teacher floor / protected deployment。

### 3.5 Stage42-CO/CP/CQ/CR：bridge/shape composer 与 proximity guard

Stage42 发现 endpoint-linear bridge 与 full-waypoint shape 有 tradeoff。最终做了 validation-only composer 和 proximity-aware guard。

Stage42-CO common-validation composer：

```text
selected full-waypoint slices: ETH_UCY|50 and ETH_UCY|100
test vs endpoint-linear ADE:
  all: +3.02%
  t50: +1.50%
  t100 raw diagnostic: +6.12%
  hard/failure: +3.28%
  easy degradation: +0.25%
gates: 14 / 14
```

Stage42-CP bootstrap：

```text
all CI: [+2.64%, +3.37%]
t50 CI: [+0.90%, +2.09%]
t100 raw diagnostic CI: [+5.39%, +6.94%]
hard/failure CI: [+2.90%, +3.68%]
near_collision@0.05 vs endpoint-linear: +0.34%
```

Stage42-CQ proximity-aware guard：

```text
all: +1.77%, CI [+1.50%, +2.05%]
t50: +1.07%, CI [+0.59%, +1.52%]
t100 raw diagnostic: +3.48%, CI [+2.91%, +4.08%]
hard/failure: +1.93%, CI [+1.63%, +2.22%]
easy degradation: +0.25%
near_collision@0.05 vs endpoint-linear: -0.06%
gates: 19 / 19
```

Stage42-CR Pareto audit：

```text
no proximity guard:
  all/t50/t100/hard = +3.02% / +1.50% / +6.12% / +3.28%
  near@0.05 = +0.34%

with proximity guard:
  all/t50/t100/hard = +1.77% / +1.07% / +3.48% / +1.93%
  near@0.05 = -0.06%
```

解释：

- no-guard 更准，但 proximity 风险更高。
- guarded 版本牺牲部分 ADE，换取 near-collision 不恶化。
- 安全敏感部署应使用 guarded composer。

### 3.6 Stage42-CV/CW/CX/CY/CZ：复现性和 paper-freeze

最新复现性链路：

```text
Stage42-CV batch runtime replay:
  25 / 25 gates
  frozen policy runtime replay exactly matches decisions and selected_xy/ADE/FDE

Stage42-CW paper refresh:
  runtime replay evidence propagated into paper/reproducibility/model-card package

Stage42-CX provenance verifier:
  21 artifacts audited
  21 gates passed

Stage42-CY worktree caveat classifier:
  Stage42 substantive dirty files = 0

Stage42-CZ paper freeze candidate manifest:
  74 files hashed
  14 / 14 gates
  freeze_status = candidate_clean
```

解释：

- 这不是新模型性能，而是证据链质量提升。
- 它证明 paper package 的关键文件、来源、运行命令、runtime replay 和 dirty-state caveat 更可审计。

## 4. 关键失败路线和失败原因

### 4.1 JEPA-only / JEPA主贡献失败

表现：

```text
JEPA non-collapse: yes
downstream lift: not proven
selector/failure/correction/t50 lift: no stable evidence
```

原因：

- non-collapse 只说明 latent 没塌，不说明对 deployment decision 有用。
- 当前核心部署目标是 gain/harm/easy-safe switching，不是单纯表征学习。
- JEPA latent 多次作为辅助加入 selector/failure probe，未稳定改善。

当前处理：

- JEPA 只能作为 auxiliary / diagnostic。
- 不能写成生成式 world model。
- 不能作为 Stage5C 执行依据。

### 4.2 Stage24 hard-class selector 失败

表现：

```text
oracle headroom: about 46.2%
trained selector t+50 improvement: -43.3%
easy degradation: 11.33%
```

原因：

- oracle best baseline 标签在 margin 很小的样本上不稳定。
- hard classification 让模型过度切换。
- 没有 cost-aware regret/harm/easy guard。
- 模型学到“切换”，没学到“什么时候不切”。

修复：

- Stage25/26 改成 expected-FDE / regret / confidence / fallback-safe selector。

### 4.3 Stage31/32 external zero-shot 和 domain alignment 失败

表现：

```text
Stage31 zero-shot external transfer:
  all improvement: strongly negative
  t50 improvement: strongly negative

Stage32 adapted selector:
  all/t50 improvement: 0.0
  gates: partial
```

原因：

- SDD 是 pixel-space，external 是 dataset-local 或 weak metric。
- horizon / sampling / scene split 不一致。
- external scene/goal/interaction 缺失。
- agent type schema 不一致。
- normalization / CORAL / latent adapter 只能缩小分布距离，不等于预测 gain/harm 有用。

修复：

- Stage33/34 建 coordinate-invariant features、relative error target、row geometry、train-only goals。
- Stage35/37 转向 selective transfer 和 causal history。

### 4.4 Stage35/36 t+50 失败

表现：

```text
Stage35:
  all improvement: +12.13%
  hard/failure: +13.98%
  easy degradation: 0.041%
  t+50 improvement: 0.0

Stage36:
  t50 rows: 16,263
  t50 oracle headroom: 22.98%
  t50 still: 0.0
```

原因：

- 不是没有样本，也不是没有 headroom。
- 问题是现有 feature/goal/context 无法支持安全切换。
- all-test objective 淹没 t50。
- policy 宁愿 fallback，不敢在 t50 切换。

修复：

- Stage37 加完整 past-only history windows、goal prototypes、switchability models、conformal safety。

### 4.5 Stage38 bounded correction 不部署

表现：

```text
Stage37 frozen:
  all +13.48%
  t50 +8.46%
  hard/failure +15.54%
  easy 0.041%

Stage38 correction decision:
  keep_stage37_selector
```

原因：

- bounded residual / correction 没有安全超过 Stage37。
- 直接修改 trajectory 容易伤 easy，clip 后收益不足。
- dynamics head 还没证明比 selector floor 更强。

当前处理：

- correction not deployable。
- Stage37 保持 external best floor。

### 4.6 Stage39/40 neural dynamics 没超过 Stage37

表现：

```text
Stage39 best neural with fallback == Stage37 same-subset
JEPA downstream lift negative
Stage40 neural_without_fallback:
  all improvement: -126.36%
  t50 improvement: -292.10%
  hard/failure improvement: -109.40%
  easy degradation: 612.31%
```

原因：

- raw endpoint/FDE loss 没教会 Stage37 的成功机制。
- neural model 没学会何时切换、何时回退、如何保护 easy。
- fallback gate 把不可靠 neural switches 吃掉。
- JEPA auxiliary 加了噪声，没有稳定 downstream lift。

修复方向：

- Stage41 改用 teacher/floor、self-gated endpoint、candidate distillation、gain/harm、all-agent policy。

### 4.7 Context / goal / scene / neighbor 独立主贡献失败

最近 Stage42-CJ/CK/DC 给出的结论：

```text
baseline-family control dominates.
goal/scene gated expert did not beat baseline-family.
neighbor/interaction gated expert did not establish independent main claim.
context switchability / gain-harm gate still not supported.
```

Stage42-DC selected `baseline_plus_knn_graph` 后：

```text
delta all vs baseline-family: +0.00037
delta t50 vs baseline-family: -0.00007
delta hard vs baseline-family: +0.00042
decision: context_switchability_not_supported
```

原因：

- 当前最强 signal 仍是 baseline-family rollout context。
- goal/scene/neighbor/graph 当前特征没有带来足够稳定的增量。
- 换成 gain/harm switchability supervision 也没有救回来。

结论：

- 不应继续重复同一套 context residual/gated protocol。
- 下一步要改变模型/数据/target，而不是继续小调。

### 4.8 Full-waypoint / shape deployment 仍有限制

Stage42-CM 说明：

```text
full_waypoint_minus_linear_bridge:
  all: -2.45%
  t50: +1.15%
  t100 raw diagnostic: +8.16%
  hard: -0.87%

ungated_full_waypoint_transformer:
  all: +29.66%
  t50: +21.52%
  t100 raw diagnostic: +35.92%
  hard: +32.94%
  easy degradation: +124.59%
  status: unsafe_not_deployable
```

原因：

- Full-waypoint 对 t50/t100 shape 有帮助，但 all-ADE 不如 endpoint-linear bridge。
- Ungated full-waypoint 很强但 easy 伤害巨大。
- Graph/group consistency positive，但 proximity caveat 存在。

当前状态：

- full-waypoint 可作为 auxiliary / guarded composer。
- 不能把 ungated full-waypoint 当 deployable world dynamics。

### 4.9 Source / legal / metric / t100 claim 仍阻塞

Stage42-DD 最新结论：

```text
domains_closed: []
domains_not_closed: ETH_UCY, TrajNet, UCY
restricted_source_specific_metric_time_candidate_exists: true
global_metric_seconds_claim_allowed: false
global_t100_deployable_claim_allowed: false
```

主要 blocker：

- ETH_UCY：source terms / conversion readiness missing；train-only t100 source-CV support missing；还需 2 个 independent t100-capable ETH_UCY sources。
- TrajNet：source terms / time calibration missing；train-only t100 source-CV missing；还需 1 个 independent t100-capable source。
- UCY：有 source-specific calibration candidates，但 legal/source confirmation 和 t100 source-CV closure 仍未完全关闭；还需 1 个 independent t100-capable UCY source 或 split。

结论：

- 不能写 global metric/seconds-level。
- 不能写 global deployable t100。
- local path / parseability 不等于 legal permission。

## 5. 当前最有意义的技术路线总结

真正有效的路线不是“更大模型”，而是：

```text
causal baseline-family rollout context
+ past-only history windows
+ validation-only gain/harm/easy-safe switch
+ source/horizon/domain-specific fallback
+ proximity-aware safety guard
+ bootstrap / source-level / runtime replay evidence
```

已经证明有效的设计原则：

- 先守住 strongest causal baseline 或 Stage37 floor。
- 只在 high-confidence / positive-gain / low-harm 时切换。
- easy 样本优先保护。
- t50 要有专门 history/goal prototype/switchability，不要被 all objective 淹没。
- full-waypoint 不能用 endpoint 成功直接替代，要单独评估 ADE/FDE shape。
- proximity / collision proxy 必须进 safety audit。
- runtime replay 必须证明 frozen policy 能按 batch 复现。

尚未证明有效的设计：

- JEPA independent downstream contribution。
- goal/scene independent main contribution。
- neighbor/interaction independent main contribution。
- ungated neural dynamics。
- global metric/seconds/t100 claim。
- Stage5C / SMC readiness。

## 6. 当前可写进论文的内容

可以写：

- M3W 是 protected dataset-local raw-frame 2.5D multi-agent world-state candidate。
- SDD pixel-space benchmark 和 Stage26 selector 是可靠 SDD 基座。
- External transfer 从 Stage31/32 失败，经 Stage37 history/prototype/safety policy 修复。
- Stage37 external t50 repair 有 bootstrap CI。
- Stage41/42 protected neural / composer / full-waypoint / proximity guard 提供进一步 evidence。
- Negative results 很重要：JEPA-only、hard selector、raw domain alignment、ungated neural、ordinary residual 都失败。
- Runtime replay、provenance、paper-freeze manifest、worktree caveat 提升了可复现性。

不能写：

- true 3D。
- foundation。
- metric。
- seconds-level。
- global t100 deployable。
- Stage5C executed。
- SMC enabled。
- JEPA 是生成式 world model。
- context/goal/scene/neighbor 是当前 main driver。
- local path 等于合法授权。

## 7. 当前研究质量判断

当前质量：

```text
强 protected 2.5D external world-state candidate / manuscript package
not yet full A-journal-ready
not foundation-track prototype
not true 3D
```

为什么不是 full A-journal-ready：

- 还没有全局 metric/time calibration。
- t100 source support 仍不足。
- ETH_UCY / TrajNet / UCY source/legal closure 未关闭。
- JEPA/Transformer/context 独立贡献仍弱。
- 当前 best 依赖 safety floor 和 baseline-family rollout context。
- 无保护 neural dynamics 不可部署。

为什么仍然有价值：

- 有从失败到修复的完整闭环。
- 有 strong baseline 对比。
- 有 no-leakage、validation-only、test-once、bootstrap、runtime replay、provenance。
- 有明确 negative results，而不是只报成功。
- Stage37/41/42 形成了外部 raw-frame 正迁移和 protected neural/composer evidence。

## 8. 下一步最短路径

最值得继续做三件事：

1. 关闭 DA-1 source/legal/time/t100 support。
   - 确认 ETH/BIWI、TrajNet++、UCY 原始 source terms。
   - 增加 independent t100-capable sources。
   - 做 train-only source-CV，不用 test 调参。

2. 做 DA-3 full-waypoint deployable dynamics repair。
   - 不能只用 endpoint-linear bridge。
   - 训练 all-agent full-waypoint 模型，同时保留 proximity/physical/easy loss。
   - 目标是超过 endpoint-linear all/hard，同时保留 t50/t100 shape gain。

3. 改变 context/scene/interaction target，而不是重复现有 gated residual。
   - 当前 DC 已说明 context switchability 不支持。
   - 需要新的任务定义：物理一致性、joint occupancy、interaction constraints、route topology 或 source-rich scene packs。

## 9. 给当前用户的直接答案

你问“做了什么、尝试了什么路线、哪些失败了、原因是什么、哪些成功了”：

答案是：

- 做了从数据采集、SDD benchmark、external conversion、selector 修复、neural dynamics、full-waypoint、runtime replay 到 source-support closure 的完整研究链。
- 最早的 JEPA/Transformer/ordinary residual 都没有直接变成主贡献。
- 真正有效的是 cost-aware / gain-harm / fallback-safe / history-aware / source-horizon-aware protected policy。
- SDD 内 Stage26 成功。
- External t50 Stage37 成功。
- Stage41/42 形成 protected neural / composer / full-waypoint evidence。
- 最新 Stage42-DD 明确 source/legal/time/t100 closure 仍未完成，所以不能越界 claim。
- 当前最安全的部署不是裸 neural，而是 Stage37/teacher floor 保护下的 Stage42 proximity-aware guarded composer / M3W-Neural v1 protected package。

一句话：

```text
M3W 现在有真实工程和实验价值，但它的强点是 protected 2.5D world-state policy，不是无保护、米制、秒级、3D、foundation world model。
```

<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:START -->
## Stage42-DE Full-Waypoint Deployment Gap Audit

- source: `fresh_stage42_de_full_waypoint_deployment_gap_audit`
- role: decide whether full-waypoint can be promoted from auxiliary/composer evidence to primary deployable world dynamics.
- gate: `17 / 17`; verdict `stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked`.
- decision: `protected_full_waypoint_composer_supported_deployment_promotion_blocked`.
- horizon_auxiliary_supported: `True`; guarded_composer_supported: `True`.
- primary deployable full-waypoint promotion: `False`.
- blockers: `['protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard', 'ungated_full_waypoint_easy_degradation_unsafe', 'source_legal_time_t100_closure_open', 'graph_group_interaction_has_proximity_caveat']`.
- Conclusion: keep Stage37/teacher or endpoint-linear safety floor; use guarded full-waypoint composer only as protected horizon/shape component until all/hard/proximity/source-support gaps are closed.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:END -->

<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:START -->
## Stage42-DF All-Hard / Proximity Full-Waypoint Repair

- source: `fresh_stage42_df_all_hard_proximity_full_waypoint_repair`
- role: validation-only repair search for the Stage42-DE all/hard/proximity full-waypoint deployment blocker.
- gate: `12 / 14`; verdict `stage42_df_all_hard_proximity_repair_partial`.
- test vs endpoint-linear: all `-0.67%`, t50 `-1.40%`, t100 raw `-0.66%`, hard `-0.72%`, easy `0.19%`.
- delta vs Stage42-CQ: all `-2.44%`, t50 `-2.46%`, t100 raw `-4.14%`, hard `-2.65%`, near@0.05 `-0.05%`.
- decision: `all_hard_proximity_repair_no_primary_promotion_keep_cq_guarded_composer`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:END -->

<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:START -->
## Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair

- source: `fresh_stage42_dg_full_waypoint_all_hard_loss_repair`
- role: actual retraining probe for all/hard/long-horizon weighted full-waypoint dynamics, following Stage42-DE/DF blockers.
- selected loss variant: `balanced` with lambda `100.0`.
- gate: `13 / 15`; verdict `stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `24.58%`, t50 `22.02%`, t100 raw `14.37%`, hard `23.75%`, easy `-25.66%`.
- delta vs Stage42-AM: all `0.00%`, t50 `0.00%`, t100 raw `0.00%`, hard `0.00%`, easy `0.00%`.
- decision: `weighted_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:END -->

<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:START -->
## Stage42-DH Full-Waypoint Proximity / Occupancy-Proxy Loss Repair

- source: `fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair`
- role: actual retraining probe for proximity/density/occupancy-proxy weighted full-waypoint dynamics after Stage42-DE/DF/DG blockers.
- selected candidate: `proximity_close_weighted` with `stage42_am_features` and lambda `100.0`.
- gate: `15 / 16`; verdict `stage42_dh_proximity_occupancy_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `25.51%`, t50 `22.14%`, t100 raw `14.34%`, hard `23.74%`, easy `-29.23%`.
- delta vs Stage42-AM: all `0.93%`, t50 `0.12%`, t100 raw `-0.03%`, hard `-0.01%`, easy `-3.57%`.
- decision: `proximity_occupancy_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:END -->

<!-- STAGE42_DI_GROUP_CONSISTENCY_FULL_WAYPOINT_REPAIR:START -->
## Stage42-DI Group-Consistency Full-Waypoint Repair

- source: `fresh_stage42_di_group_consistency_full_waypoint_repair`
- role: explicit all-agent group-consistency / proximity repair over source-level full-waypoint predictions after Stage42-DE/DF/DG/DH blockers.
- selected repair: `{'mode': 'repel_unsafe', 'min_sep': 0.08, 'margin': 0.0, 'strength': 0.5}`.
- gate: `17 / 17`; verdict `stage42_di_group_consistency_full_waypoint_repair_pass_promotable`.
- test vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- delta vs Stage42-AM: all `0.14%`, t50 `0.35%`, t100 raw `-0.02%`, hard `0.14%`, easy `0.03%`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- decision: `promote_stage42_di_group_consistency_full_waypoint_repair`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DI_GROUP_CONSISTENCY_FULL_WAYPOINT_REPAIR:END -->
