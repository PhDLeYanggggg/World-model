# M3W 当前总账：路线、失败、成功与真实质量

更新时间：2026-05-27  
结果来源：`cached_verified` 汇总已有 Stage18-Stage42 报告、gate、README、model/data card、`research_state.json`；少数最近 Stage42 报告本身标记为 `fresh_run`，但本 README 只是总结，不是新的训练、转换或评估。  
项目名：**M3W: Real-World Multimodal Agent-Scene World Model**

## 0. 先说真实边界

当前 M3W 不是 true 3D world model。  
当前 M3W 不是 large-scale foundation world model。  
当前 M3W 仍是 **protected dataset-local / raw-frame 2.5D multi-agent world-state candidate**。

必须继续保留的限制：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- 外部 ETH/UCY/TrajNet/OpenTraj 结果主要是 dataset-local / unverified weak-metric diagnostic，不可写成统一物理米制结果。
- t+50 / t+100 是 raw-frame horizon；除非完成 FPS、annotation stride、homography、scale 审计，否则不能写成 seconds-level。
- self-audited / visual-prior / inferred labels 不是 human gold。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- JEPA 表征预训练不是 latent rollout，也不能包装成生成式 world model。
- protected selector / safety floor 的成功不能包装成无保护神经网络已经完全可靠。
- 当前所有跨域成功都依赖严格 no-leakage：无 future endpoint input、无 central velocity、无 test endpoint goals、无 test-threshold tuning。

## 1. 当前最强结论

当前最强结果不是单一路线，而是分层：

| 场景 | 当前 best deployable / candidate | 状态 |
|---|---:|---|
| SDD pixel-space official split | Stage26 cost-aware expected-FDE selector | 可部署基座 |
| 外部 t+50 dataset-local transfer | Stage37 history + goal-prototype + conformal safe selector | 可部署外部 t50 selector |
| protected neural endpoint / tail dynamics | M3W-Neural v1 / Stage41 protected self-gated + composite-tail family | 候选，不是无保护神经动力学 |
| source/domain protected full-waypoint | Stage42 source-level / group-consistency protected policies | 强证据候选，仍受 safety floor / claim boundary 约束 |
| metric / seconds-level / true 3D | 无 | not ready |
| Stage5C latent generative | 无 | 禁止执行 |
| SMC | 无 | 禁止启用 |

一句话评价：

> M3W 已从“SDD-only selector scaffold”推进到“跨外部数据的 protected 2.5D multi-agent world-state candidate”，但它仍不是 true 3D / foundation / metric / seconds-level world model；当前最可信贡献是 **cost-aware safe selection + causal history + baseline-family rollout context + protected neural / full-waypoint refinements**，不是 JEPA 独立贡献，也不是无保护 Transformer/Hybrid 动力学。

## 2. 关键成功结果

### 2.1 Stage26：SDD cost-aware selector 修复 hard-class selector

之前 hard-class selector 失败的核心原因：

- oracle headroom 很大，但 best-baseline hard label 低 margin、噪声高；
- selector 对 easy samples 过度切换；
- 只预测“哪个 baseline 最好”会忽略错误切换的成本；
- 训练目标不是 regret-minimizing。

Stage26 改成 expected-FDE / cost-aware / confidence-gated / fallback-safe selection 后，成为 SDD 当前 best deployable：

| 指标 | Stage26 结果 |
|---|---:|
| t+50 improvement | 约 `+14.58%` |
| hard/failure improvement | 约 `+11.23%` |
| easy degradation | 约 `+1.81%` |
| 结论 | 过 gate，作为 SDD best deployable baseline |

意义：

- 证明“直接分类 baseline class”是错任务；
- 正确任务是预测 expected FDE / gain / harm / failure risk，然后保守 fallback；
- Stage26 仍是 SDD 层面的稳定基座。

### 2.2 Stage37：外部 t+50 transfer 修复

Stage31-36 的外部迁移长期失败，直到 Stage37 加入 past-only history window、scene-agnostic goal prototypes、t+50 switchability、conformal safety。

Stage37 final：

| 指标 | 结果 |
|---|---:|
| rows | `66303` |
| all improvement | `+13.48%` |
| t+50 improvement | `+8.46%` |
| t+50 bootstrap CI | `[+7.69%, +9.15%]` |
| hard/failure improvement | `+15.54%` |
| easy degradation | `0.041%` |
| gates | `16 / 16` |
| verdict | `stage37_t50_transfer_repaired_deployable` |

意义：

- 外部 t+50 从 0.0 safe fallback 推进为正迁移；
- t+50 的成功不是靠调 threshold，而是靠 history window + goal prototype + switchability/gain/harm/conformal safety；
- 当前 external t50 deployable floor 仍是 Stage37。

### 2.3 Stage41 / M3W-Neural v1：protected neural candidate

Stage39/40 的 Transformer/JEPA/Hybrid 早期尝试没有超过 Stage37；Stage41 之后通过 endpoint geometry、self-gating、composite-tail bounded neural dynamics，形成 M3W-Neural v1 候选。

M3W-Neural v1 package 中记录的 best metrics vs Stage37 floor：

| 指标 | M3W-Neural v1 |
|---|---:|
| rows | `55528` |
| all improvement vs Stage37 floor | `+21.03%` |
| t+50 improvement | `+13.65%` |
| t+100 raw diagnostic improvement | `+14.69%` |
| hard/failure improvement | `+20.38%` |
| easy degradation | `0.00%` |
| gates | `41 / 41` |
| caveat | protected candidate; not true 3D / not foundation / not Stage5C |

另一个 Stage41 fresh self-gated endpoint candidate 曾达到更强的 endpoint evidence：

| 指标 | fresh self-gated endpoint candidate |
|---|---:|
| all improvement vs floor | `+41.96%` |
| t+50 improvement | `+40.62%` |
| t+100 raw diagnostic | `+45.73%` |
| hard/failure | `+43.61%` |
| easy degradation | `0.00%` |
| caveat | endpoint geometry verified，但仍是 safety-gated，不是无保护 latent rollout |

意义：

- 神经路线最终不是完全失败；
- 可写的神经贡献是 protected endpoint/composite-tail neural dynamics under Stage37/teacher safety floor；
- 不能写成无保护 Transformer/JEPA/Hybrid 已经替代 Stage37。

### 2.4 Stage42 source-level / full-waypoint / group-consistency evidence

Stage42 长期做了很多“能否从 endpoint selector 推进到 full-waypoint world-state dynamics”的验证。关键成功路径是 source-level protected full-waypoint / group-consistency family。

重要节点：

| 阶段 | 结果 | 意义 |
|---|---:|---|
| Stage42-AM | all `+24.58%`, t50 `+22.02%`, t100 raw `+14.37%`, hard `+23.75%` | source-level raw-frame full-waypoint protected probe 成立 |
| Stage42-FE | all/t50/hard `+26.41% / +23.15% / +24.81%`, near@0.05 `1.32%` | constrained FC-to-DI safety composer，兼顾 accuracy 与 proximity safety |
| Stage42-FH | all/t50/t100raw/hard `+34.98% / +28.97% / +20.57% / +33.10%` | 修复 UCY weak-domain，dual-domain positive-safe |
| Stage42-FI | 2000-bootstrap CI low all/t50/t100raw/hard `+34.62% / +28.46% / +19.96% / +32.73%` | 冻结 replay + bootstrap，结果不是 test-tuned 偶然 |
| Stage42-HP | all/t50/t100raw/hard `+24.72% / +22.36% / +14.35% / +23.89%`; near@0.05 delta `-0.55%` | group-consistency source-level breakdown，通过 `23 / 23` gates |
| Stage42-HQ | all/t50/t100raw/hard `+32.89% / +26.99% / +21.12% / +31.89%` | 修复 HP 的 UCY zero-gain weak slice |
| Stage42-HR | all/t50/t100raw/hard `+27.72% / +26.99% / +6.79% / +25.93%`; t100 easy degradation `+2.56% -> -0.31%` | 修复 HQ 剩余 t100 easy harm，牺牲一部分 t100 raw gain 换安全 |

Stage42 的真实含义：

- 证明 protected full-waypoint / group-consistency world-state policy 可以在 raw-frame dataset-local 外部 split 上提供稳定正证据；
- 证明安全边界必须保留：teacher / floor / Stage37 safety / validation-only gates 仍然必要；
- 仍不能 claim true 3D、metric、seconds-level、foundation、Stage5C 或 SMC。

## 3. 失败路线与失败原因

### 3.1 Hard classification selector

失败表现：

- Stage24 validation-selected selector t+50 improvement `-43.3%`；
- easy degradation `11.33%`；
- oracle headroom 大，但训练 selector 反而大幅伤害 easy cases。

失败原因：

- best-baseline label 是 high-noise / low-margin label；
- 样本里很多 baseline FDE 差距很小，强行 one-hot 分类会过拟合；
- selector 没有建模 harm over fallback；
- 没有 confidence gate / predicted gain margin / easy guard。

修复：

- Stage25/26 改为 expected-FDE / regret-aware / cost-aware / confidence-gated fallback policy。

### 3.2 JEPA downstream 主线

失败表现：

- Stage18 JEPA non-collapse，但 selector / failure / correction / official t+50 没有 lift；
- Stage23/24/39 的 JEPA probe 多次没有稳定 downstream lift；
- Stage39 JEPA failure AUROC with JEPA 低于 baseline。

失败原因：

- JEPA latent non-collapse 不等于对 selector/failure/trajectory 有用；
- scene/image/trajectory latent 没有和 cost-aware switch decision 对齐；
- 当前任务主要由 causal history + baseline-family rollout + safety floor 驱动，JEPA 表征未成为边际贡献。

结论：

- JEPA 可保留为 representation diagnostic；
- 不能写成 M3W 当前主贡献；
- 不能写成生成式 world model。

### 3.3 SDD -> external zero-shot

失败表现：

- Stage31 external zero-shot transfer：all improvement `-92.67%`，t50 `-278.57%`；
- Stage32 普通 normalization / latent alignment 后 external adapted selector 约 `0` improvement；
- Stage33 仍是 safe fallback `0.0`，不是正迁移。

失败原因：

- SDD pixel-space 与 external dataset-local coordinates 不兼容；
- scene/goal/interaction 信息缺失；
- agent type、horizon、track length、scale/homography 不匹配；
- latent distribution alignment 缩小距离不等于 predictive lift。

修复路径：

- Stage34 row geometry + train-only goals 得到局部 t50/hard signal；
- Stage35 selective transfer 使 all/hard 正但 t50 仍 0；
- Stage37 history/prototype/conformal 才修复外部 t50。

### 3.4 Ordinary residual / correction specialist

失败表现：

- Stage22/23/24 correction 没有可靠改善 hard/failure；
- Stage38 bounded correction 在 Stage37 保护下没有安全超过 Stage37；
- 无保护 residual 往往 easy harm 很高。

失败原因：

- 直接预测 residual 容易在 easy cases 上过修正；
- 没有可靠 gain/harm/failure gate 时，residual 会破坏 strongest/floor baseline；
- correction 需要先有 selector/failure/headroom 支撑。

结论：

- 当前 correction 不作为 deployable 主模型；
- 只允许 bounded / protected / diagnostic。

### 3.5 Unprotected Transformer / Hybrid neural dynamics

失败表现：

- Stage39 Transformer/JEPA/Hybrid trained，但 neural_with_fallback 没超过 Stage37；
- Stage40 best neural with fallback 等同或低于 Stage37；neural_without_fallback 灾难性失败：
  - all improvement `-126.36%`
  - t50 `-292.10%`
  - easy degradation `+612.31%`

失败原因：

- 神经网络学到了部分 dynamics，但没有学会“不要切错”；
- 没有 safety floor 时 easy cases 被严重破坏；
- 模型容易复制 Stage37 或被 fallback 吃掉；
- endpoint/full-waypoint 输出需要几何对齐与 safety gate。

修复：

- Stage41 self-gated endpoint / composite-tail family 在 Stage37/teacher floor 保护下形成候选；
- 仍不能部署无保护 neural dynamics。

### 3.6 Scene/goal 与 neighbor/interaction 独立贡献

失败表现：

- Stage42-CJ goal/scene gated expert：validation gate 选 baseline_family_control，不选 goal/scene；
- Stage42-CK neighbor/interaction gated expert：validation gate 也选 baseline_family_control；
- Stage42-AQ/AR/AS neural/sequence/graph residual context 均未超过 baseline-family first-stage。

失败原因：

- 当前手工 scene/goal/neighbor features 的边际信号弱；
- 主要解释力来自 baseline-family rollout context；
- held-out scenes 的 train-only goals 不稳定；
- graph/interaction 特征没有足够强的结构监督。

结论：

- scene/goal/neighbor/interaction 可以作为 auxiliary diagnostic；
- 不能写成独立主贡献。

### 3.7 Metric / seconds-level / true 3D claim

失败或 blocked 原因：

- SDD 仍是 pixel-space；
- external 有 H/FPS/annotation-step evidence，但 source-specific homography direction、coordinate convention、scale、license/terms 未全闭环；
- ETH-Person XML 技术 dry-run positive，但 terms/allowed-use 未确认；
- TrajNet snippets 短，t100 source support 不足。

结论：

- 继续写 dataset-local / raw-frame 2.5D；
- 不能写 metric / seconds-level / true 3D。

## 4. 成功路线的共同机制

从 Stage26 到 Stage42，真正有效的机制很清楚：

1. **把任务从 hard classification 改成 cost-aware expected-risk policy**  
   预测 FDE / gain / harm / failure，再决定是否切换。

2. **保守 fallback 是必要机制，不是临时补丁**  
   低信心、低 margin、easy-risk、高 harm-risk 样本必须回退到 strongest baseline / Stage37 / teacher floor。

3. **past-only history window 是外部 t50 的关键输入**  
   Stage37 t50 修复证明 history length、speed/curvature/stop-go、neighbor density/TTC、goal prototype 比普通 metadata 更有用。

4. **baseline-family rollout context 是当前最强解释变量**  
   多次 ablation 显示 history/goal/neighbor 单独贡献不稳定，baseline-family relative rollout 才是 dominant mechanism。

5. **神经网络只有在 safety-gated / self-gated / composite-tail 下才有用**  
   无保护 neural dynamics 仍会伤 easy；protected neural endpoint/composite-tail 可以成为候选。

6. **source/domain/horizon 不能混成一个全局 claim**  
   TrajNet、UCY、ETH_UCY 的支持程度不同，t50/t100/horizon slice 也不同；必须按 slice 写证据。

## 5. 当前模型质量判断

当前 M3W 的质量可以这样定位：

| 维度 | 当前水平 |
|---|---|
| 工程可运行性 | 高：有脚本、报告、gate、pytest、readme、state、policy hash |
| SDD selector 基座 | 成立：Stage26 过 gate |
| 外部 t50 transfer | 成立：Stage37 过 `16 / 16`，CI 正 |
| protected neural evidence | 部分成立：M3W-Neural v1 / Stage41 protected candidate |
| full-waypoint / group-consistency evidence | 部分成立：Stage42 多个 protected source-level positive-safe 结果 |
| unprotected neural dynamics | 不成立 |
| JEPA 独立贡献 | 不成立 |
| scene/goal/interaction 独立贡献 | 不成立 |
| t100 global deployable claim | 仍受 source/horizon/easy guard 限制 |
| metric / seconds-level claim | 不成立 |
| true 3D / foundation claim | 不成立 |
| CCF-A / A刊候选 | 有候选证据包雏形，但仍缺外部 source/legal/time/metric 闭环与更强独立模块贡献 |

## 6. 当前 best deployable 分层

推荐写法：

```text
SDD deployable floor:
  Stage26 cost-aware expected-FDE selector

External t50 deployable floor:
  Stage37 history/prototype/conformal safe selector

Protected neural / world-state candidate:
  M3W-Neural v1 composite-tail / self-gated endpoint family under Stage37/teacher safety floor

Protected full-waypoint candidate:
  Stage42 source-level / group-consistency policies under validation-only gates and floor safety

Not deployable:
  JEPA-only
  Transformer-only without fallback
  Hybrid without safety floor
  ordinary residual correction
  raw latent alignment
  global floor-free neural dynamics
```

## 7. 还不能写的 claim

绝对不能写：

- “M3W 是 true 3D world model”
- “M3W 是 foundation world model”
- “SDD / external 是统一 metric benchmark”
- “t+50/t+100 是 seconds-level long horizon”
- “JEPA 是生成式 world model”
- “Stage5C 已执行”
- “SMC 已启用”
- “simulation success 等于 real-world success”
- “self-audited silver 是 human gold”
- “external fallback 0.0 是正迁移”
- “unprotected neural dynamics 已经可靠部署”
- “scene/goal/interaction 是当前独立主贡献”

## 8. 下一步最值得做

1. **Stage42-HR 后续：把 t100 easy guard 与 group-consistency policy 做 freeze/replay/bootstrap**  
   目标是确认 HR 不是一次性 artifact，并把 t100 easy safety 写进 frozen deployment contract。

2. **补外部 source/legal/time/geometry 闭环**  
   尤其是 ETH/UCY、UCY、TrajNet 的 official terms、homography/scale/FPS/annotation stride、t100-capable independent sources。没有这个闭环，metric/seconds/global t100 仍不能写。

3. **重训真正结构化 interaction / scene model，而不是继续堆 tabular features**  
   当前 scene/goal/neighbor 单独贡献不稳。下一步应当用更强的 graph/sequence/world-state objective，并仍由 Stage37 / teacher floor 保护。

4. **把 paper package 从“结果堆叠”整理成一条主论点**  
   推荐主论点是：`safe cost-aware protected world-state dynamics under causal baseline-family rollout context`，而不是 JEPA 或 pure Transformer。

## 9. 最终一句话

M3W 已经做出了有价值的真实路线探索：从 SDD cost-aware selector，到 external t50 repaired deployable selector，再到 protected neural / full-waypoint / group-consistency world-state candidate。失败也很清楚：JEPA、raw latent alignment、hard-class selector、ordinary residual、unprotected Transformer/Hybrid、scene/goal/neighbor 独立贡献都没有稳定站住。当前项目质量是 **强 protected 2.5D multi-agent world-state candidate**，不是 true 3D，也不是 foundation；下一步要从 protected candidate 继续推进 source/time/geometry 闭环和更强结构化 world dynamics。

<!-- STAGE42_HS_T100_EASY_GUARD_FREEZE:START -->
## Stage42-HS Frozen T100 Easy Guard

- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`
- role: freeze Stage42-HR validation-only domain|t100 easy guard as a lightweight policy/replay artifact.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json`
- policy hash: `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e`
- gate: `27 / 27`; verdict `stage42_hs_t100_easy_guard_freeze_pass`.
- replay: decision table exact `True`, metric summary exact `True`.
- guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- t100 easy degradation after guard: `-0.31%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HS_T100_EASY_GUARD_FREEZE:END -->

<!-- STAGE42_HT_T100_EASY_GUARD_RUNTIME:START -->
## Stage42-HT Runtime T100 Easy Guard Policy

- source: `fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy`
- role: convert the frozen Stage42-HS domain|t100 easy guard into a callable runtime policy API.
- gate: `19 / 19`; verdict `stage42_ht_t100_easy_guard_runtime_policy_pass`.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json`
- policy hash: `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e`
- runtime rule: TrajNet|100 falls back to floor; UCY|100 keeps candidate; unknown t100 domains fallback to floor; non-t100 rows are unchanged.
- inherited guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HT_T100_EASY_GUARD_RUNTIME:END -->
