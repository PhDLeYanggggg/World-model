# M3W 用户版详细总账：路线、失败、成功、当前质量与下一步

更新时间：2026-05-27

工作目录：`/Users/yangyue/Downloads/World`

本文用途：按用户要求，把 M3W 长期目标内已经做过的事情、尝试过的路线、失败原因、成功证据、当前模型质量、可部署边界和下一步最短路径集中到一个 README。本文是总结文件，不是新训练结果；不会把旧结果写成新跑结果，不会把 diagnostic 写成 deployable，不会把 not_run 写成完成。

## 1. 当前一句话结论

M3W 当前最诚实定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

也就是说，它已经不再只是早期的 SDD-only demo。它有 SDD、external top-down、source/domain、runtime replay、bootstrap、no-leakage、claim guard 等一整条证据链。但它仍然不是：

- true 3D world model
- large-scale foundation world model
- global metric / meter-level predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative execution
- SMC-ready system

当前 best deployable 分层：

| 场景 | 当前最强结果 | 是否可部署 | 边界 |
| --- | --- | --- | --- |
| SDD pixel/raw-frame | Stage26 cost-aware selector | 是 | 仅限 SDD pixel/raw-frame，不是 metric。 |
| External t+50 dataset-local/raw-frame | Stage37 causal-history + goal-prototype safe selector | 是 | external selector-level deployable，不是 true 3D。 |
| Protected neural / world-state | M3W-Neural v1 composite-tail safe-switch under Stage37/teacher floor | 候选可报告 | 受 Stage37/teacher floor 保护；不是 ungated neural deployment。 |
| Source/domain/full-waypoint | Stage42-FH/FI frozen protected policy family | 可作为 protected evidence | 允许 source/domain robust claim；不允许 uniform horizon claim。 |
| Partial t50 floor relaxation | Stage42-GT target slices TrajNet|50 + UCY|50 | narrow 支持 | 只支持局部 t50 floor relaxation，不支持 global floor removal。 |
| h100 / uniform horizon | blocked | 否 | TrajNet|100 和 UCY|100 仍需 source/legal/guarded conversion。 |

最新 claim guard：

- Stage42-GZ full-waypoint claim guard：通过 `18 / 18`，阻止把 endpoint-only 或 ungated full-waypoint 写成 learned full-waypoint dynamics。
- Stage42-HA full-waypoint overclaim linter：扫描 15 个关键 README/报告/论文包文件，最终 violations = `0`，gate `14 / 14`。

## 2. 我实际做过/尝试过的主路线

### 2.1 数据与 benchmark 路线

做了什么：

- 建立 WAM-style data registry、web dataset acquisition、license audit、download/user-action plan。
- 接入 SDD，本地解压并转换成 per-video world-state shards。
- 构建 SDD scene packs、lazy episodes、HardBench、BaselineFailureBench、GoalBench、no-leakage audit、strongest causal baselines。
- 后续扩展 OpenTraj / UCY / ETH-UCY / TrajNet 外部 top-down pedestrian 数据，建立 external feature store、row geometry、history windows、goal prototypes、source/domain splits。

结果：

- SDD benchmark 成功建立，但它是 pixel-space raw-frame benchmark。
- External 数据成功进入 dataset-local/raw-frame 评估链，但并没有统一 metric / seconds-level 标定。
- 一些 external source 仍受 legal/source/path/calibration blocker 约束，不能把 prefill 或 registry 当 converted/evaluated 数据。

### 2.2 强因果 baseline 与 fallback 路线

做了什么：

- 系统比较 constant position、causal constant velocity、damped velocity、constant acceleration、turn-rate、scene-clamped、goal/prototype-directed 等 baseline。
- 将 strongest causal baseline 作为所有 learned policy 的 floor。
- 评估 oracle headroom、selector regret、harm over fallback、easy degradation、hard/failure improvement。

结果：

- 这是整个项目最稳的骨架。
- 早期很多 learned selector / neural model 失败，不是因为没有 oracle headroom，而是因为错误切换会伤 easy cases。
- 后来的 Stage26、Stage37、Stage42 protected policies 都依赖这条 fallback-safe 路线。

### 2.3 SDD selector 路线

做了什么：

- 先训练 hard classification selector，预测“哪个 baseline 最好”。
- 发现 hard-class selector 对低 margin 样本过度切换。
- 改成 expected-FDE / regret-aware / confidence-gated / fallback-safe selector。

结果：

- Stage24 hard-class selector 失败：t+50 improvement 约 `-43.3%`，easy degradation 约 `11.33%`。
- Stage26 cost-aware selector 成功：t+50 约 `+14.58%`，hard/failure 约 `+11.23%`，easy degradation 约 `+1.81%`。
- 结论：selector 不能做硬分类，必须做 cost-aware / regret-aware / conservative fallback。

### 2.4 JEPA 表征路线

做了什么：

- 训练过 trajectory-only JEPA、scene/trajectory JEPA、interaction-aware JEPA。
- 检查 latent non-collapse、selector probe、failure predictor probe、goal predictor、hard/failure correction lift。

结果：

- 多次 non-collapse，但 downstream lift 不稳定或没有提升。
- Stage18、Stage19、Stage22、Stage23、后续 M3W/Stage42 都没有证明 JEPA 是主贡献。
- 结论：JEPA 只能作为 auxiliary/diagnostic representation，不能写成 latent generative world model，也不能作为当前主 claim。

### 2.5 Transformer / Hybrid neural dynamics 路线

做了什么：

- 训练 Transformer-only、JEPA-only、JEPA+Transformer hybrid。
- 后续又做 protected neural dynamics、full-waypoint sequence dynamics、group-consistency full-waypoint、source/domain policy。

结果：

- Stage39 Transformer/JEPA/Hybrid 真实训练跑通，但 neural_with_fallback 没有超过 Stage37，world dynamics candidate gate fail。
- 无保护 neural 或 ungated endpoint/full-waypoint 容易严重伤 easy cases。
- 后续 M3W-Neural v1 / Stage41 / Stage42 的 protected neural evidence 有价值，但依赖 Stage37/teacher floor。

结论：

- 当前不能说“神经网络独立超过 Stage37 并可无保护部署”。
- 可以说“protected neural/full-waypoint/group-consistency world-state evidence 在安全 floor 下有贡献”。

### 2.6 External cross-domain 路线

做了什么：

- 先做 SDD -> external zero-shot transfer。
- 然后做 normalization、relative targets、latent adapter、row geometry、scene packs、train-only goals。
- Stage37 改为 past-only history window + scene-agnostic goal prototypes + t50 switchability/gain/harm + conformal safety。

结果：

- Stage31 zero-shot 外部迁移大失败：all improvement 约 `-92.67%`，t50 约 `-278.57%`。
- 普通 normalization 和 latent adapter 缩小分布差异，但不带来 predictive lift。
- Stage37 修复 external t50：all `+13.48%`，t50 `+8.46%`，t50 bootstrap CI `[+7.69%, +9.15%]`，hard/failure `+15.54%`，easy degradation `0.041%`，gate `16 / 16`。

结论：

- 外部迁移不是靠 SDD latent zero-shot 完成的。
- 真正有效的是 past-only history、目标原型、switchability/gain/harm、safe fallback。

### 2.7 Correction / residual 路线

做了什么：

- 训练 bounded correction、horizon-specific correction、hard-only correction、t50-only correction。
- 形式是 selected baseline + bounded residual。

结果：

- Stage38 correction 没有安全超过 Stage37。
- 普通 residual 容易伤 easy，尤其当 strongest baseline 已经很强时。

结论：

- 当前不部署 correction specialist。
- correction 只有在 selector/failure/hard/easy gates 全过时才可作为后续计划，不能绕过 Stage37。

### 2.8 Safety / physical validity 路线

做了什么：

- 引入 easy degradation、harm over fallback、near-collision@0.05、jagged-rate、physical validity proxy。
- 做 proximity-aware composer、group consistency、waypoint-wise repel、temporal repel、Pareto composer、objective-level safety training。

结果：

- 多个路线出现“accuracy 更高但 near-collision 更差”或“proximity 修复但 ADE/hard 降低”。
- Stage42-CQ/CR/FE 证明 safety guard 是必要的。
- Stage42-GT 证明局部 t50 floor relaxation 可安全，但 global floor removal 仍不允许。

### 2.9 Source/domain/full-waypoint 路线

做了什么：

- 从 endpoint-linear bridge 进入 full-waypoint shape evidence。
- 做 source/frame/horizon group-consistency full-waypoint policy。
- 对 TrajNet/UCY/ETH_UCY 等 source/domain 做 robust audit、frozen replay、bootstrap。

结果：

- Stage42-FH source/domain protected composer 成功：all/t50/t100raw/hard = `34.98% / 28.97% / 20.57% / 33.10%`。
- Stage42-FI frozen replay：policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`，exact replay diff `0`，2000-bootstrap CI low all/t50/t100raw/hard = `34.62% / 28.46% / 19.96% / 32.73%`。
- Stage42-FJ/FK/FL/FM/FN/FO/FP 进一步证明：可以写 dual-domain/source robust，但不能写 uniform horizon robust。

### 2.10 Claim guard / paper evidence 路线

做了什么：

- 建立 module contribution ledger、claim-boundary linter、paper claim evidence audit、full-waypoint claim guard、overclaim linter、source/legal blockers、reviewer replay package。

结果：

- Stage42-FU/GJ 锁定可写主贡献模块：history、domain expert、safe switch、teacher floor、group-consistency full-waypoint、full-waypoint shape、endpoint bridge。
- 被阻止作为独立主 claim 的模块：JEPA、Transformer、scene/goal、neighbor/interaction。
- Stage42-GZ/HA 最新 guard/linter 防止把 endpoint-only、ungated full-waypoint、metric/seconds、true 3D、foundation、Stage5C、SMC 写成成功。

## 3. 失败路线、失败原因与处理

| 失败/受阻路线 | 表现 | 根因 | 处理 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 t50 约 `-43.3%`，easy degradation 约 `11.33%` | oracle label low-margin，高歧义；硬分类导致 easy cases 过度切换 | 改 expected-FDE / regret-aware / fallback-safe selector |
| Stage18/19/22/23 JEPA | non-collapse 但无 downstream lift | representation target 与部署目标错位；latent variance 不等于 gain/harm | 降级为 diagnostic/auxiliary |
| SDD->external zero-shot | all 约 `-92.67%`，t50 约 `-278.57%` | 坐标、scale、horizon、agent type、scene/goal context mismatch | external row geometry + relative target + history/prototype |
| latent adapter / CORAL | latent gap 缩小但 selector 无正迁移 | distribution alignment 不等于 task alignment | 不作为 success claim |
| ordinary residual correction | 不稳定超过 Stage37，容易伤 easy | 直接改轨迹比选择/回退更危险 | 不部署 correction |
| unprotected Transformer/Hybrid | 无保护 neural 不安全或无法超过 Stage37 | dataset-local/raw-frame、缺 metric/scene grounding、teacher floor 仍必要 | 只报告 protected neural candidate |
| scene/goal 独立 claim | 多轮 ablation 下不稳定或低于 baseline-family | train-only goal proxy 对 held-out/source shift 支持有限 | 只能 auxiliary，不做主贡献 |
| neighbor/interaction 独立 claim | scalar/graph neighbor 特征无法稳定独立提升 | 当前 interaction 表达太弱，不能替代 group-consistency | 只允许 group-consistency full-waypoint 作为受限贡献 |
| proximity / repel 修复 | accuracy 与 near-collision 常互相牺牲 | post-hoc 几何修复会损 ADE/hard | 用 constrained safety fallback |
| uniform h100 / horizon claim | TrajNet|100、UCY|100 仍 weak | low-margin ambiguity、source support 稀疏、h100 context 不足、legal conversion 未 ready | 先做 source/legal/guarded conversion，不继续盲调 |

## 4. 成功路线与关键证据

| 成功点 | 关键数字 | 结论边界 |
| --- | --- | --- |
| Stage26 SDD cost-aware selector | t50 `+14.58%`；hard/failure `+11.23%`；easy degradation `+1.81%` | SDD pixel/raw-frame best deployable |
| Stage37 external t50 repair | all `+13.48%`；t50 `+8.46%`；CI `[+7.69%, +9.15%]`；hard/failure `+15.54%`；easy `0.041%`；gate `16/16` | external dataset-local/raw-frame selector success |
| M3W-Neural v1 protected candidate | all `+21.03%`；t50 `+13.65%`；t100raw `+14.69%`；hard/failure `+20.38%`；easy `0.00%` | protected neural candidate，不是 ungated neural deployment |
| Stage42-CQ proximity guard | all `+1.77%`；t50 `+1.07%`；t100raw `+3.48%`；near@0.05 不劣于 endpoint/floor | safety-sensitive protected composer |
| Stage42-DL/DM runtime replay | rows `47,458`；all/t50/t100raw/hard `+24.72% / +22.36% / +14.35% / +23.89%`；near@0.05 `1.94% -> 1.38%` | reviewer/runtime replay evidence |
| Stage42-FE constrained safety composer | all/t50/hard `26.41% / 23.15% / 24.81%`；near@0.05 `1.32%` | 修复 FC proximity blocker |
| Stage42-FH source/domain composer | all/t50/t100raw/hard `34.98% / 28.97% / 20.57% / 33.10%`；TrajNet/UCY positive-safe | dual-domain/source protected success |
| Stage42-FI frozen replay | replay diff `0`；CI low all/t50/t100raw/hard `34.62% / 28.46% / 19.96% / 32.73%` | frozen policy，不是 test-tuned 偶然输出 |
| Stage42-GT partial floor relaxation | target union t50 rows `11,538`；t50 `+28.97%`；hard `+28.97%`；easy `-21.41%`；near@0.05 `-0.74pp` | 只支持 TrajNet|50 + UCY|50 局部 relaxation，不支持 global floor removal |
| Stage42-GZ/HA claim guards | GZ gate `18/18`；HA scans `15` files, violations `0`, gate `14/14` | 防止 full-waypoint / metric / Stage5C / SMC overclaim |

## 5. 当前 best deployable 到底是谁

当前不是一个单一“万能模型”，而是分层部署：

1. **SDD：Stage26 cost-aware selector。**
   这是 SDD pixel/raw-frame benchmark 上当前最强可部署组件。

2. **External t50：Stage37 causal-history + goal-prototype safe selector。**
   这是 external dataset-local/raw-frame t50 迁移上当前最稳的 deployable selector。

3. **Protected neural / source-domain：M3W-Neural v1 + Stage42-FH/FI protected policy family。**
   这是目前最强的 protected world-state candidate，但部署仍依赖 Stage37/teacher floor 和 safe switch。

4. **不部署：ungated neural、ordinary residual、unprotected full-waypoint。**
   这些路线要么伤 easy，要么 proximity 不安全，要么没有超过 Stage37。

## 6. 当前大概是什么质量

如果按论文/工程成熟度分层：

```text
已成立：
  严格 no-leakage 的 dataset-local/raw-frame 2.5D multi-agent world-state candidate
  有 SDD、external t50、source/domain、runtime replay、bootstrap、claim guard 证据
  有清晰失败分析和不可 claim 边界

部分成立：
  protected neural / full-waypoint / group-consistency evidence
  source/domain robust evidence
  partial t50 floor relaxation

尚未成立：
  true 3D
  global metric / seconds-level prediction
  large-scale foundation world model
  ungated neural world dynamics
  uniform horizon / h100 robust success
  JEPA 或 Transformer 作为独立主贡献
  Stage5C / SMC readiness
```

所以当前质量不是“demo”，也不是“foundation”。最准确说法是：

```text
一个有比较完整证据链的 protected 2.5D multi-agent world-state research candidate。
```

## 7. 当前最重要的 blocker

1. **h100 / uniform horizon blocker。**
   TrajNet|100 与 UCY|100 仍没有被稳定修复。Stage42-FY 后已经明确：不能继续用同一套特征/threshold 反复重试。

2. **source/legal blocker。**
   Stage42-GW/GX/GY 已经把 UCY h100 candidate、integrity、terms prefill 做好，但 legal acceptance / allowed use / source identity / local path 仍需要用户确认。agent 不能代填 license acceptance。

3. **metric/seconds blocker。**
   还没有全局 homography/scale/effective seconds 证据，所以仍不能写 metric 或 seconds-level。

4. **ungated neural blocker。**
   无保护 Transformer/Hybrid/Full-waypoint 仍不能部署；teacher/Stage37 floor 仍是必要安全机制。

5. **independent module contribution blocker。**
   JEPA、Transformer、scene/goal、neighbor/interaction 当前不能作为独立主 claim。

## 8. 下一步最短路径

1. **先做 legal/source confirmation。**
   使用 Stage42-GY 生成的 terms prefill，让用户确认 UCY/ETH_UCY/TrajNet 的 official source identity、allowed use、terms accepted、local path。没有这个确认，conversion_ready 必须继续是 0。

2. **只对 legal-ready source 做 guarded conversion。**
   转换后重新跑 no-leakage、source-CV、baseline、Stage37/Stage42 policy replay。不能把 registry、prefill、hint 当 converted/evaluated data。

3. **再修 h100 / uniform horizon。**
   需要真实 long-horizon source support、row-level h100 context、更强 history/neighbor/group features 和 stricter easy safety gate。

4. **神经网络路线继续但必须 protected。**
   训练目标应是 gain/harm、switchability、group-consistency、full-waypoint consistency、source/horizon-aware decision，而不是普通 residual。Stage37/teacher floor 不能全局移除。

## 9. 严格禁止的 claim

这些现在仍不能写：

- M3W 是 true 3D world model。
- M3W 是 foundation world model。
- SDD/external 结果是 metric / meter-level。
- t50/t100 是 seconds-level long horizon。
- JEPA 是 latent generative world model。
- Transformer/Hybrid 已经独立超过 Stage37 并可 ungated deployment。
- full-waypoint endpoint-linear success 等于 learned full-waypoint dynamics。
- partial t50 floor relaxation 等于 global floor removal。
- source terms prefill 等于 permission。
- candidate path / registry 等于 converted dataset。
- Stage5C 已执行。
- SMC 已启用。

## 10. 主要证据文件入口

- 根 README 索引：`README_RESULTS.md`
- 本文件：`README_M3W_USER_DETAILED_SUMMARY_ZH.md`
- 历史长版总账：`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`
- M3W-Neural v1 package：`outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- Stage37 external t50：`outputs/stage37_t50_history/report_stage37_final.md`
- Stage38 robustness：`outputs/stage38_external_robustness/report_stage38_final.md`
- Stage39 neural dynamics：`outputs/stage39_neural_dynamics/report_stage39_final.md`
- Stage42 full-waypoint claim guard：`outputs/stage42_long_research/full_waypoint_claim_guard_stage42.md`
- Stage42 full-waypoint overclaim linter：`outputs/stage42_long_research/full_waypoint_overclaim_linter_stage42.md`
- Stage42 floor-relaxation safety stress：`outputs/stage42_long_research/floor_relaxation_safety_stress_stage42.md`

## 11. 最后总结

这个目标内最有价值的经验是：

1. 强 baseline 很强，模型不能靠“预测更复杂”自动赢。
2. hard classification selector 会伤 easy；cost-aware / expected-FDE / safe fallback 才有效。
3. JEPA non-collapse 不等于 downstream 有用。
4. 无保护 neural dynamics 当前不安全。
5. External transfer 的关键不是 SDD zero-shot latent，而是 past-only history、goal prototypes、relative targets、gain/harm/safety decision。
6. 真正可写的贡献不是“我们有一个万能世界模型”，而是“在严格无泄露、dataset-local/raw-frame top-down 多智能体数据上，构建了 protected world-state policy，并通过 safe switch、teacher floor、group-consistency/full-waypoint protected evidence 获得稳定提升，同时明确限制 true 3D/metric/foundation/ungated neural claims”。

当前 best deployable 仍是分层保护策略，不是无保护大模型：

```text
SDD: Stage26 cost-aware selector
External t50: Stage37 safe selector
Source/domain protected world-state: Stage42-FH/FI frozen policy family
Neural/full-waypoint: protected candidate only, under Stage37/teacher floor
```

<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:START -->
## Stage42-HB Teacher-Floor Necessity Meta-Audit

- source: `fresh_stage42_hb_teacher_floor_necessity_meta_audit`
- gate: `16 / 16`
- verdict: `stage42_hb_teacher_floor_necessity_meta_audit_pass`
- Direct conclusion: Stage37 / teacher floor is the current safety mechanism and rollout-context floor, not merely a disposable crutch.
- Protected current all/t50/t100raw/hard/easy: `21.03%` / `13.65%` / `14.69%` / `20.38%` / `0.00%`.
- Ungated endpoint/full-waypoint easy degradation remains unsafe: `124.59%` / `124.59%`.
- Narrow t50 floor relaxation is supported only on selected slices: rows `11538`, t50 `28.97%`, hard `28.97%`, easy `-21.41%`.
- Global floor removal and floor-free neural deployment remain false.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:END -->

<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:START -->
## Stage42-HC Floor-Alternative Gate Stress Matrix

- source: `fresh_stage42_hc_floor_alternative_gate_stress`
- gate: `14 / 14`
- verdict: `stage42_hc_floor_alternative_gate_stress_pass`
- Tested Stage42-E internal self-gate, uncertainty gate, conformal risk gate, harm predictor, teacher-dependent gates, and bounded residual families as floor alternatives.
- floor-free deployable count: `0`; teacher-dependent deployable count: `6`.
- best floor-free candidate `harm_predictor_gate` reaches all/t50/hard `35.95%` / `25.20%` / `35.86%` but is not deployable because `['near_collision_delta_over_1pp']`.
- best deployable teacher-dependent candidate `current_composite_tail_policy` reaches all/t50/hard `21.03%` / `13.65%` / `20.38%` with easy `0.00%`.
- Deployment decision remains: keep Stage37/teacher floor globally; allow only validation-backed partial t50 relaxation on selected slices.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:END -->

<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:START -->
## Stage42-HD Floor-Free Proximity-Guard Repair

- source: `fresh_stage42_hd_floor_free_proximity_guard_repair`
- gate: `13 / 13`
- verdict: `stage42_hd_floor_free_proximity_guard_repair_pass`
- Tested floor-free internal/harm/uncertainty/conformal gates with a validation-selected proximity guard.
- pre-guard deployable count: `0`; post-guard deployable count: `4`.
- best post-guard family `harm_predictor_gate` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%` with easy `0.00%` and collision delta `-0.47%`.
- The teacher gate is not used in this repair, but causal floor fallback remains required; this is not global floor removal.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:END -->

<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:START -->
## Stage42-HE Floor-Free Proximity-Guard Robustness Audit

- source: `fresh_stage42_he_floor_free_proximity_guard_robustness`
- gate: `21 / 21`
- verdict: `stage42_he_floor_free_proximity_guard_robustness_pass`
- Audits the Stage42-HD teacherless proximity-guard repaired gate with 2000-bootstrap and per-domain/per-horizon checks.
- policy `harm_predictor_gate` with min_sep `0.05` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%`.
- bootstrap CI lows all/t50/t100raw/hard `20.38%` / `13.22%` / `12.94%` / `19.57%`; easy CI high `-16.17%`.
- robust_positive_domains: `ETH_UCY, TrajNet, UCY`; weak_domain_horizon_slices: `none`.
- Teacher gate is not used, but causal floor fallback remains required. This is not global floor removal, not metric/seconds, not true 3D, not Stage5C, and not SMC.
<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:END -->

<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:START -->
## Stage42-HF Teacherless Gate Deployment Contract

- source: `fresh_stage42_hf_teacherless_gate_deployment_contract`
- verdict: `stage42_hf_teacherless_gate_deployment_contract_pass`
- gates: `15 / 15`
- result: Stage42-HE supports a teacherless proximity-guarded switch gate, but only with causal floor fallback.
- metrics: all `20.74%`, t50 `13.82%`, t100 raw diagnostic `13.68%`, hard/failure `19.99%`, easy degradation `0.00%`.
- allowed claim: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked claims: global causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C execution, and SMC.
- deployment default remains protected causal-floor fallback; Stage42-HF is a claim/deployment contract refresh, not new training.
<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:END -->

<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:START -->
## Stage42-HG Teacherless / Floor-Free Claim Linter

- source: `fresh_stage42_hg_teacherless_claim_linter`
- verdict: `stage42_hg_teacherless_claim_linter_pass`
- gates: `15 / 15`
- scanned files: `18`; violations: `0`.
- allowed phrase: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked: global floor-free neural deployment, causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C, and SMC.
- role: applies Stage42-HF contract to the paper/README surface; this is not new training or threshold tuning.
<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:END -->
