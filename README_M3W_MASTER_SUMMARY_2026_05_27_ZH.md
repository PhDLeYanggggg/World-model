# M3W 当前目标总总结：尝试路线、失败原因、成功证据与真实质量

更新时间：2026-05-27  
项目名：M3W: Real-World Multimodal Agent-Scene World Model  
结果来源：`cached_verified_summary`，汇总已有 Stage18 到 Stage42-IB 的报告、gate、README、model/data card、`research_state.json` 与最近提交记录。  
本 README 是总结文件，不是新训练、不是新评估、不是新转换、不是新下载。

## 0. 真实边界

当前 M3W 不能被包装成以下任何一种东西：

- 不是 true 3D world model。
- 不是 large-scale foundation world model。
- 不是 global metric prediction model。
- 不是 seconds-level long-horizon predictor。
- 不是 ungated neural dynamics deployment。
- 不是 Stage5C latent generative rollout。
- 不是 SMC。

当前真实定位是：

```text
M3W = protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

关键限制必须一直保留：

- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external ETH/UCY/TrajNet/OpenTraj 结果主要是 dataset-local / unverified weak-metric diagnostic，不能混写成统一物理米制结果。
- t+50 / t+100 是 raw-frame horizon；effective seconds、FPS/stride、homography、scale 没有全局验证。
- future endpoint / future waypoint 只允许作为 supervised label 或 evaluation label，不能作为 inference input。
- central velocity 不作为 official input。
- test endpoints 不用于构建 goals。
- self-audited / visual-prior / auto-silver labels 不是 human gold。
- 当前部署仍依赖 conservative fallback / safety floor；无保护 neural dynamics 仍不安全。

## 1. 一句话结论

这个长期目标内真正完成的是：把项目从 SDD-only selector scaffold，推进到一个受保护的外部 dataset-local raw-frame 2.5D 多智能体 world-state 候选系统。

成功主线不是 JEPA-only，也不是无保护 Transformer，而是：

```text
causal feature store
+ expected-FDE / gain-harm selector
+ conservative fallback
+ external causal history windows
+ scene-agnostic goal prototypes
+ protected neural endpoint / composite-tail dynamics
+ protected full-waypoint / group-consistency policy
+ strict no-leakage and claim guards
```

当前最强分层：

| 场景 | 当前 best deployable / candidate | 状态 |
| --- | --- | --- |
| SDD pixel raw-frame benchmark | Stage26 cost-aware failure-assisted selector | best deployable SDD baseline |
| external t+50 dataset-local transfer | Stage37 causal-history + goal-prototype safe selector | deployable external t50 selector |
| protected neural world-state | M3W-Neural v1 composite-tail safe-switch bounded dynamics | candidate, not ungated neural |
| protected full-waypoint / group-consistency | Stage42 protected source/domain full-waypoint family | strong protected 2.5D evidence |
| t100 | Stage42-HR/HS/HT/HV raw-frame t100 easy-guard replay | diagnostic only, not seconds-level |
| metric/time | restricted queue ready candidates = 0 | blocked |
| Stage5C | not executed | forbidden |
| SMC | not enabled | forbidden |

## 2. 主要阶段做了什么

| 阶段 | 做了什么 | 结果 | 结论 |
| --- | --- | --- | --- |
| Stage18 | SAM-JEPA-2.5D representation pretraining | non-collapse，但 selector / failure / correction / official t+50 无 downstream lift | JEPA 不能作为主贡献 |
| Stage19 | WAM-style data registry、simulation/video/top-down 数据策略 | 建立数据方向，但不把 simulation/video 包装成 official trajectory success | 正确方向是补真实 top-down 数据 |
| Stage20-21 | Web acquisition + SDD/OpenTraj 数据准备；SDD 转 world-state shards | SDD 8 scenes / 60 videos / 10,300 tracks / 10.6M rows；no-leakage pass | SDD 成为 pixel raw-frame official benchmark |
| Stage22 | SDD scene packs、episodes、GoalBench、HardBench、BaselineFailureBench、strong baselines | strongest causal baseline = damped_velocity；existing model transfer 无 learned improvement | 需要 SDD-specific selector |
| Stage23 | medium 目标因旧 NPZ I/O 太慢降级 quick-plus | selector +2.66%，failure AUROC 0.6498，JEPA 无 lift | quick-plus 不能当 medium |
| Stage24 | 修 I/O、fast cache、true medium index | I/O 加速约 12.66x；600k medium index；oracle headroom 46.2%；hard-class selector -43.3%；failure AUROC 0.8715 | 问题是 selector 任务定义，不是数据量 |
| Stage25 | selector failure forensics、regret/fallback 策略 | 定位 low-margin label、easy over-switch、hard classification 问题 | 必须做 cost-aware expected-FDE selector |
| Stage26 | feature-complete cost-aware selector | t50 +14.58%，hard/failure +11.23%，easy degradation 1.81% | SDD best deployable 成立 |
| Stage31 | external feature store + zero-shot transfer | SDD->external all -92.67%，t50 -278.57%；adapted 0 | 外部 domain gap 很大 |
| Stage32 | normalization / latent alignment | adapted 仍 0；mixed-domain 会伤 SDD easy | 普通 normalization 和 latent distance alignment 不够 |
| Stage33 | coordinate-invariant features / relative targets | 仍主要 safe fallback | 缺逐行几何与 goals |
| Stage34 | external row geometry + train-only goals | t50 / hard 有局部正信号，但 all 负、easy 高 | 不可部署 |
| Stage35 | data expansion + hard/easy/failure labels + selective transfer | all +12.13%，hard +13.98%，easy 0.041%，但 t50 = 0 | t50 是唯一阻塞 |
| Stage36 | t50-specific policy/forensics/curriculum | t50 仍 0；oracle headroom 22.98% | 缺 past-only history/prototype context |
| Stage37 | history windows + scene-agnostic goal prototypes + conformal safe selector | all +13.48%，t50 +8.46%，CI [+7.69%, +9.15%]，hard +15.54%，easy 0.041%，gates 16/16 | external t50 修复成功 |
| Stage38 | freeze Stage37；尝试 bounded correction / multi-domain audit | correction 不超过 Stage37；ETH/TrajNet 仍有 blocker | Stage37 保持 external best |
| Stage39 | Transformer / JEPA / Hybrid neural dynamics under Stage37 floor | neural 没超过 Stage37；JEPA non-collapse 但无 lift | 不部署 neural |
| Stage40 | 5-10 trial neural optimization | best neural 等于 Stage37 subset；without fallback 灾难性失败 | Stage37 仍 best deployable |
| Stage41 | M3W-Neural v1 protected candidate package | protected composite-tail neural candidate，gates 41/41 | protected neural candidate，不是 ungated dynamics |
| Stage42 | source/domain/full-waypoint/group-consistency/claim guards/metric-time guards | 多个 protected full-waypoint / group-consistency 分支通过；context 独立主 claim 被关闭；metric/time ready=0 | 形成 protected 2.5D paper package |
| Stage42-HZ/IA/IB | source terms confirmation packet、HZ->CG bridge、bridged dry-run validator | HZ gate 22/22，IA gate 17/17，IB gate 16/16；conversion_ready = 0 | legal/source intake 链路打通，但用户确认前不转换不评估 |

## 3. 成功路线与具体证据

### 3.1 Stage26：SDD cost-aware selector

Stage24 的 hard-class selector 明明有 46.2% oracle headroom，却 t50 -43.3%，easy degradation 11.33%。原因是直接学习 one-hot best-baseline label 会在低 margin 样本上过拟合，并且不会惩罚伤害 easy cases。

Stage26 改成 expected-FDE / gain-harm / fallback-safe selector：

| 指标 | 结果 |
| --- | ---: |
| t+50 improvement | +14.58% |
| hard/failure improvement | +11.23% |
| easy degradation | 1.81% |
| correction specialist | not trained |
| Stage5C | false |
| SMC | false |

成功原因：

- 输入来自过去：speed、acceleration、heading change、curvature、density、nearest neighbor、TTC、agent type、horizon、split type、goal distance、baseline rollout diagnostics。
- 不再预测单一 best class，而是预测 expected FDE / risk / gain / harm。
- confidence low、predicted gain 小、easy risk 高时回退 strongest baseline。
- 使用 Stage24 已过 gate 的 failure predictor 作为辅助。

### 3.2 Stage37：external t50 safe selector

Stage35 已经 all/hard 过了，但 t50 = 0。Stage36 证明 t50 有 16,263 rows 和 22.98% oracle headroom，说明不是没有可学空间，而是现有特征不足以支持安全切换。

Stage37 增加：

- K=8/16/32/64 past-only history windows。
- scene-agnostic goal prototypes：straight_continue、slow_stop、left_turn、right_turn、group_follow、density_avoid 等。
- t50-specific failure / gain / harm predictors。
- conformal safety rule，validation 校准 easy degradation 和 harm_over_fallback。

最终结果：

| 指标 | 结果 |
| --- | ---: |
| external rows | 66,303 |
| all improvement | +13.48% |
| t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | stage37_t50_transfer_repaired_deployable |

这是第一条 external t50 可部署正迁移路线。

### 3.3 M3W-Neural v1：protected neural candidate

Stage39/40 的无保护 neural 没有超过 Stage37；Stage41 才把路线改成 Stage37/teacher floor 保护下的 self-gated endpoint / composite-tail bounded neural dynamics。

M3W-Neural v1 package 记录：

| 指标 | 结果 |
| --- | ---: |
| gates | 41 / 41 |
| all improvement vs Stage37 floor | +21.03% |
| t+50 improvement vs Stage37 floor | +13.65% |
| t+100 raw-frame diagnostic improvement | +14.69% |
| hard/failure improvement | +20.38% |
| easy degradation | 0.00% |
| positive external domains | 3 |
| bootstrap evidence | pass |
| multiseed replication | pass |

另一个 fresh self-gated endpoint candidate 曾达到：

| 指标 | 结果 |
| --- | ---: |
| all improvement vs floor | +41.96% |
| t+50 improvement | +40.62% |
| t+100 raw diagnostic | +45.73% |
| hard/failure | +43.61% |
| easy degradation | 0.00% |

但必须强调：这是 protected neural evidence，不是无保护 neural replacement，不是 latent rollout，不是 Stage5C。

### 3.4 Stage42：protected full-waypoint / group-consistency

Stage42 的核心价值是把 endpoint selector 成功扩展到 protected full-waypoint / source-level / group-consistency world-state evidence，并同步建立 claim guards。

代表结果：

| 节点 | 指标 | 结论 |
| --- | --- | --- |
| Stage42-AM | all +24.58%，t50 +22.02%，t100 raw +14.37%，hard +23.75% | source-level full-waypoint protected probe 成立 |
| Stage42-FE | all/t50/hard +26.41% / +23.15% / +24.81%，near@0.05 = 1.32% | constrained FC-to-DI safety composer 成立 |
| Stage42-FH | all/t50/t100raw/hard +34.98% / +28.97% / +20.57% / +33.10% | 修复 UCY weak-domain，dual-domain positive-safe |
| Stage42-FI | CI low all/t50/t100raw/hard +34.62% / +28.46% / +19.96% / +32.73% | frozen replay + 2000-bootstrap，非偶然 |
| Stage42-DJ/DM | all/t50/t100raw/hard +24.72% / +22.36% / +14.35% / +23.89%，near@0.05 1.94% -> 1.38% | group-consistency runtime policy 成立 |
| Stage42-HR | all/t50/t100raw/hard +27.72% / +26.99% / +6.79% / +25.93%，t100 easy degradation -0.31% | 修复 HQ 暴露的 t100 easy harm |
| Stage42-HS | policy hash 固化，decision/metric replay exact，gate 27/27 | frozen policy 可复放 |
| Stage42-HT | runtime API gate 19/19 | policy 可调用 |
| Stage42-HV | 47,458 test rows row-cache replay，selected XY/ADE/switch/metrics exact，gate 28/28 | 本地真实 row-level batch replay 成立 |

Stage42 的真实含义：

- 可以写 protected dataset-local raw-frame 2.5D full-waypoint / group-consistency world-state evidence。
- 不能写 true 3D、foundation、metric、seconds-level、Stage5C 或 SMC。
- t100 只能写 raw-frame diagnostic，即使 HR/HV 修复了 t100 easy harm。

## 4. 失败路线与原因

### 4.1 Hard-class selector

失败表现：

- Stage24 selector t50 improvement = -43.3%。
- easy degradation = 11.33%。

原因：

- oracle best baseline label low-margin、噪声高。
- one-hot best class 不表达“错切换的成本”。
- 没有 predicted gain / harm / confidence / easy guard。

修复：

- Stage25/26 改成 cost-aware expected-FDE selector。

### 4.2 JEPA downstream 主线

失败表现：

- Stage18 JEPA non-collapse，但 selector / failure / correction / official t50 无 lift。
- Stage23/24/39 多次 JEPA probe 无稳定 downstream lift。

原因：

- latent variance non-collapse 不等于 selector/failure/trajectory 有用。
- JEPA target 没有对齐 gain/harm/switchability。
- 当前任务主要由 causal history、baseline-family rollout、safety floor 驱动。

结论：

- JEPA 保留为 diagnostic / auxiliary representation。
- 不能作为当前主贡献。
- 不能写成生成式 world model。

### 4.3 SDD -> external zero-shot

失败表现：

- Stage31 SDD->external all = -92.67%。
- t50 = -278.57%。
- Stage32 adapted selector 约 0 improvement。

原因：

- SDD pixel 与 external dataset-local 坐标不兼容。
- scene/goal/interaction 信息缺失。
- agent type、horizon、track length、scale/homography 不匹配。
- latent distribution alignment 缩小距离不等于 predictive lift。

修复：

- Stage34 补 row geometry / train-only goals。
- Stage35 selective transfer 修 all/hard。
- Stage37 history/prototype/conformal 修 t50。

### 4.4 Ordinary residual / correction

失败表现：

- Stage22-24 correction 没有可靠改善 hard/failure。
- Stage38 bounded correction 没超过 Stage37。
- 无保护 residual 常常伤 easy。

原因：

- residual 容易 over-correct easy cases。
- 没有 reliable gain/harm gate 时，correction 会破坏 strongest/floor baseline。

结论：

- 当前 correction 不能作为 deployable 主模型。
- 只能 bounded / protected / diagnostic。

### 4.5 Unprotected Transformer / Hybrid neural dynamics

失败表现：

- Stage39 Transformer/JEPA/Hybrid trained，但 neural_with_fallback 没超过 Stage37。
- Stage40 neural_without_fallback 灾难性失败：
  - all improvement -126.36%
  - t50 -292.10%
  - easy degradation +612.31%

原因：

- neural 学到部分 dynamics，但没有学会“不要切错”。
- 没有 safety floor 时 easy cases 被严重破坏。
- 模型容易复制 Stage37 或被 fallback 吃掉。

修复：

- Stage41 改成 self-gated endpoint / composite-tail protected neural candidate。
- 仍不能部署无保护 neural dynamics。

### 4.6 Scene/goal 与 neighbor/interaction 独立主贡献

失败表现：

- Stage42-CJ goal/scene gated expert 被 validation 选回 baseline_family_control。
- Stage42-CK neighbor/interaction gated expert 也被选回 baseline_family_control。
- Stage42-AQ/AR/AS neural/sequence/graph residual context 均未超过 baseline-family first-stage。

原因：

- 当前 hand-built scene/goal/neighbor features 太弱。
- 当前 residual target 不是有效 context learning target。
- baseline-family rollout context 已解释大部分可用 gain。

结论：

- scene/goal/neighbor/interaction 目前只能写辅助或 diagnostic。
- 不能写成独立主贡献。

### 4.7 Metric/time / source legal conversion

失败或 blocked 表现：

- Stage42-HM/HN/HZ/IA/IB 都显示 restricted metric/time/source conversion ready = 0。
- HZ packet 有 local path 和 parseable hints，但 terms accepted = 0。
- IA bridge structurally pass，但 ready-if-activated = 0。
- IB dry-run validator pass，但 conversion_ready_targets = 0。

原因：

- local path found 不等于 legal terms accepted。
- OpenTraj toolkit license 不自动等于所有 dataset 权限。
- 官方 source identity、allowed use、terms acceptance 需要用户确认。

结论：

- 不能自动转换、不能自动评估、不能写 metric/seconds claim。
- 下一步必须由用户确认 source terms / local path / allowed use。

## 5. 当前 best deployable 分层

```text
SDD:
  Stage26 cost-aware failure-assisted selector

External t50:
  Stage37 causal-history + goal-prototype + conformal safe selector

Protected neural/world-state:
  M3W-Neural v1 composite-tail safe-switch under Stage37/teacher floor

Protected full-waypoint/group-consistency:
  Stage42 source/domain protected group-consistency/full-waypoint family
  Stage42-HR/HS/HT/HV t100 easy-guard runtime/replay family

Not deployable:
  hard-class selector
  JEPA-only
  latent-only alignment
  ordinary residual/correction
  unprotected Transformer/Hybrid
  floor-free neural deployment
  metric/seconds-level conversion
  Stage5C
  SMC
```

## 6. 当前大概是什么质量

比较诚实的定位：

```text
protected 2.5D cross-dataset world-state candidate
with strong safety-gated selector/full-waypoint evidence
but not a true 3D or foundation world model
```

可以作为论文候选材料的部分：

- Stage26 SDD cost-aware selector。
- Stage37 external t50 safe transfer。
- M3W-Neural v1 protected neural candidate。
- Stage42 protected full-waypoint / group-consistency source-level evidence。
- Stage42 claim boundary / no-leakage / replay / bootstrap / runtime policy package。

不能作为主 claim 的部分：

- JEPA 独立贡献。
- 无保护 Transformer / Hybrid。
- scene/goal 或 neighbor/interaction 独立主贡献。
- metric/seconds-level prediction。
- true 3D / foundation。
- Stage5C / SMC。

如果按 A刊/CCF-A 证据链看：

- 已经有 serious protected 2.5D external world-state manuscript package 的材料。
- 还不够 broad true-3D/foundation/world-model claim。
- 最短缺口是：更多 legally verified external domains、metric/time calibration、真正正向的 JEPA/Transformer ablation、减少 teacher/floor 依赖且不破坏 proximity safety。

## 7. 下一步最值得做

1. 完成 source terms / local path / allowed use 用户确认  
   当前 HZ/IA/IB 已把确认包、桥接模板、dry-run validator 都打通，但 conversion_ready = 0。必须用户确认后才能做 guarded conversion。

2. 在确认后的 external source 上做 guarded conversion + no-leakage + source-CV  
   目标是补 ETH/UCY/TrajNet/UCY 的独立外部源，减少 source-concentration caveat。

3. 做 metric/time 校准，但只在证据充分时写 weak metric / seconds-level  
   没有 verified homography/FPS/stride 前，继续 raw-frame / dataset-local wording。

4. 如果继续神经模型，必须换目标而不是重复残差  
   当前 sequence/graph residual context 已关闭；若重启，应做 gain/harm/switchability 或更完整 sequence architecture，并保持 Stage37/floor safety。

5. 继续保留 Stage37 / Stage42 protected policies 作为安全地板  
   不允许无保护 neural 替代，直到 easy/proximity/no-leakage gates 全过。

## 8. 本 README 对应的重要证据文件

- `README_RESULTS.md`
- `README_M3W_CURRENT_MASTER_SUMMARY_ZH.md`
- `README_M3W_RESEARCH_ROUTES_FAILURES_SUCCESSES_2026_05_27_ZH.md`
- `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- `outputs/stage42_long_research/a_journal_gap_stage42.md`
- `outputs/stage42_long_research/failure_taxonomy_stage42.md`
- `outputs/stage42_long_research/context_model_closure_stage42.md`
- `outputs/stage42_long_research/source_terms_confirmation_packet_stage42.md`
- `outputs/stage42_long_research/source_terms_hz_to_cg_intake_bridge_stage42.md`
- `outputs/stage42_long_research/source_terms_ia_bridged_validator_dry_run_stage42.md`

## 9. 最终当前结论

项目是否跑通：是，作为 protected dataset-local / raw-frame 2.5D world-state candidate 跑通。  
是否是真 3D：否。  
是否是 foundation：否。  
是否是 metric / seconds-level：否。  
Stage5C 是否执行：否。  
SMC 是否启用：否。  
当前 best deployable：分层部署，SDD 用 Stage26，external t50 用 Stage37，protected neural/full-waypoint 用 M3W-Neural v1 + Stage42 protected policies。  
当前最大阻塞：legal/source confirmation、external source diversity、metric/time calibration、JEPA/Transformer 独立 downstream lift、floor-free safety。

<!-- STAGE42_IC_CURRENT_CLAIM_EVIDENCE_CLOSURE:START -->
## Stage42-IC Current Claim / Evidence Closure

- source: `fresh_stage42_ic_current_claim_evidence_closure`
- verdict: `stage42_ic_current_claim_evidence_closure_pass`; gates `16 / 16`.
- supported claims: `6`; blocked/diagnostic claims: `7`.
- t100 row replay rows: `47458`; source terms conversion-ready now: `0`.
- IC closes the current paper-package claim map: supported claims remain protected dataset-local/raw-frame 2.5D, while true-3D/foundation/metric-seconds/Stage5C/SMC and JEPA/Transformer independent-main claims remain blocked.
- This is not new training, download, conversion, or evaluation; it is a claim/evidence closure over existing fresh/cached_verified artifacts.
<!-- STAGE42_IC_CURRENT_CLAIM_EVIDENCE_CLOSURE:END -->

<!-- STAGE42_ID_PAPER_CLAIM_CONTRACT:START -->
## Stage42-ID Paper Claim Contract

- source: `fresh_stage42_id_paper_claim_contract`
- verdict: `stage42_id_paper_claim_contract_pass`; gates `15 / 15`.
- contract rows: `13`; supported claims `6`; blocked claims `7`.
- paper files existing: `8 / 8`; files with raw/dataset caveat `8`; files with Stage5C/SMC boundary `8`.
- ID locks manuscript wording: supported claims are protected dataset-local/raw-frame 2.5D; true-3D/foundation/metric-seconds/Stage5C/SMC claims remain forbidden.
- This is a paper-claim contract only, not new training, conversion, download, or evaluation.
<!-- STAGE42_ID_PAPER_CLAIM_CONTRACT:END -->

<!-- STAGE42_IE_PAPER_CONTRACT_COMPLIANCE:START -->
## Stage42-IE Paper Contract Compliance

- source: `fresh_stage42_ie_paper_contract_compliance`
- verdict: `stage42_ie_paper_contract_compliance_pass`; gates `14 / 14`.
- paper files checked: `9 / 9`.
- supported anchors covered: `5 / 5`; unbounded overclaim hits: `0`.
- blocked claims covered as limitations: `7 / 7`.
- IE verifies the current paper package obeys the Stage42-ID contract: protected dataset-local/raw-frame 2.5D only; no true-3D/foundation/metric-seconds/Stage5C/SMC overclaim.
- This is compliance verification only, not new training, conversion, download, or evaluation.
<!-- STAGE42_IE_PAPER_CONTRACT_COMPLIANCE:END -->

<!-- STAGE42_IF_T50_GAIN_HARM_STABILITY_AUDIT:START -->
## Stage42-IF T50 Gain/Harm Stability Audit

- source: `fresh_stage42_if_t50_gain_harm_stability_audit`
- verdict: `stage42_if_t50_gain_harm_ci_blocker_identified`
- gates: `13 / 14`
- ADE t50 mean / CI low: `0.006596` / `-0.017931`
- FDE t50 mean / CI low: `0.057431` / `0.046360`
- negative ADE t50 seeds: `1`
- validation-selected seed test ADE t50: `0.028352`
- row bootstrap status: `not_run_blocked_by_missing_row_errors_in_stage42p_artifact`
- conclusion: Stage42-P is positive on mean t+50 and stable on FDE t+50, but ADE t+50 is not yet seed-CI stable enough for a paper-level t+50 ADE claim.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IF_T50_GAIN_HARM_STABILITY_AUDIT:END -->

<!-- STAGE42_IG_T50_GAIN_HARM_ROW_BOOTSTRAP:START -->
## Stage42-IG T50 Gain/Harm Row Bootstrap

- source: `fresh_stage42_ig_t50_gain_harm_row_bootstrap`
- verdict: `stage42_ig_row_bootstrap_validates_selected_seed_with_multiseed_blocker`
- gates: `15 / 15`
- validation-selected seed: `151`
- selected ADE t50 / CI low: `0.028352` / `0.023371`
- selected FDE t50 / CI low: `0.067566` / `0.060976`
- selected ADE hard/failure: `0.054677`
- selected ADE easy degradation: `0.007574`
- multiseed ADE t50 CI low remains: `-0.017931`
- conclusion: validation-selected row-level t+50 evidence is positive, but seed-stable ADE t+50 remains an open blocker.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IG_T50_GAIN_HARM_ROW_BOOTSTRAP:END -->
