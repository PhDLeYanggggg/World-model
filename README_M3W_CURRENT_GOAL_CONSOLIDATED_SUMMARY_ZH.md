# M3W 当前长期目标总账：路线、失败、成功与当前质量

更新时间：2026-05-27

工作目录：`/Users/yangyue/Downloads/World`

本文用途：按用户要求，把 M3W 长期目标内已经做过的事情、尝试过的路线、失败原因、成功证据、当前 best deployable、当前质量判断和下一步最短路径集中到一个 README。本文是总结与证据索引，不是新训练结果；凡读取旧结果均按 `cached_verified` 处理，不把 cached 写成 fresh，不把 not_run 写成完成。

## 1. 最高层结论

M3W 当前已经从早期的 2.5D trajectory scaffold，推进为一个有 SDD、external top-down pedestrian、source/domain protected policy、runtime replay、bootstrap、no-leakage、claim-boundary linter 支撑的 **protected dataset-local / raw-frame 2.5D multi-agent world-state candidate**。

但必须继续诚实承认：

- 不是 true 3D world model。
- 不是 large-scale foundation world model。
- 不是 global metric / meter-level predictor。
- 不是 seconds-level long-horizon predictor。
- SDD 仍是 pixel-space benchmark。
- external 仍是 dataset-local / unverified weak-metric diagnostic。
- t+50 / t+100 仍是 raw-frame horizon。
- self-audited / visual-prior / inferred labels 不是 human gold。
- JEPA 不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。

当前最强可部署分层：

| 用途 | 当前 best | 关键结果 | 当前结论 |
| --- | --- | --- | --- |
| SDD pixel raw-frame | Stage26 cost-aware selector | t+50 约 `+14.58%`，hard/failure 约 `+11.23%`，easy degradation 约 `+1.81%` | SDD 内 best deployable；不是 metric/seconds |
| external t+50 | Stage37 causal-history + goal-prototype safe selector | all `+13.48%`，t50 `+8.46%`，t50 CI `[+7.69%, +9.15%]`，hard/failure `+15.54%`，easy `0.041%`，gates `16/16` | external selector-level deployable |
| protected neural | M3W-Neural v1 composite-tail safe-switch under Stage37/teacher floor | all `+21.03%`，t50 `+13.65%`，t100 raw diagnostic `+14.69%`，hard/failure `+20.38%`，easy degradation `0` | protected neural evidence；不部署 ungated neural |
| source/domain protected policy | Stage42-FH/FI frozen protected policy family | all/t50/t100raw/hard `34.98% / 28.97% / 20.57% / 33.10%`; FI CI low `34.62% / 28.46% / 19.96% / 32.73%`; exact replay diff `0` | stronger source/domain protected family；不能写 uniform horizon |
| runtime/reviewer replay | Stage42-DM / DL replay path | rows `47458`; switch exact match `true`; all/t50/t100raw/hard `+24.72% / +22.36% / +14.35% / +23.89%`; near@0.05 `1.94% -> 1.38%` | 可复放证据强 |
| paper claim boundary | Stage42-FU/FV/GJ/GK/GS | claim lock / linter / gap reconciler passed | 支持 protected source-level 2.5D claim；阻止 JEPA/Transformer/context 过度主 claim |

一句话：**当前最有价值的成果不是“无保护大模型”，而是以强 causal floor 为底座、通过 cost-aware / gain-harm / safe-switch / group-consistency 保护的 2.5D 多智能体 world-state policy family。**

## 2. 我尝试过的主要路线

### 2.1 数据与 benchmark 路线

做过：

- 建立 SDD world-state shards、scene packs、lazy episodes、HardBench、BaselineFailureBench、GoalBench。
- 建立 external OpenTraj / ETH-UCY / TrajNet / UCY 相关 feature store、row geometry、source-level split、train-only goals、history windows、goal prototypes。
- 建立 no-leakage audit、horizon audit、raw-frame / dataset-local claim guard。
- 扫描 source/legal/calibration blocker，并生成 user action / intake / source confirmation / legal gate 包。

成功：

- SDD 成为 pixel raw-frame official benchmark。
- external t50 在 Stage37 修复为 deployable selector-level positive transfer。
- Stage42 source/domain protected policy 形成更强 source-level raw-frame evidence。

失败或未完成：

- external metric/seconds claim 没有完成。原因是 homography direction、scale、annotation stride、source-specific timing 和 legal source terms 未全局确认。
- external broader source expansion 仍被 terms/path/source identity 阻塞。Stage42-GS 明确 `source_legal_conversion` 仍是 open blocker，contract-ready / auto-download-ready 都不能当作 completed conversion。

### 2.2 强因果 baseline 与 fallback 路线

做过：

- 系统比较 constant_position、constant_velocity_causal_fd、damped_velocity、constant_acceleration、turn_rate、scene_clamped、goal/prototype-directed 等 causal baselines。
- 所有 learned model 都必须超过 strongest causal baseline / Stage26 / Stage37 floor，并保持 easy 不坏。

成功：

- 这条路线最终成为整个 M3W 的安全骨架。
- Stage26、Stage37、Stage42-FH/FI、M3W-Neural v1 都依赖这个保护机制。

原因：

- 真实多智能体轨迹里 easy cases 很多，强 baseline 很难打。
- 如果 learned model 在 easy cases 上乱切换，整体部署质量会立即下降。
- 因此核心不是“总是相信模型”，而是“只有在 predicted gain 高、harm 低、confidence 足够时才切换，否则 fallback”。

### 2.3 Selector 路线

做过：

- hard classification selector。
- expected-FDE / regret-aware selector。
- soft-label / margin-aware / hierarchical selector。
- failure-assisted selector。
- conservative fallback selector。
- horizon-specific t50 selector。
- source/domain group-consistency full-waypoint composer。

失败：

- Stage24 hard selector：t+50 improvement 约 `-43.3%`，easy degradation 约 `11.33%`。
- 原因：oracle headroom 大不等于 hard label 可学；low-margin 样本被强制分类，easy cases 被错误切换；selector confidence/calibration 不可靠。

成功：

- Stage26 cost-aware selector 在 SDD 上成功。
- Stage37 t50 专用 safe selector 在 external 上成功。
- Stage42-FH/FI protected source/domain policy 进一步成功。

关键教训：

- 不能把 selector 做成“预测哪个 baseline 最好”的硬分类器。
- 必须预测 expected FDE / gain / harm / risk，并有 fallback。

### 2.4 JEPA 路线

做过：

- Stage18/19/22/23/24/27+ 多轮 JEPA representation pretraining。
- trajectory-only、scene/trajectory、interaction-aware JEPA、masked latent、future latent probe。

结果：

- 多数时候 non-collapse。
- 但 selector / failure predictor / correction / official t50 没有稳定 downstream lift。

失败原因：

- latent variance 不是任务贡献。
- JEPA 目标与实际 deployment objective（gain/harm/easy safety/horizon switchability）错位。
- raw-frame dataset-local 外部数据缺 scene/metric grounding，JEPA 表征不自动变成可部署 dynamics。

当前处理：

- JEPA 只能作为 auxiliary / diagnostic。
- 不能写成主贡献。
- 不能写成 latent generative world model。

### 2.5 Transformer / Hybrid neural dynamics 路线

做过：

- Transformer-only。
- JEPA+Transformer Hybrid。
- causal temporal sequence encoder。
- protected neural dynamics。
- full-waypoint sequence dynamics。
- composite-tail bounded neural dynamics。

失败：

- Stage39/40 普通 Transformer/JEPA/Hybrid 没有超过 Stage37。
- neural without fallback 灾难性或不安全。
- bounded correction 没有安全超过 Stage37。

原因：

- Stage37 floor / teacher policy 太强。
- 直接 residual / correction 容易伤 easy。
- 无保护 neural 输出没有足够稳定的 proximity / collision / smoothness / easy safety。
- 当前数据仍是 dataset-local/raw-frame，不是 metric/scene-grounded 3D。

成功边界：

- M3W-Neural v1 在 Stage37/teacher floor 保护下形成 candidate。
- Stage42-H/I/J/K/L/P 等证明 causal sequence/full-waypoint/gain-harm 有一些真实信号。
- 但不能说“ungated neural dynamics 已成功部署”。

### 2.6 External cross-domain 路线

做过：

- Stage31 external transfer。
- Stage32 domain normalization / latent alignment。
- Stage33 coordinate-invariant features / relative targets。
- Stage34 row geometry / train-only goals / relative-error baselines。
- Stage35 selective transfer。
- Stage36 horizon-specific t50 repair。
- Stage37 past-only history window + scene-agnostic goal prototypes + t50 safe selector。

失败：

- Stage31 SDD -> external zero-shot 崩：all `-92.67%`，t50 `-278.57%`。
- Stage32 normalization/adaptation 后仍没有 positive transfer。
- latent adapter 缩小分布距离，但没有 predictive lift。
- Stage35 all/hard positive 但 t50 仍 `0.0`。

原因：

- coordinate / scale / horizon / agent-type / scene-goal context 不兼容。
- held-out scene 缺 train-scene goals。
- external t50 需要 long-horizon switchability，不是普通 all-test objective 能学好。

成功：

- Stage37 修复 external t50：all `+13.48%`，t50 `+8.46%`，hard/failure `+15.54%`，easy `0.041%`，t50 bootstrap CI `[+7.69%, +9.15%]`。

关键教训：

- 跨域不是靠普通 normalization。
- 需要 past-only history window、scene-agnostic goal prototypes、horizon-specific gain/harm、conformal/easy-safe fallback。

### 2.7 Full-waypoint / group-consistency 路线

做过：

- Endpoint-only bridge。
- Full-waypoint sequence dynamics。
- Static-gated full-waypoint repair。
- Horizon-aware static gate。
- Row-level gain/harm full-waypoint selector。
- Group-consistency full-waypoint runtime policy。
- Proximity-aware composer guard。

失败：

- Ungated full-waypoint neural easy degradation 极高。
- 全局 full static context 会伤害 protected ADE。
- 一些 scalar proximity/occupancy / repel / risk repair 在 ADE 和 proximity 之间产生 tradeoff。

成功：

- Stage42-CO common-validation bridge/shape composer：all `+3.02%`，t50 `+1.50%`，t100raw `+6.12%`，hard `+3.28%`。
- Stage42-CP 2000-bootstrap positive。
- Stage42-CQ proximity guard：all `+1.77%`，t50 `+1.07%`，t100raw `+3.48%`，hard `+1.93%`，near@0.05 不劣于 endpoint-linear。
- Stage42-DM runtime replay：rows `47458`，switch exact match `true`，near@0.05 `1.94% -> 1.38%`。
- Stage42-FE/FH/FI 形成 source/domain robust protected policy。

当前边界：

- Full-waypoint shape / endpoint bridge / group-consistency 可以写成 protected bounded components。
- 不能写成 floor-free global dynamics。

## 3. 失败路线总表

| 路线 | 失败表现 | 根因 | 当前状态 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 t50 `-43.3%`，easy `11.33%` | low-margin oracle label、class imbalance、confidence bad、easy 被错切 | 已废弃，改 expected-FDE / gain-harm |
| JEPA 主线 | non-collapse 但 downstream 无稳定 lift | pretraining target 和 deployment objective 错位 | auxiliary/diagnostic |
| zero-shot SDD->external | all `-92.67%`，t50 `-278.57%` | coordinate/scale/horizon/scene mismatch | 用 Stage37 history/prototype 修复 external |
| latent adapter | MMD/CORAL 距离下降但预测无提升 | distribution alignment 不等于 task alignment | 不作为主 claim |
| bounded residual / ordinary correction | hard/easy 不稳，未超过 Stage37 | 直接改轨迹易伤 easy | 不部署 |
| unprotected Transformer/Hybrid | 不超过 Stage37 或不安全 | floor 太强，数据 grounding 不足 | 只允许 protected neural evidence |
| scene/goal 独立主 claim | goal/scene candidates 低于 baseline-family control | current scene/goal features 不够 material | diagnostic only |
| neighbor/interaction 独立主 claim | scalar/kNN graph 低于 baseline-family control | current graph/context 表达不够 | diagnostic only |
| scalar proximity/occupancy objective | 局部提升但 safety/accuracy tradeoff | scalar target 表达不了完整 group dynamics | 采用 constrained guard |
| uniform horizon robustness | TrajNet|100 / UCY|100 weak | source support / long-horizon context / low-margin ambiguity | blocked，需 legal source support |
| metric/seconds claim | 无全局 homography/stride/scale verified | 证据不足 | 禁止 |
| external source expansion | conversion_ready_now `0` | terms/path/source identity 未确认 | 只能 user action/intake |

## 4. 成功路线总表

| 成功路线 | 证据 | 能写什么 | 不能写什么 |
| --- | --- | --- | --- |
| Stage26 SDD cost-aware selector | t50 `+14.58%`，hard `+11.23%`，easy `+1.81%` | SDD pixel raw-frame selector success | metric / true 3D |
| Stage37 external t50 safe selector | all `+13.48%`，t50 `+8.46%`，CI `[+7.69%, +9.15%]`，hard `+15.54%`，easy `0.041%` | external dataset-local t50 deployable selector | foundation / metric / seconds |
| M3W-Neural v1 protected | all `+21.03%`，t50 `+13.65%`，hard `+20.38%`，easy `0` | protected neural candidate | ungated neural deployment |
| Stage42-CO/CP bridge-shape composer | all `+3.02%`，t50 `+1.50%`，t100raw `+6.12%`，hard `+3.28%`，bootstrap positive | protected bridge/full-waypoint auxiliary evidence | global floor-free full-waypoint |
| Stage42-CQ proximity guard | all `+1.77%`，t50 `+1.07%`，near@0.05 no worse than endpoint-linear | safety-sensitive protected composer | accuracy-only overclaim |
| Stage42-DM replay | 47,458 rows exact runtime replay; near@0.05 improves | reviewer/runtime reproducibility | new training claim |
| Stage42-FH/FI | all/t50/t100raw/hard `34.98% / 28.97% / 20.57% / 33.10%`; CI low positive; replay diff 0 | dual-domain/source protected positive-safe family | uniform horizon claim |
| Stage42-FU/GJ module lock | allowed: history/domain/safe-switch/teacher-floor/group-consistency/full-waypoint/endpoint-bridge | clean paper contribution boundary | JEPA/Transformer/context as independent main contribution |
| Stage42-GS gap reconciler | stale gaps reconciled; open blockers explicit | current gap state clean | source/legal conversion completed |

## 5. 当前 paper / A-journal 质量判断

当前可以支撑的是：

- 一个严肃的 protected 2.5D external world-state dynamics manuscript package。
- 有 fresh/cached-verified no-leakage、bootstrap、runtime replay、claim linter、module ledger、paper gap reconciler。
- 有 SDD、external t50、source/domain protected full-waypoint/group-consistency 的实证链。

当前还不能支撑的是：

- true 3D claim。
- foundation model claim。
- global metric / seconds-level horizon claim。
- ungated neural dynamics claim。
- JEPA/Transformer 作为主贡献 claim。
- broad source-level generalization without terms-confirmed independent external sources。
- Stage5C 或 SMC claim。

一句话质量评级：

```text
当前是“protected 2.5D multi-agent world-state candidate / strong engineering-research evidence package”。
它有投稿候选材料雏形，但仍缺 source/legal expansion、metric/time calibration、floor-free safety、JEPA/Transformer independent contribution。
```

## 6. 当前最短下一步

1. **先完成 source/legal confirmation。**
   Stage42-GS 仍显示 `source_legal_conversion` 是 open blocker。必须由用户确认 official terms、allowed use、local path、source identity。否则不能转换、不能评估、不能写新 external evidence。

2. **只对 legally ready sources 做 guarded conversion。**
   转换后必须重新跑 no-leakage、baseline、source-CV、validation-only policy selection、final test once。不能把 registry / prefill / local parseability 写成 converted/evaluated result。

3. **用新增 source support 重新修 h100 / uniform horizon。**
   TrajNet|100、UCY|100 不是继续调当前 threshold 能解决的问题；需要 source support、long-horizon evidence、calibration/terms 关闭后再做。

4. **如果继续 neural dynamics，目标必须是 safety-relevant。**
   不训练普通 residual。优先 train gain/harm、group-consistency、source/horizon switchability、full-waypoint consistency，并继续保留 Stage37/teacher floor。

5. **继续维护 claim linter / paper package。**
   每次新增结果都要更新 README_RESULTS、research_state、paper gap、module ledger；不得把 not_run / diagnostic / cached 结果写成 fresh success。

## 7. 证据索引

关键文件：

- `README_RESULTS.md`
- `README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md`
- `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`
- `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- `outputs/stage37_t50_history/report_stage37_final.md`
- `outputs/stage37_t50_history/world_model_gate_stage37.md`
- `outputs/stage40_neural_optimization/report_stage40_final.md`
- `outputs/stage41_breakthrough/report_stage41_final.md`
- `outputs/stage42_long_research/report_stage42_final.md`
- `outputs/stage42_long_research/paper_ready_evidence_matrix_stage42.md`
- `outputs/stage42_long_research/a_journal_gap_stage42.md`
- `outputs/stage42_long_research/paper_gap_reconciler_stage42.md`
- `outputs/stage42_long_research/stage42_stage_gs_gate.md`

## 8. 最后给用户的直接回答

你问“在这个目标内做了什么、尝试了什么路线、哪些失败、哪些成功、原因是什么”。直接回答：

- 我尝试了数据扩展、SDD official benchmark、external cross-domain transfer、cost-aware selector、failure predictor、JEPA、Transformer、Hybrid、bounded correction、full-waypoint sequence、group-consistency、proximity/safety guard、source/legal/calibration guard、paper claim linter。
- 失败最多的是：hard-class selector、JEPA downstream、zero-shot external、latent-only alignment、unprotected neural、ordinary residual/correction、scene/goal/neighbor 独立主贡献、uniform horizon/h100。
- 失败原因不是单一模型太小，而是：强 baseline 很强、easy cases 极易被伤、external 坐标/scale/horizon 不一致、raw-frame 没 metric grounding、低 margin oracle label 很多、source/legal/calibration 未闭合。
- 成功的是：cost-aware/fallback-safe selector、Stage37 external t50 safe selector、Stage42 source/domain protected policy、group-consistency full-waypoint protected evidence、runtime replay、bootstrap、claim-boundary linter。
- 当前 best deployable 不是无保护神经网络，而是 **Stage37/teacher-floor protected M3W policy family**，source/domain 层面由 Stage42-FH/FI/DL/DM/CQ/GS 等证据支撑。
- 当前不能说成 true 3D、foundation、metric、seconds-level、Stage5C 或 SMC。

<!-- STAGE42_GU_FLOOR_RELAXATION_SAFETY_REFRESH:START -->
## Stage42-GU Floor Relaxation Safety Refresh

- source: `fresh_stage42_gu_floor_relaxation_paper_refresh`
- role: propagates Stage42-GT all-agent safety stress evidence into the paper package and guards against floor overclaims.
- input GT verdict: `stage42_gt_floor_relaxation_safety_stress_pass`; input BY/BZ/EN gates passed: `True` / `True` / `True`.
- target union t50 rows: `11538`.
- target union t50 improvement: `28.97%`.
- target union hard/failure improvement: `28.97%`.
- target union easy degradation: `-21.41%`.
- target union near-collision@0.05 delta: `-0.74%`.
- target union jagged-rate delta: `0.00%`.
- Supported claim: narrow validation-backed t50 partial floor relaxation has all-agent safety support for the audited slices.
- Unsupported claims: global floor removal, floor-free neural deployment, teacher/floor context removal, metric/seconds-level prediction, Stage5C execution, and SMC readiness.
- Result source label: `fresh_run` synthesis from already-produced Stage42-BY/BZ/EN/GT artifacts; no new training, no new download, no new conversion, no test threshold tuning.
- Verification after implementation: focused pytest passed; full suite passed with `929 passed`.
<!-- STAGE42_GU_FLOOR_RELAXATION_SAFETY_REFRESH:END -->

<!-- STAGE42_GV_FLOOR_RELAXATION_SOURCE_ROBUSTNESS:START -->
## Stage42-GV Floor Relaxation Source Robustness

- source: `fresh_stage42_gv_floor_relaxation_source_robustness`
- role: source-level all-agent robustness audit for Stage42-GT partial t50 floor relaxation.
- gate: `14 / 14`; verdict `stage42_gv_floor_relaxation_source_robustness_pass_with_source_concentration_caveat`.
- source-safety-positive slices: `['TrajNet|50', 'UCY|50']`.
- source-concentration-limited slices: `['TrajNet|50', 'UCY|50']`.
- broad source-level generalization claim allowed: `False`.
- Claim boundary: major-source support only; not broad source-level generalization, not global floor removal, not floor-free neural, not metric/seconds-level, not Stage5C, not SMC.
<!-- STAGE42_GV_FLOOR_RELAXATION_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_GW_H100_BLOCKER_CLOSURE_DECISION:START -->
## Stage42-GW H100 Blocker Closure Decision

- source: `fresh_stage42_gw_h100_blocker_closure_decision`
- gate: `17 / 17`; verdict `stage42_gw_h100_blocker_closure_decision_pass`
- weak keys: `['TrajNet|100', 'UCY|100']`
- technical support exists count: `1`; legal conversion ready count: `0`; can run repair now count: `0`
- `UCY|100`: local technical candidates exist, but terms/source identity/guarded conversion are not ready; user action required before repair.
- `TrajNet|100`: hard blocker remains because current local TrajNet snippets are too short for raw-frame h100/t100 repair.
- Boundary: no download, no conversion, no training, no evaluation; uniform h100/t100 claim remains blocked; no metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_GW_H100_BLOCKER_CLOSURE_DECISION:END -->

<!-- STAGE42_GX_UCY_H100_CANDIDATE_INTEGRITY:START -->
## Stage42-GX UCY H100 Candidate Integrity Manifest

- source: `fresh_stage42_gx_ucy_h100_candidate_integrity_manifest`
- gate: `17 / 17`; verdict `stage42_gx_ucy_h100_candidate_integrity_manifest_pass`
- UCY candidate files: `6`; existing `6`; target-family candidates `2`.
- parsed rows: `98032`; parsed t100 windows: `11848`; unique hashes `6`.
- This locks file identity/hash/parse stats only. It is not legal permission, not conversion, not evaluation, and not h100 repair.
- `UCY|100` remains terms/source-identity blocked; `TrajNet|100` remains long-source blocked. No metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_GX_UCY_H100_CANDIDATE_INTEGRITY:END -->
