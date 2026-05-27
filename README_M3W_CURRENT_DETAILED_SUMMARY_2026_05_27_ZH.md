# M3W 当前详细总结：尝试路线、失败原因、成功证据与当前质量

更新时间：2026-05-27  
工作目录：`/Users/yangyue/Downloads/World`  
结果来源：`cached_verified` 汇总既有 Stage18 到 Stage42 报告、gate、README 与 `research_state.json`；最近纳入的 fresh 证据包括 Stage42-HE/HF/HG/HI。  
本文性质：总结文件，不是新训练、不是新转换、不是新下载、不是新评估；不会把 cached 结果写成 fresh。

## 0. 当前最诚实结论

M3W 当前已经从早期 SDD-only selector 进化为一个 **protected dataset-local / raw-frame 2.5D multi-agent world-state candidate**。

它现在有较完整的证据链：

- SDD 上有 cost-aware selector 成功结果。
- External top-down pedestrian 数据上有 t+50 safe selector 成功结果。
- Protected neural / full-waypoint / group-consistency 路线有正向 evidence。
- Source/domain/frozen replay/bootstrap/no-leakage/claim linter 都已建立。
- Stage42 已经把 paper claim boundary、deployment contract、teacherless gate wording、restricted metric/time readiness 都机器化记录。

但它仍然不能称为：

- true 3D world model
- large-scale foundation world model
- global metric / meter-level predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative rollout
- SMC-ready system

当前 best deployable 是分层的：

| 使用场景 | 当前最强结果 | 可部署状态 | 必须保留的边界 |
| --- | --- | --- | --- |
| SDD pixel/raw-frame | Stage26 cost-aware selector | 可部署候选 | 只限 SDD pixel/raw-frame，不是 metric。 |
| External t+50 dataset-local/raw-frame | Stage37 causal-history + goal-prototype safe selector | 可部署候选 | external dataset-local/raw-frame，不是 true 3D。 |
| Protected neural/world-state | M3W-Neural v1 composite-tail safe-switch bounded neural dynamics | protected candidate | 必须在 Stage37/teacher floor 或 causal floor fallback 保护下。 |
| Source/domain/full-waypoint | Stage42-FH/FI frozen protected source/domain policy | paper-usable protected evidence | 允许 source/domain protected claim，不允许 uniform horizon overclaim。 |
| Teacherless switch gate | Stage42-HE/HF teacherless proximity-guarded switch gate | bounded paper/deployment claim | teacher gate 可移除，但 causal floor fallback 必须保留。 |
| Restricted metric/time | Stage42-HI/HJ restricted metric/time readiness + source-CV preflight | technical-after-terms candidate | ready now = 0；UCY after terms 可规划 robust source-CV；ETH_UCY 因当前 t100 source support 不足仍 blocked；必须用户确认 terms/path/source identity 后再 conversion/no-leakage/source-CV/final-test。 |

## 1. 你问“这个目标内做了什么”，实际做过的主线

### 1.1 数据、benchmark、registry、license 主线

做过：

- 建立 WAM-style data registry 与 web dataset acquisition agent。
- 搜索并登记 SDD、OpenTraj、TrajNet++、ETH/UCY、UCY Crowd、Aerial/top-down pedestrian、egocentric video、traffic diagnostic 等数据源。
- 建立 license/access audit：区分 automatic download、manual terms、login/application、local path verification。
- 接入 SDD，转换成 per-video world-state shards。
- 构建 SDD scene packs、lazy episodes、GoalBench、HardBench、BaselineFailureBench、no-leakage audit、strongest causal baselines。
- 扩展 external top-down 数据：OpenTraj/UCY/ETH_UCY/TrajNet，建立 external feature store、row geometry、history windows、goal prototypes、source/domain splits。

成功：

- SDD official pixel-space raw-frame benchmark 建立。
- External dataset-local raw-frame feature/eval 链建立。
- Source/domain 和 history/goal prototype 机制建立。

失败或 blocked：

- 外部数据不能直接写成 global metric/seconds-level。
- 一些 source 仍受 legal/source/path/terms 阻塞。
- Stage42-HI 只证明 ETH/UCY 有 H/FPS/stride 技术线索，`restricted_metric_time_ready_now_count = 0`，不能写成当前 metric/time result。
- Stage42-HJ 进一步解析本地 candidate rows：UCY 有 3 个 usable-after-terms sources，可规划 leave-one-source style robust source-CV；ETH_UCY 只有 `ETH_seq_eth` 同时具备 t50/t100，`ETH_seq_hotel` t100=0，所以 ETH_UCY source-CV 仍 blocked。

原因：

- SDD 是 pixel-space annotation；homography/scale/effective seconds 未验证。
- External source 的 official terms、source identity、local path confirmation、conversion/no-leakage/source-CV 还未全部闭环。
- ETH_UCY 当前缺第二个 t100-capable usable source；这不是模型失败，而是 source-CV 支持不足。
- registry 或 prefill 不是 permission，也不是 converted/evaluated data。

### 1.2 Strongest causal baseline 与 safety floor 主线

做过：

- 系统评估 constant position、causal constant velocity、damped velocity、constant acceleration、turn-rate、scene-clamped、goal/prototype-directed 等 baseline。
- 把 strongest causal baseline / Stage37 policy / teacher floor 作为 learned policy 的 fallback floor。
- 评估 oracle headroom、selector regret、harm over fallback、easy degradation、hard/failure improvement。

成功：

- 这条路线成为 M3W 最稳的安全骨架。
- 后续 Stage26、Stage37、Stage42 protected policies 都依赖它。
- Stage42-HF 进一步确认：可以允许 `teacherless proximity-guarded switch gate with causal floor fallback`，但不能移除 causal floor fallback。

失败或 blocked：

- Global floor removal 没过。
- Ungated neural / floor-free global deployment 没过。

原因：

- 某些 floor-free policy raw improvement 很高，但 near-collision/proximity safety 不达标。
- Easy case 一旦被错误切换，整体可部署性就崩。

### 1.3 SDD selector 主线

做过：

- 从 hard classification selector 开始，预测“哪个 baseline 最好”。
- 发现 hard-class selector 对 low-margin / ambiguous 样本过度切换。
- 改成 expected-FDE / regret-aware / confidence-gated / fallback-safe selector。

失败：

- Stage24 validation-selected hard-class selector 失败：
  - t+50 improvement 约 `-43.3%`
  - easy degradation 约 `11.33%`

原因：

- Oracle best label 很多是 low margin。
- best baseline 与 second-best baseline 差距小，硬标签噪声大。
- 直接强制切换会伤害 easy cases。

成功：

- Stage26 cost-aware selector 成功：
  - t+50 improvement 约 `+14.58%`
  - hard/failure improvement 约 `+11.23%`
  - easy degradation 约 `+1.81%`

结论：

- Selector 不能做 hard classification。
- 必须做 cost-aware / regret-aware / confidence-gated / fallback-safe policy。

### 1.4 JEPA 表征主线

做过：

- Stage18/19/22/23 以及后续 M3W 中训练过 trajectory-only JEPA、scene/trajectory JEPA、interaction-aware JEPA。
- 检查 non-collapse、latent variance、selector probe、failure predictor probe、goal predictor、hard/failure correction lift。

成功：

- 多次确认 JEPA non-collapse。

失败：

- Downstream lift 没有稳定证明。
- JEPA 没有成为 selector/failure/correction/trajectory 的主贡献。

原因：

- Non-collapse 不等于 task lift。
- JEPA target 与实际部署目标不完全一致。
- 当前数据/目标更需要 gain/harm/switchability 和 baseline-family rollout context，而不是纯 representation pretraining。

结论：

- JEPA 当前只能作为 auxiliary / diagnostic representation。
- 不能把 JEPA 说成 latent generative world model。
- 不能把 JEPA 写成当前主贡献。

### 1.5 Transformer / Hybrid neural dynamics 主线

做过：

- 训练 Transformer-only、JEPA-only、JEPA+Transformer hybrid。
- Stage39 开始真实训练 Causal Transformer / JEPA / Hybrid neural dynamics。
- Stage40 做 neural optimization trials，加入 teacher distillation、horizon-specific heads、domain embedding、hard/failure oversampling、safe fallback。

失败：

- Stage39 neural 没有超过 Stage37，因此不部署。
- Stage40 best neural_with_fallback 与 Stage37 reference 基本一致；neural_without_fallback 灾难性失败：
  - all improvement `-126.36%`
  - t50 improvement `-292.10%`
  - easy degradation `612.31%`

原因：

- 无保护 neural residual 直接改轨迹，容易破坏 easy cases。
- 数据是 dataset-local/raw-frame，不是 metric world dynamics。
- 神经网络在当前证据下更像“protected candidate/reranker”，不是独立 dynamics replacement。

成功：

- M3W-Neural v1 protected candidate 成立：
  - all improvement vs Stage37 floor `+21.03%`
  - t+50 improvement `+13.65%`
  - t+100 raw-frame diagnostic `+14.69%`
  - hard/failure `+20.38%`
  - easy degradation `0.00%`
  - gates `41 / 41`

边界：

- 这是 protected neural candidate under Stage37/teacher floor。
- 不是 ungated neural deployment。
- 不是 foundation world model。

### 1.6 External zero-shot / domain alignment 主线

做过：

- Stage31 评估 SDD -> external zero-shot。
- Stage32 做 normalization、relative targets、domain adapter、CORAL/whitening。
- Stage33 做 coordinate-invariant features、relative-error targets、domain-conditioned selector。
- Stage34 补 external row geometry、train-only goals、scene packs、horizon split。
- Stage35 做 selective transfer policy。
- Stage36 专修 t+50。
- Stage37 引入 full past-only history windows + scene-agnostic goal prototypes + t50 switchability/gain/harm + conformal safety。

失败：

- Stage31 zero-shot external transfer 大失败：
  - all improvement 约 `-92.67%`
  - t50 约 `-278.57%`
- Stage32 adapted selector 约 0 improvement。
- Latent adapter 缩小分布距离，但没有 predictive lift。
- Stage35 all/hard positive，但 t50 = 0，不可部署。
- Stage36 t50 仍为 0。

原因：

- SDD pixel 与 external dataset-local/weak metric 坐标不兼容。
- horizon 定义、scale、agent type、scene/goal context 不一致。
- held-out scenes 没有 train-scene goal context。
- 普通 normalization 修的是分布，不一定修任务。
- all-objective 会压制 t+50 专门切换。

成功：

- Stage37 修复 external t+50：
  - all improvement `+13.48%`
  - t+50 improvement `+8.46%`
  - t+50 bootstrap CI `[+7.69%, +9.15%]`
  - hard/failure `+15.54%`
  - easy degradation `0.041%`
  - gates `16 / 16`
  - verdict `stage37_t50_transfer_repaired_deployable`

核心有效机制：

- past-only history windows
- scene-agnostic goal prototypes
- t50-specific switchability/gain/harm
- conservative/conformal safety fallback

### 1.7 Correction / residual 主线

做过：

- Stage38 bounded correction / dynamics head。
- 训练 linear/ridge/small MLP/horizon-specific/hard-only/t50-only correction。

失败：

- Correction with fallback 没有安全超过 Stage37。
- 普通 residual 或 unbounded residual 容易伤 easy cases。

原因：

- Strongest baseline / Stage37 已经很好。
- Residual 直接改轨迹的风险比 selector 切换更大。
- 当前缺 metric/scene-grounded dynamics，纯 residual 容易过拟合 dataset-local geometry。

结论：

- Stage38 correction not deployable。
- 当前仍保留 Stage37 / protected policy 作为 floor。

### 1.8 Full-waypoint / source-domain / group-consistency 主线

做过：

- 从 endpoint-linear bridge 扩展到 full-waypoint shape。
- 做 common validation bridge-shape composer。
- 做 proximity-aware composer guard。
- 做 group-consistency full-waypoint policy。
- 做 source/domain/horizon/frozen replay/bootstrap。

成功：

- Stage42-DL/DM runtime replay：
  - rows `47,458`
  - all/t50/t100raw/hard `+24.72% / +22.36% / +14.35% / +23.89%`
  - near@0.05 从 `1.94%` 降到 `1.38%`
- Stage42-FE constrained safety composer：
  - all/t50/hard `+26.41% / +23.15% / +24.81%`
  - near@0.05 `1.32%`
- Stage42-FH source/domain composer：
  - all/t50/t100raw/hard `+34.98% / +28.97% / +20.57% / +33.10%`
  - TrajNet 和 UCY positive-safe
- Stage42-FI frozen replay：
  - policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`
  - replay diff `0`
  - 2000-bootstrap CI low all/t50/t100raw/hard `+34.62% / +28.46% / +19.96% / +32.73%`

失败或边界：

- Uniform horizon claim 仍不允许。
- TrajNet|100、UCY|100 仍有 weak/h100/source/legal blocker。

原因：

- h100/t100 source support 稀疏。
- 部分 horizon slice low-margin / high ambiguity。
- UCY/ETH metric-time candidate 仍需 terms/path/source identity confirmation。

### 1.9 Floor-free / teacherless 主线

做过：

- Stage42-HC 测试 floor alternatives：internal self-gate、uncertainty gate、conformal risk gate、harm predictor、teacher-dependent gates、bounded residual。
- Stage42-HD 给 floor-free candidates 加 proximity guard。
- Stage42-HE 做 robustness。
- Stage42-HF 固化 deployment contract。
- Stage42-HG 做 claim linter。

失败：

- Global floor-free neural deployment 失败。
- Causal floor removal 失败。
- 有些 floor-free candidate raw all/t50/hard 很高，但 near-collision/proximity 不安全。

成功：

- Stage42-HE/HF 允许一个非常窄的 bounded claim：

```text
teacherless proximity-guarded switch gate with causal floor fallback
```

关键数字：

- policy family `harm_predictor_gate`
- teacher_gate_used `False`
- causal_floor_fallback_used `True`
- all improvement `+20.74%`
- t50 improvement `+13.82%`
- t100 raw diagnostic `+13.68%`
- hard/failure `+19.99%`
- easy degradation `0.00%`
- bootstrap_n `2000`
- robust positive domains `ETH_UCY`, `TrajNet`, `UCY`
- Stage42-HF gate `15 / 15`
- Stage42-HG claim linter violations `0`

边界：

- 这不是 global floor removal。
- 这不是 ungated neural deployment。
- causal floor fallback 仍必须保留。

## 2. 当前哪些路线成功了

按“可作为当前项目有效成果”的强度排序：

1. **Stage26 SDD cost-aware selector**
   - SDD pixel/raw-frame 上最强 deployable baseline policy。
   - 证明 cost-aware/regret-aware selector 比 hard-class selector 更合理。

2. **Stage37 external t50 safe selector**
   - 第一次把 external t50 从 0 修成可部署正迁移。
   - 证明 past-only history + goal prototype + switchability/gain/harm + conformal safety 有效。

3. **M3W-Neural v1 protected candidate**
   - protected neural candidate 在 Stage37/teacher floor 下有正向 evidence。
   - 有 bootstrap/multiseed/pure-UCY/endpoint-to-full bridge 等支持。

4. **Stage42 source/domain protected full-waypoint policy**
   - Stage42-FH/FI 给出 source/domain protected success 和 frozen replay。
   - 可以写 protected full-waypoint/source-domain evidence。

5. **Stage42 teacherless proximity-guarded switch gate**
   - 证明 teacher gate 可以在特定 repaired policy 中不使用。
   - 但 causal floor fallback 仍是安全底座。

6. **Claim guard / linter / deployment contract**
   - 防止把 selector-level/protected evidence 写成 true 3D/foundation/metric/ungated neural。
   - 对论文包和 README 很重要。

## 3. 当前哪些路线失败了，核心原因是什么

| 路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| Hard-class selector | Stage24 t50 大负，easy 伤害大 | low-margin oracle label 噪声；硬切换 | 已替换为 expected-FDE/regret-aware |
| JEPA 主线 | non-collapse 但 no downstream lift | 表征目标与部署目标错位 | 降级为 auxiliary/diagnostic |
| SDD->external zero-shot | all/t50 大负 | coordinate/scale/horizon/scene/agent mismatch | 改成 external history/prototype/safe selector |
| Latent adapter/CORAL | latent gap 变小但无 predictive lift | 分布对齐不等于任务对齐 | 不作为 success claim |
| Ordinary residual/correction | 没安全超过 Stage37 | 直接改轨迹风险高，伤 easy | 不部署 |
| Unprotected Transformer/Hybrid | neural_without_fallback 灾难性失败 | 缺安全 floor；dataset-local dynamics 不稳 | 只允许 protected neural candidate |
| Scene/goal 独立主贡献 | ablation 不稳定 | train-only goal proxy 对 held-out/source shift 弱 | 只能 auxiliary |
| Neighbor/interaction 独立主贡献 | lift 不稳定 | 当前 interaction token 不足以独立驱动 | 只允许 group-consistency protected claim |
| Uniform h100/t100 | weak horizon/source blocker | source support 稀疏、legal/terms 未闭环 | blocked，不写完成 |
| Global floor removal | safety 不过 | near-collision/proximity risk | blocked |
| Metric/seconds claim | ready_now=0 | terms/path/source identity/conversion/source-CV 未完成 | blocked |

## 4. 当前模型大概是什么质量

如果按研究质量分层：

```text
不是 demo 级；
不是 foundation 级；
不是 true 3D；
不是 metric/seconds-level；
是一个有较强工程闭环和可写论文雏形的 protected 2.5D world-state candidate。
```

更具体地说：

- **工程闭环质量**：较强。已经有数据、feature store、safe policy、runtime replay、bootstrap、linter、deployment contract。
- **SDD 内部质量**：较强。Stage26 是可靠 baseline selector。
- **External t50 质量**：较强。Stage37 成功修复 t50，并有 CI。
- **Protected neural/world-state 质量**：中到较强。M3W-Neural v1 和 Stage42 full-waypoint/source-domain evidence 可写，但必须强调 protected/fallback。
- **跨域泛化质量**：中等偏强，但仍受 source diversity / legal / horizon blockers 限制。
- **世界动力学质量**：有 protected evidence，但还不能说 neural dynamics 独立主导。
- **论文候选质量**：可以形成 protected 2.5D external world-state dynamics manuscript draft；还不到 broad true-3D/foundation/world-model claim。

## 5. 现在可以写的 claim 与不能写的 claim

可以写：

- M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
- Stage26 在 SDD pixel/raw-frame 上给出 cost-aware selector improvement。
- Stage37 在 external t50 上给出 deployable safe selector improvement。
- M3W-Neural v1 是 protected neural candidate under Stage37/teacher floor。
- Stage42-FH/FI 支持 source/domain protected full-waypoint evidence。
- Stage42-HE/HF 支持 teacherless proximity-guarded switch gate **with causal floor fallback**。
- Stage42-HI 说明 ETH/UCY 有 restricted metric/time technical candidates after terms，但当前不能 claim。

不能写：

- true 3D world model
- foundation world model
- global metric predictor
- seconds-level long-horizon result
- ungated neural dynamics deployable
- global floor-free deployment
- causal floor removal
- JEPA 主贡献
- Transformer-only 主贡献
- scene/goal 或 neighbor/interaction 独立主贡献
- Stage5C 已执行
- SMC 已启用
- user-action / terms prefill = legal permission
- registry-only / not_run = converted/evaluated

## 6. 最短下一步

1. **完成 restricted metric/time source terms confirmation**
   - Stage42-HI 已识别 6 个 ETH/UCY 技术候选。
   - Stage42-HJ 显示 UCY 在 terms confirmed 后可做 robust source-CV，t50/t100 after-terms windows 约 `9554 / 5605`；ETH_UCY 仍需要更多 t100-capable usable source。
   - 但 ready now = 0。
   - 需要用户确认 source terms/path/source identity 后，才能做 conversion/no-leakage/source-CV/final-test；不能把 preflight 写成 conversion/evaluation。

2. **继续 source-CV / independent source 修复**
   - 当前 evidence 已强，但 source diversity 和 uniform horizon 仍是限制。
   - 目标是让 protected policy 不只在当前 source/domain 上强。

3. **保留 causal floor fallback，研究更窄的 teacherless gate**
   - Stage42-HE/HF 已允许 teacherless proximity-guarded switch gate。
   - 下一步只能在 causal floor fallback 存在时扩展，不能做 global floor removal。

4. **不要再把 JEPA/Transformer 当主线硬推**
   - 除非新 ablation 证明 downstream lift。
   - 当前最有用的是 history、safe switch、domain/source policy、full-waypoint protected shape、group consistency。

## 7. 最终当前判定

项目是否跑通：是，作为 protected dataset-local/raw-frame 2.5D world-state candidate 跑通。  
是否 true 3D：否。  
是否 foundation：否。  
是否 metric/seconds-level：否。  
Stage5C 是否执行：否。  
SMC 是否启用：否。  
当前 best SDD deployable：Stage26 cost-aware selector。  
当前 best external deployable：Stage37 safe selector / Stage42 protected causal-floor fallback family。  
当前 best protected neural candidate：M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor。  
当前最新 bounded teacherless claim：Stage42-HF `teacherless proximity-guarded switch gate with causal floor fallback`。  
当前最大 blocker：restricted metric/time legal-confirmed conversion + source-CV + uniform h100/horizon/source-diversity closure。

<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:START -->
## Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight

- source: `fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight`
- verdict: `stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked`
- gates: `16 / 16`
- augmented ETH_UCY independent sources after terms: `5`.
- augmented ETH_UCY t50/t100 windows after terms: `4397` / `1433`.
- cached BL technical t100 safe-positive: `True`; ready now: `False`.
- conclusion: ETH_UCY source-CV blocker is technically repairable after terms using ETH-Person XML candidates, but conversion/evaluation and metric/seconds claims remain blocked until user-confirmed terms and guarded rerun.
<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:END -->

<!-- STAGE42_HL_RESTRICTED_METRIC_TIME_POST_HK_CLAIM_GUARD:START -->
## Stage42-HL Restricted Metric/Time Post-HK Claim Guard

- source: `fresh_stage42_hl_restricted_metric_time_post_hk_claim_guard`
- verdict: `stage42_hl_restricted_metric_time_post_hk_claim_guard_pass`
- gates: `15 / 15`
- files scanned / violations: `14` / `0`.
- HK after-terms source support: `5` sources, t50/t100 windows `4397` / `1433`.
- ready now: `False`; conversion ready targets now: `0`.
- conclusion: the paper/README package remains claim-safe after HK; ETH_UCY source support is technically repairable after terms, but restricted metric/time conversion/evaluation remains blocked until user confirmation and guarded rerun.
<!-- STAGE42_HL_RESTRICTED_METRIC_TIME_POST_HK_CLAIM_GUARD:END -->
