# M3W 长期目标研究总结：路线、失败原因、成功证据与当前结论

更新时间：2026-05-26  
工作目录：`/Users/yangyue/Downloads/World`  
总结来源：`cached_verified` 汇总既有报告；其中 Stage42-R/S/T 的最新分支结果来自本轮前序 `fresh_run` 报告。  

这个 README 是给长期目标看的总账本：把 M3W 从 SDD selector 走到 external protected neural world-state candidate 的路线、失败、成功、边界和下一步集中写清楚。它不是论文包装稿，也不是“已经 foundation / true 3D”的声明。

## 1. 当前总结论

当前最强可部署候选仍是受保护路径：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
safety floor = Stage37 selector / teacher floor
deployment rule = protected / conservative / validation-selected
Stage5C executed = false
SMC enabled = false
```

它已经不是最早的 selector-only SDD demo：现在有 external dataset-local raw-frame 结果、bootstrap、多 seed、strict pure-UCY、all-agent、endpoint-to-full、full-waypoint、static-gated 和 row-cache combo 分支证据。

但必须继续诚实承认：

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- external 是 dataset-local / unverified weak-metric diagnostic，不能混写成统一米制。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 未全局验证。
- self-audited / visual-prior / auto-silver 不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 未执行。
- SMC 未启用。
- 无保护 neural dynamics 仍不安全，部署仍需要 Stage37/teacher floor。

## 2. 当前最有意义的数字

### M3W-Neural v1 主 package

相对于 Stage37/teacher floor：

| 指标 | 结果 |
| --- | ---: |
| gates | 41 / 41 |
| rows | 55,528 |
| all ADE improvement | 21.03% |
| t+50 ADE improvement | 13.65% |
| t+100 raw-frame diagnostic ADE improvement | 14.69% |
| hard/failure ADE improvement | 20.38% |
| easy degradation | 0.00% |
| positive external domains | 3 |
| all-agent composite FDE improvement | 19.82% |
| all-agent composite FDE@50 improvement | 17.39% |

2000-bootstrap lower bounds：

| slice | low | mid | high |
| --- | ---: | ---: | ---: |
| all | 20.67% | 21.02% | 21.39% |
| t+50 | 13.06% | 13.66% | 14.26% |
| t+100 raw-frame diagnostic | 13.96% | 14.69% | 15.37% |
| hard/failure | 19.99% | 20.39% | 20.76% |

### Stage37 external deployable selector

Stage37 是 external t+50 修复的第一个可部署正迁移点：

| 指标 | 结果 |
| --- | ---: |
| rows | 66,303 |
| all improvement | +13.48% |
| t+50 improvement | +8.46% |
| t+50 bootstrap CI | [+7.69%, +9.15%] |
| hard/failure improvement | +15.54% |
| easy degradation | 0.041% |
| gates | 16 / 16 |
| verdict | `stage37_t50_transfer_repaired_deployable` |

### Stage42 full-waypoint / row-cache 分支

Stage42-C 把 endpoint/bridge 证据推进到 reconstructed full future waypoint：

| 指标 | 结果 |
| --- | ---: |
| protected full-waypoint ADE all | 18.58% |
| protected full-waypoint ADE t+50 | 14.80% |
| protected full-waypoint ADE t+100 raw-frame diagnostic | 22.86% |
| protected full-waypoint ADE hard/failure | 19.52% |
| easy degradation | 0.00% |
| positive domains | ETH_UCY, TrajNet |

Stage42-R 把 Stage42-J/P 的 report-level combo 变成 row-cache-backed combo：

| 指标 | 结果 |
| --- | ---: |
| source | `fresh_run_from_row_prediction_cache` |
| gates | 15 / 15 |
| ADE all | 0.052387 |
| ADE t+50 | 0.037934 |
| ADE t+50 CI low | 0.027740 |
| ADE t+100 raw-frame diagnostic | 0.041846 |
| ADE hard/failure | 0.054792 |
| easy degradation | 0.001102 |
| FDE t+50 | 0.100059 |

Stage42-S 冻结 row combo policy：

| 项 | 值 |
| --- | --- |
| verdict | `stage42_s_frozen_row_combo_policy_pass` |
| gates | 13 / 13 |
| positive domains | ETH_UCY, TrajNet |
| UCY status | fallback-only in this combo stress |
| policy hash | `33450e033e14b10293b8a10796d934d7689e39358ab5eaa338d684a36b015d3f` |

Stage42-T 诊断 UCY：

| 项 | 值 |
| --- | --- |
| verdict | `stage42_t_ucy_transfer_blocked_no_candidate_predictions` |
| gates | 8 / 11 |
| UCY rows | 9,540 |
| UCY all/t50/hard/easy | 0.0 / 0.0 / 0.0 / 0.0 |
| root cause | Stage42-R row cache 对 UCY 没有非 floor Stage42-J/P 候选预测 |

这意味着 UCY 当前不是 threshold 没调好，而是没有可切换的 UCY candidate source。下一步要训练 UCY-aware 或 source-agnostic candidate prediction source，或者重建合法 UCY train/val calibration。

Stage42-U 继续检查一个最自然的修复：把 Stage41 strict pure-UCY endpoint neural candidate 桥接到 Stage42 full-waypoint。结果是 blocker，不是成功：

| 指标 | UCY zara03 test |
| --- | ---: |
| ADE all | -0.070821 |
| ADE t+50 | -0.492070 |
| hard/failure | -0.083302 |
| easy degradation | 0.566646 |
| gates | 7 / 8 |
| verdict | `stage42_u_ucy_endpoint_to_full_bridge_failed_blocker` |

这个结果说明：Stage41 pure-UCY endpoint residual 在 endpoint 指标上成立，但不能直接通过线性插值当作 full-waypoint world-state success。下一步必须训练 UCY-aware full-waypoint candidate source，或学习 validation-selected waypoint-shape bridge。

## 3. 走过的路线

### 路线 A：早期 JEPA / WAM-style 表征

尝试内容：

- Stage18 SAM-JEPA-2.5D。
- Stage19 WAM-style simulation / top-down / ego-video data registry 和 JEPA pretraining。
- 后续 Stage22/23/24/39/40 也多次检查 JEPA auxiliary。

结果：

```text
JEPA non-collapse = yes
downstream lift = no
deployable contribution = no
```

失败原因：

- JEPA latent 有方差不代表对 selector/failure/correction 有用。
- 下游任务需要因果历史、hard/easy/failure、gain/harm、fallback 结构，普通 representation pretraining 没对齐部署目标。
- JEPA 不能当 latent generative rollout，Stage5C 没开启。
- Stage39 JEPA downstream failure AUROC lift 为负，说明当时 JEPA 特征更像噪声。

结论：

JEPA 目前保留为 diagnostic / auxiliary，不进入 deployable path。

### 路线 B：SDD official pixel-space benchmark 和 selector 稳定化

尝试内容：

- Stage21/22 建 SDD world-state shards、scene packs、episodes、GoalBench、HardBench、BaselineFailureBench。
- Stage23 quick-plus 失败后，Stage24 修 I/O，建立 fast cache 和 true medium index。
- Stage25 诊断 selector oracle headroom 大但训练 selector 负提升。
- Stage26 改成 feature-complete cost-aware expected-FDE selector。

成功结果：

```text
Stage26 selector:
t50 improvement ~= 14.58%
hard/failure improvement ~= 11.23%
easy degradation ~= 1.81%
```

关键经验：

- 不能做 hard classification：“哪个 baseline 最好”。
- 必须预测 expected FDE / risk / gain / harm。
- 必须 fallback 到 strongest causal baseline。
- 必须保护 easy case。

失败原因总结：

- Stage24 hard-class selector 虽然 oracle headroom 46.2%，但 trained selector t50 为 -43.3%，easy degradation 11.33%。
- 根因是 low-margin 样本太多、oracle label 不稳定、hard label 不是 cost-aware、selector 大量过度切换 easy cases。

结论：

Stage26 是 SDD 上第一个可靠 selector 基座，但它仍是 SDD pixel-space，不是 metric/3D/foundation。

### 路线 C：SDD 到 external zero-shot / domain alignment

尝试内容：

- Stage31 建 external feature store、latent cache、baseline、transfer eval。
- Stage32 做 normalization、CORAL、latent adapter、domain-conditioned selector。
- Stage33 做 coordinate-invariant features、relative-error target、domain-conditioned selector。
- Stage34 补 row geometry、train-only goals、scene packs、relative baselines。

初始失败：

```text
Stage31 SDD -> external zero-shot:
all improvement ~= -92.67%
t50 ~= -278.57%
```

失败原因：

- SDD pixel 与 external dataset-local coordinates 不兼容。
- external scene/goal/interaction 缺失。
- agent type 分布和标注方式不同。
- horizon / track length / frame step 不匹配。
- scale / homography / metric 影响大但未验证。
- latent adapter 能缩小分布距离，但没有 predictive lift。

Stage34 局部正信号：

- t+50 diagnostic lift 约 +6.6%。
- hard/failure 约 +18% 到 +25%。
- 但 all-test 为负，easy degradation 高，所以不能部署。

结论：

普通 normalization / latent alignment 不够，external 需要逐行几何、目标原型、hard/easy 判别和选择性迁移。

### 路线 D：Selective transfer 和 external t+50 修复

Stage35：

```text
all improvement = +12.13%
hard/failure improvement = +13.98%
easy degradation = 0.041%
t50 improvement = 0.0
verdict = not deployable
```

Stage36 取证发现：

- t+50 有 16,263 test rows。
- t+50 oracle headroom 约 22.98%。
- 不是没有可学空间。
- 问题是现有特征/goal/context 不支持安全切换，all-test objective 淹没了 t+50。

Stage37 修复：

- 构建 past-only history windows K=8/16/32/64。
- 构建 scene-agnostic goal prototypes：straight/stop/turn/u-turn/group-follow/density-avoid/exit-like。
- 训练 t+50 failure/gain/harm/switchability models。
- 用 conformal / conservative safety gate 保护 easy。

成功结果：

```text
all improvement = +13.48%
t50 improvement = +8.46%
t50 bootstrap CI = [+7.69%, +9.15%]
hard/failure = +15.54%
easy degradation = 0.041%
gates = 16/16
```

结论：

Stage37 是 external t+50 deployable selector candidate。但它仍主要是 selector-level / policy-level，不是无保护 neural dynamics。

### 路线 E：bounded correction / residual

尝试内容：

- Stage38 在 Stage37 fallback 保护下训练 bounded correction。
- 比较 correction without fallback / with fallback / hard-only / t50-only。

结果：

```text
Stage38 deployment decision = keep_stage37_selector
correction not deployable
```

失败原因：

- correction 有 t50 diagnostic lift，但损失 all/hard 或 safety。
- bounded residual 仍容易在 easy cases 或非目标 slice 上带来负迁移。
- 没有证明比 Stage37 selector 更稳。

结论：

普通 residual / correction 不是下一条主线，除非 selector/failure/headroom 足够可靠并且 easy preservation 过 gate。

### 路线 F：Stage39/40 真神经动力学

尝试内容：

- Causal Transformer candidate ranker。
- t50 curriculum ranker。
- hard/failure oversampled ranker。
- Stage37 teacher-distilled safe ranker。
- JEPA auxiliary candidate ranker。
- hybrid MoE deeper ranker。

结果：

```text
Stage40 best neural = causal_transformer_candidate_ranker
with fallback metrics = same as Stage37 on same subset
neural deploy = false
verdict = keep_stage37_selector
```

无保护 neural 失败非常明显：

```text
neural_without_fallback all = -1.2636
t50 = -2.9210
hard/failure = -1.0940
easy degradation = 6.1231
```

失败原因：

- Transformer 学到的安全可切换区域太少，fallback gate 最终 switch_rate = 0。
- JEPA non-collapse 但 downstream lift 负。
- Hybrid 受 JEPA 噪声和不稳定 candidate 影响，不能超过 Stage37。
- raw FDE / endpoint loss 没有直接教会“何时不要切”。

结论：

Stage39/40 证明“直接端到端 neural”不够安全。神经网络必须作为 protected candidate / bounded dynamics 分支，不能直接替代 Stage37。

### 路线 G：Stage41 M3W-Neural v1 protected neural package

尝试内容：

- composite-tail bounded neural dynamics。
- Stage37/teacher floor。
- multiseed / bootstrap。
- strict pure-UCY neural retrain。
- all-agent composite world-state。
- endpoint-to-full bridge。
- ablation coverage。

成功结果：

```text
gates = 41 / 41
all = +21.03%
t50 = +13.65%
t100 raw-frame diagnostic = +14.69%
hard/failure = +20.38%
easy degradation = 0.00%
positive external domains = 3
```

strict pure-UCY：

```text
all = +9.01%
t50 = +8.80%
t100 raw-frame diagnostic = +8.31%
hard/failure = +9.36%
easy = 0.00%
bootstrap lows all/t50/t100/hard = 8.89 / 8.63 / 8.07 / 9.23%
```

结论：

这是目前最强 protected neural evidence。但它依赖 safety floor，所以不能包装成 ungated neural world model。

### 路线 H：Stage42 full-waypoint、static gate、row-level combo

尝试内容：

- Stage42-A data/calibration audit。
- Stage42-B external validation stress。
- Stage42-C full-waypoint dynamics。
- Stage42-D/E causal ablation 和 safety floor。
- Stage42-F paper package。
- Stage42-G fresh retrained ablation Phase1。
- Stage42-H causal sequence ablation。
- Stage42-I sequence-to-full-waypoint。
- Stage42-J static-gated full-waypoint repair。
- Stage42-K/L fresh checkpoint + horizon-aware repair。
- Stage42-M/N policy distillation / row-level alpha teacher。
- Stage42-O/P explicit gain/harm + t50-specific repair。
- Stage42-Q combo preflight。
- Stage42-R row prediction cache combo。
- Stage42-S frozen row combo policy。
- Stage42-T UCY unseen transfer diagnosis。

关键成功：

- Stage42-B protected external validation 仍为正。
- Stage42-C full-waypoint positive on ETH_UCY and TrajNet。
- Stage42-H 证明 history tokens 在 sequence encoder 下有强贡献。
- Stage42-J 证明 static/context 不是无效，而是必须 validation-gated。
- Stage42-P 修复 mean t50，但 CI low 仍负。
- Stage42-R row cache combo 让 t50 CI low 变正。
- Stage42-S 冻结 policy artifact，记录 hash 和 schema。

关键失败：

- Stage42-I 全 static/context sequence-to-waypoint 负，说明静态上下文全局混入会伤模型。
- Stage42-M slice-level alpha distillation 不能学到 row-level gain/harm。
- Stage42-N row-level alpha teacher all/hard 为正，但 t50 仍负。
- Stage42-O explicit gain/harm all/hard 为正，但 t50 仍略负。
- Stage42-T UCY 不能靠 transfer rule 修复，因为 row cache 没有 UCY 非 floor 候选预测。

结论：

Stage42 已经让 full-waypoint / row-level protected branch 更可信，但 UCY fallback-only 和 floor dependence 仍是硬差距。

## 4. 失败路线总表

| 路线 | 状态 | 失败原因 |
| --- | --- | --- |
| JEPA-only / early JEPA downstream | 失败 | non-collapse 不等于 downstream lift；selector/failure/correction 未改善。 |
| hard-class baseline selector | 失败 | 低 margin oracle label 噪声大，过度切换 easy cases。 |
| SDD -> external zero-shot | 失败 | 坐标尺度、scene/goal、agent type、horizon 全部 domain gap。 |
| raw normalization / CORAL / latent adapter | 部分失败 | 缩小分布距离不等于预测有用。 |
| Stage34 full external transfer | 不可部署 | t50/hard 局部正，但 all/easy 失败。 |
| Stage35 selective transfer | 不可部署 | all/hard/easy 过，但 t50 = 0。 |
| Stage36 threshold/policy search | 失败 | t50 需要新历史/目标特征，不是阈值问题。 |
| bounded residual correction | 不可部署 | t50 局部有信号，但 all/hard/easy 不稳。 |
| Stage39/40 ungated neural | 失败 | easy degradation 大，without fallback 灾难性负迁移。 |
| JEPA auxiliary in Stage39/40 | 失败 | downstream lift 负，给 ranker 增加噪声。 |
| Stage42 global static context | 失败 | 静态/场景信息全局混入导致 full-waypoint ADE 负。 |
| Stage42 policy alpha distillation | 部分失败 | 粗粒度 alpha 教不会 row-level gain/harm。 |
| Stage42 UCY row combo transfer | blocker | 当前 row cache 没有 UCY 非 floor candidate predictions。 |

## 5. 成功路线总表

| 路线 | 状态 | 成功点 |
| --- | --- | --- |
| Stage26 cost-aware expected-FDE selector | SDD 成功 | t50/hard/easy 三个 gate 同时过。 |
| Stage37 history + goal prototype + safety selector | external 成功 | 修复 external t50，gates 16/16。 |
| Stage41 composite-tail bounded neural dynamics | protected neural 成功 | all/t50/t100/hard 都正，easy 0。 |
| strict pure-UCY neural retrain | 成功 | source-heldout UCY branch bootstrap-stable positive。 |
| endpoint-to-full bridge | 成功 | ETH_UCY/TrajNet full-waypoint ADE/FDE lower bounds positive。 |
| all-agent composite world-state | 成功 | 不是只做单 agent endpoint。 |
| Stage42-C full-waypoint dynamics | 成功 | reconstructed full future waypoint positive on ETH_UCY/TrajNet。 |
| Stage42-H sequence history ablation | 成功 | history tokens 在 causal sequence encoder 下贡献强。 |
| Stage42-J static-gated full-waypoint repair | 成功 | partial static/context validation-gated 后有效。 |
| Stage42-R row prediction cache combo | 成功 | combo t50 CI low 为正，修复 Stage42-Q 只能 preflight 的问题。 |
| Stage42-S frozen row combo policy | 成功但有 UCY 限制 | ETH_UCY/TrajNet 正；UCY fallback-only。 |
| Stage42-U UCY endpoint-to-full bridge | 失败但有用 | 证明 pure-UCY endpoint 成功不能直接算 full-waypoint 成功；需要 UCY full-waypoint candidate。 |

## 6. 当前 best deployable 与 branch evidence 的关系

当前最强可部署主线：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
under Stage37 / teacher safety floor
```

Stage42-R/S 是重要的新分支证据：

- 它让 Stage42-J/P 的 row-level combo 有了 cache-backed test-once 证据。
- 它增强 full-waypoint / t50 branch。
- 它不是替代整个 M3W-Neural v1 package 的新总模型。
- 它暴露了 UCY fallback-only 的真实 blocker。

所以口径应是：

```text
M3W-Neural v1 remains current strongest protected deployable package.
Stage42-R/S is the latest full-waypoint row-cache combo branch, positive on ETH_UCY and TrajNet, with UCY still blocked.
```

## 7. 为什么现在不直接做 Stage5C / SMC

Stage5C latent generative 和 SMC 仍禁止，因为：

- selector/failure/correction 的外部泛化仍依赖 safety floor。
- UCY 在 Stage42-S/R combo 中仍 fallback-only。
- JEPA 没有稳定 downstream lift。
- ungated neural 不安全。
- metric/time calibration 未完成。
- 还没有 stochastic proposal coverage lift 证据。

如果现在执行 Stage5C 或 SMC，会把一个受保护的 2.5D selector/world-state candidate 误包装成 generative/foundation/world model，这是不诚实的。

## 8. 当前最短下一步

1. **修 UCY candidate source**  
   训练或缓存 UCY-aware / source-agnostic candidate prediction source。Stage42-T 已证明 UCY 不是 threshold 问题，而是没有非 floor prediction。

2. **减少 safety floor 依赖**  
   做 proximity-safe internal gate / collision-aware gate，目标是让 neural 自己学会“不切错”，而不是完全依赖 Stage37/teacher floor。

3. **继续 full-waypoint retrained ablation**  
   同一 protocol 下补 JEPA、Transformer、hybrid、no-scene、no-goal、no-interaction、no-fallback、full-waypoint-shape 重训。

4. **做 metric/time audit**  
   FPS、annotation stride、homography、meter-per-pixel 没证据前，继续禁止 metric / seconds-level claim。

5. **扩充合法外部 top-down 数据**  
   需要更多真实 top-down pedestrian/drone scene+trajectory 数据，不能只靠当前 converted external 状态。

## 9. 关键证据文件

- 本 README：`/Users/yangyue/Downloads/World/README_M3W_GOAL_SUMMARY_ZH.md`
- 中文长账本：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md`
- M3W-Neural v1 report：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/report_m3w_neural_v1.md`
- evidence matrix：`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.md`
- Stage37 final：`/Users/yangyue/Downloads/World/outputs/stage37_t50_history/report_stage37_final.md`
- Stage38 final：`/Users/yangyue/Downloads/World/outputs/stage38_external_robustness/report_stage38_final.md`
- Stage39 final：`/Users/yangyue/Downloads/World/outputs/stage39_neural_dynamics/report_stage39_final.md`
- Stage40 final：`/Users/yangyue/Downloads/World/outputs/stage40_neural_optimization/report_stage40_final.md`
- Stage42 final：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/report_stage42_final.md`
- Stage42-R row cache：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/row_prediction_cache_stage42.md`
- Stage42-S frozen policy：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/frozen_row_combo_policy_stage42.md`
- Stage42-T UCY diagnosis：`/Users/yangyue/Downloads/World/outputs/stage42_long_research/ucy_unseen_transfer_stage42.md`

## 10. 最终口径

可以说：

- M3W 已经从 SDD-only selector 推进到 protected external 2.5D neural world-state candidate。
- Stage37 修复了 external t+50 deployability。
- Stage41/M3W-Neural v1 在 protected setting 下有稳定正提升。
- Stage42-C/R/S 提供了 full-waypoint、row-cache combo 和 frozen policy 分支证据。

必须说：

- 当前不是 true 3D。
- 当前不是 foundation。
- 当前不是 metric / seconds-level。
- 当前不是 ungated neural deployment。
- Stage5C 和 SMC 没有执行。
- UCY 在 Stage42-R/S combo 中仍 fallback-only；Stage42-T 诊断为缺少非 floor candidate source。

当前一句话：

```text
M3W 当前是一个强的、受保护的、dataset-local raw-frame 2.5D 多智能体 world-state candidate；它已经有跨外部数据的正证据，但仍依赖 safety floor，UCY/full metric/true 3D/foundation 目标还没有完成。
```
