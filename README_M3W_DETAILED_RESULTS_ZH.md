# M3W 目标内执行路线、失败原因、成功证据总览

更新时间：2026-05-26
工作目录：`/Users/yangyue/Downloads/World`
结果来源：`cached_verified` 汇总历史阶段报告、README、gate report 和 `research_state.json`；Stage42-BD 本地 t100 source inventory 为 `fresh_local_path_inventory`，Stage42-BE/BF/BG 为本轮连续 fresh evidence。

这份 README 是给人的一页式研究复盘：在 M3W 这个长期目标里做了什么、试了哪些路线、哪些失败、失败原因是什么、哪些成功、当前最好可部署模型是谁、哪些 claim 仍然禁止。

## 0. 不能越界的事实

当前仍必须诚实承认：

- M3W 不是 true 3D world model。
- M3W 不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D multi-agent world-state candidate。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- External top-down 数据仍是 dataset-local / unverified weak-metric diagnostic，不是统一米制世界坐标。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography、metric scale、effective seconds 没有完成全局验证。
- self-audited / visual-prior / auto-silver 标签不是 human gold。
- JEPA 是 representation / auxiliary，不是 latent generative rollout。
- Stage5C latent generative 没有执行。
- SMC 没有启用。
- 无保护 neural dynamics 不部署；所有强结果都依赖 safety floor / conservative fallback。

当前最准确的一句话：

```text
M3W 已经从 SDD-only selector scaffold 推进到 protected dataset-local raw-frame 2.5D multi-agent world-state candidate。
当前最强可部署证据来自 Stage37 / teacher safety floor 保护下的 baseline-family / row-level full-waypoint policy。
它还不是 true 3D、foundation、metric 或 seconds-level world model。
```

## 1. 当前 best deployable

当前最强可部署路线：

```text
M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
deployment protection = Stage37 selector / teacher safety floor
dominant mechanism = causal baseline-family rollout context + validation-safe gain/harm/easy guard
```

它可以写成：

```text
protected dataset-local raw-frame 2.5D multi-agent world-state candidate
```

不能写成：

```text
true 3D world model
foundation world model
metric trajectory predictor
seconds-level long-horizon predictor
ungated neural dynamics model
latent generative world model
SMC-ready model
```

## 2. 主要尝试路线

| 路线 | 做了什么 | 结果 | 原因 |
| --- | --- | --- | --- |
| Early BPSG-MA / 2.5D scaffold | 建 per-agent multi-agent trajectory world-state、strongest causal baseline fallback、failure diagnostics。 | 成功作为基座。 | 可运行、可审计、可 fallback，但不是 3D/foundation。 |
| JEPA / WAM-style representation | 训练 SAM-JEPA-2.5D、WAM-style data registry、后续多轮 non-collapse/downstream 检查。 | 失败为主。 | JEPA non-collapse，但 selector/failure/correction/t+50 没有稳定 downstream lift。 |
| SDD official pixel benchmark | SDD 转 per-video world-state shards；建 scene packs、episodes、GoalBench、HardBench、BaselineFailureBench。 | 成功。 | SDD 成为 official pixel-space raw-frame benchmark；但无 verified homography/scale。 |
| Stage24 hard selector | 利用大 oracle headroom 训练 validation-selected selector。 | 失败。 | t+50 improvement = -43.3%，easy degradation = 11.33%；hard label 低 margin、过度切换。 |
| Stage25/26 cost-aware selector | 从 hard class 改为 expected-FDE / regret / gain-harm / fallback-safe selector。 | 成功。 | t+50 约 +14.58%，hard/failure 约 +11.23%，easy degradation 约 1.81%。 |
| Stage31/32 external zero-shot/domain alignment | SDD selector / latent 迁移到 OpenTraj/ETH-UCY/UCY/TrajNet；做 normalization/CORAL/adapter。 | 失败。 | 坐标、horizon、scene/goal、agent type、scale/homography 都不一致；alignment 不等于任务目标对齐。 |
| Stage33/34 external geometry | 坐标不变特征、relative targets、row geometry、train-only goals、scene packs。 | 局部成功，不可部署。 | t50/hard 有正信号，但 all/easy 不稳。 |
| Stage35 selective transfer | external hard/easy/failure labels，gain/harm/easy gate。 | 部分成功。 | all +12.13%、hard/failure +13.98%、easy 0.041%，但 t+50 = 0。 |
| Stage37 causal history + goal prototypes | past-only K=8/16/32/64 history、scene-agnostic goal prototypes、switchability/gain/harm/conformal safety。 | 成功。 | all +13.48%，t+50 +8.46%，t50 CI [+7.69%, +9.15%]，hard/failure +15.54%，easy 0.041%，16/16 gates。 |
| Stage38 bounded correction | 在 Stage37 policy 保护下训练 bounded correction。 | 不部署。 | correction 没有安全超过 Stage37，residual 容易伤 easy。 |
| Stage39/40 Transformer / JEPA / Hybrid neural | 训练 Causal Transformer、JEPA auxiliary、Hybrid、teacher distillation、多任务 loss。 | 诊断为主。 | 无保护 neural 不安全；受保护 neural 没稳定超过 Stage37；JEPA downstream lift 未证明。 |
| Stage41/42 protected full-waypoint / source-level evidence | composite-tail、full-waypoint dynamics、row cache、source-level split、full-waypoint eval、retrained ablations、baseline-family mechanism。 | 成功形成 evidence package。 | 强结果来自 safety floor + baseline-family rollout context + validation-only policy，而不是自由 neural rollout。 |

## 3. 失败路线和根因

| 失败路线 | 失败表现 | 根因 | 后续修复 |
| --- | --- | --- | --- |
| JEPA-only | non-collapse，但下游 lift 无。 | 表征目标和 selector/failure/gain/harm 决策目标不对齐。 | 保留为 auxiliary/diagnostic，不作为当前主贡献。 |
| Hard classification selector | oracle headroom 大，trained selector 反而负，easy 被伤。 | low-margin oracle label、label ambiguity、confidence calibration 差、过度切换。 | 改成 expected-FDE / regret / fallback-safe selector。 |
| SDD -> external zero-shot | all/t50 严重负。 | SDD pixel-space 与 external dataset-local 坐标/尺度/horizon/goal/agent type 不兼容。 | 建 external geometry、relative target、history、goal prototype。 |
| 普通 normalization / latent adapter | 分布距离缩小但预测无 lift。 | 统计分布对齐不等于 baseline choice / gain/harm 目标对齐。 | 转向 row geometry 和 selective transfer。 |
| Stage35/36 t+50 | all/hard/easy 正，但 t50 = 0。 | t50 有 oracle headroom，但 feature/context 不够，policy 不敢安全切换。 | Stage37 加 history windows、goal prototypes、switchability。 |
| Bounded correction / ordinary residual | 没有稳定超过 Stage37。 | 直接改 trajectory 风险高，容易伤 easy；bounded 后收益不足。 | 不部署 correction，只作为 diagnostic。 |
| Ungated neural dynamics | easy harm / 不安全。 | 没有 safety floor 时，neural output 对 easy/fallback 样本伤害大。 | 保留 Stage37/teacher floor。 |
| Endpoint-to-full bridge | endpoint 成功不能转 full-waypoint。 | endpoint 不等于 trajectory shape，线性桥接不可靠。 | 改成直接 full-waypoint training/eval。 |
| History/goal/neighbor/graph 独立贡献 | 多轮 source-level ablation 未证明增量。 | 当前 evidence 被 baseline-family rollout context 主导。 | 下一步需要更强 graph/scene-rich neural protocol。 |
| t100 稳健正收益 | AY 一度有正 t100，但 AZ/BA 更严验证后不稳。 | 独立 t100 source-CV 支持不足，easy safety 不稳。 | source-CV guard 把 t100 回退为 0；需要更多 t100-capable sources。 |

## 4. 成功路线和关键证据

| 成功路线 | 关键结果 | 可信边界 |
| --- | ---: | --- |
| SDD official benchmark | 8 scenes / 60 videos / 10,300 tracks / 10,616,256 rows；no-leakage pass。 | SDD pixel-space raw-frame，不是 metric。 |
| Stage26 SDD selector | t+50 约 +14.58%；hard/failure 约 +11.23%；easy degradation 约 1.81%。 | SDD best deployable selector。 |
| Stage37 external t50 repair | all +13.48%；t50 +8.46%；t50 CI [+7.69%, +9.15%]；hard/failure +15.54%；easy 0.041%。 | external deployable selector candidate，dataset-local/raw-frame。 |
| M3W-Neural v1 protected package | all ADE +21.03%；t50 +13.65%；t100 raw diagnostic +14.69%；hard/failure +20.38%；easy 0。 | protected deployment，不是 ungated neural。 |
| Stage42-AM source-level full-waypoint | test rows 47,458；ADE all +24.58%；t50 +22.02%；t100 raw diagnostic +14.37%；hard/failure +23.75%；easy -25.66%。 | fresh source-level raw-frame full-waypoint probe。 |
| Stage42-AU baseline-family mechanism | `family_baseline_rel_only` all +27.38%、t50 +23.73%；`baseline_family_all` all +28.78%、t50 +31.54%。 | 证明当前 source-level 主机制是 baseline-family rollout context。 |
| Stage42-AW UCY validation repair | UCY all +37.45%；t50 +24.53%；hard/failure +35.51%；easy negative。 | train-only internal validation 修复 UCY support。 |
| Stage42-AX repaired robustness | global CI lows: all +35.31%、t50 +28.54%、t100 raw +20.29%、hard +33.52%。 | 两个外部 domain positive；但 h100 easy weak slice。 |
| Stage42-AY t100 easy safety | h100 easy degradation 修到 -0.650%，CI high 0.983%；all +30.55%；t50 +28.97%；hard +27.98%。 | 更安全，但 AZ/BA 后 t100 仍不能主 claim。 |
| Stage42-BA train-only t100 source-CV | all +28.10%；t50 +28.97%；hard +25.16%；easy -37.24%；t100 raw diagnostic 0。 | t100 positive 缺 source-CV 支持；安全回退。 |
| Stage42-BB t100 data gap audit | ETH_UCY 需 2 个额外 t100-safe sources；TrajNet 需 1 个；UCY 需 1 个。 | 把 t100 blocker 转成行动清单。 |
| Stage42-BC source acquisition plan | 6 candidates，5 official sources，6 local paths；high priority = UCY Crowd / TrajNet++ / OpenTraj / ETH-UCY。 | 不自动下载受限数据，不绕 license。 |
| Stage42-BD local t100 inventory | 93 files scanned；74 parseable；8 t100-capable；4 already used；4 novel candidates；estimated novel t100 windows = 6,257。 | 仅 inventory；还没 conversion/eval。 |
| Stage42-BE local t100 conversion readiness | 4 candidates；4 schema-ready；estimated t50 windows = 15,813；estimated t100 windows = 6,257；UCY source-CV feasible after conversion。 | 仍是 readiness；full feature store / source-CV / eval 还没跑。 |
| Stage42-BF local t100 schema conversion | 4 sources converted in-memory；t50 eval windows = 15,058；t100 eval windows = 6,071；UCY source-CV baseline-family positive，mean +60.70%，min +49.15% vs constant velocity。 | 这是 causal baseline/source-CV audit，不是 M3W policy training；t100 claim 仍 blocked。 |
| Stage42-BG local t100 protected policy | validation-selected protected baseline-family policy；UCY t100 source-CV mean +44.09%，min +43.86%，max easy degradation 1.13%；13/13 gates。 | UCY local support positive；ETH_UCY 仍 blocked；global t100 claim 仍 forbidden。 |
| Stage42-BH independent-source audit | 8 个 t100-capable files 去重为 5 个 independent sources；UCY mean +48.34%，min +34.06%，但 max easy degradation 6.33%；13/14 gates。 | 更严格 source 去重后 UCY 仍未 easy-safe；ETH_UCY/TrajNet 仍是 hard blocker。 |

## 5. Stage42-BD 本轮新增发现

Stage42-BD 做的是本机本地路径 inventory，不训练、不下载、不改变模型 claim。它扫描 OpenTraj / ETH / UCY / TrajNet 本地目录，识别下一步可能进入 Stage42-BE conversion/no-leakage/source-CV 的 t100 candidate。

```text
source = fresh_local_path_inventory
verdict = stage42_bd_local_t100_source_inventory_pass
gates = 10 / 10
files_scanned = 93
parseable_files = 74
t100_capable_files = 8
already_used_t100_files = 4
novel_t100_candidate_files = 4
estimated_novel_t100_windows = 6257
stage42_be_conversion_recommended = true
```

Top novel t100 candidates:

| local file | domain | rows | agents | max track | estimated t100 windows |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETH/seq_eth/biwi_eth_10fps.txt` | ETH_UCY | 5,492 | 360 | 114 | 14 |
| `UCY/students01/students001.txt` | UCY_or_ETH_UCY | 21,813 | 415 | 352 | 1,949 |
| `UCY/students03/obsmat_px.txt` | UCY_or_ETH_UCY | 21,859 | 428 | 540 | 3,415 |
| `UCY/students03/students003.txt` | UCY_or_ETH_UCY | 17,953 | 434 | 289 | 879 |

解释：

- 这些只是 local candidates，不是 converted dataset，不是 evaluated result。
- 下一步必须做 Stage42-BE conversion、train/val/test split、no-leakage、source-CV。
- 如果它们通过，才可能改变 t100 blocker；现在还不能把 t100 写成已修复。

## 6. Stage42-BE 本轮继续推进

Stage42-BE 把 BD 的 4 个 local candidates 从“文件存在”推进到“schema conversion readiness / no-leakage readiness”：

```text
source = fresh_local_conversion_readiness
verdict = stage42_be_local_t100_conversion_readiness_pass
gates = 12 / 12
candidate_files = 4
schema_conversion_ready_files = 4
estimated_t10_windows = 51061
estimated_t25_windows = 32451
estimated_t50_windows = 15813
estimated_t100_windows = 6257
domains_with_source_cv_after_conversion = UCY
full_feature_store_written = false
training_run = false
evaluation_run = false
```

关键解释：

- `UCY/students01/students001.txt`、`UCY/students03/obsmat_px.txt`、`UCY/students03/students003.txt` 三个 UCY-like sources 在 actual conversion 后足够构成 leave-one-source-out source-CV readiness。
- `ETH/seq_eth/biwi_eth_10fps.txt` 只有 14 个 estimated t100 windows，只能补一点 ETH_UCY 支持，单独不足以修 ETH_UCY t100 blocker。
- 本步骤不写 full feature store，不训练，不评估，不改变 t100 claim。
- 下一步仍必须是 Stage42-BF actual schema conversion + no-leakage + train-only source-CV。

## 7. Stage42-BF 本轮 actual in-memory conversion

Stage42-BF 真正解析这 4 个本地候选源，修正了 `UCY/students03/obsmat_px.txt` 的 8 列坐标布局风险，并在内存中构建 causal windows，计算 baseline-family FDE 和 UCY source-CV holdout baseline audit：

```text
source = fresh_in_memory_schema_conversion
verdict = stage42_bf_local_t100_schema_conversion_pass
gates = 12 / 12
candidate_sources = 4
converted_sources = 4
t50_eval_windows = 15058
t100_eval_windows = 6071
source_cv_domains_evaluated = ETH_UCY, UCY
source_cv_domains_positive_vs_constant_velocity = UCY
UCY mean holdout improvement vs constant_velocity = 0.607043
UCY minimum holdout improvement vs constant_velocity = 0.491545
materialized_feature_store_written = false
training_run = false
t100_positive_claim_allowed = false
```

解释：

- 这一步已经不是单纯 readiness：它做了 actual in-memory schema conversion 和 causal baseline audit。
- UCY 三源 source-CV 下，validation-selected baseline-family 在 holdout t100 上全部强于 constant velocity。
- ETH_UCY 没有足够 folds，仍不能修复 ETH_UCY t100 blocker。
- 这仍不是 M3W protected policy training，因此不能把 t100 写成已修复。下一步应是 Stage42-BG：在这些转换源上训练/评估 protected policy，并继续保持 no-leakage 和 validation-only threshold。

## 8. Stage42-BG 本轮 protected policy source-CV

Stage42-BG 把 BF 的 in-memory conversion 继续推进到 validation-selected protected policy source-CV：

```text
source = fresh_source_cv_protected_policy
verdict = stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker
gates = 13 / 13
candidate_sources = 4
t50_policy_windows = 15058
t100_policy_windows = 6071
source_cv_domains_evaluated = UCY
source_cv_domains_blocked = ETH_UCY
UCY_t100_source_cv_supported = true
UCY_t100_mean_improvement_vs_fallback = 0.440938
UCY_t100_min_improvement_vs_fallback = 0.438579
UCY_t100_max_easy_degradation = 0.011340
global_t100_positive_claim_allowed = false
```

解释：

- 这一步确实训练/选择了一个轻量 protected baseline-family policy，但不是神经模型训练。
- selection 只用 train / validation source，holdout source 只评估一次。
- UCY local t100 source-CV 是 positive 且 easy-safe。
- t50 在这个新本地 source-CV 协议下不全 easy-safe，因此不能把 BG 写成全面 horizon success。
- ETH_UCY 只有 1 个 t100-capable source，TrajNet 不在这批新本地候选里，所以 global t100 positive claim 仍 blocked。

## 9. Stage42-BH 本轮 independent-source audit

Stage42-BH 更严格地审计“独立 source”：

```text
source = fresh_local_independent_source_audit
verdict = stage42_bh_independent_t100_source_audit_partial
gates = 13 / 14
raw_t100_capable_files = 8
independent_t100_sources = 5
duplicate_or_alternate_format_group_count = 2
UCY_independent_sources = 4
ETH_UCY_independent_sources = 1
TrajNet_independent_sources = 0
UCY_t100_mean_improvement_vs_fallback = 0.483414
UCY_t100_min_improvement_vs_fallback = 0.340559
UCY_t100_max_easy_degradation = 0.063323
```

解释：

- `UCY/students03/obsmat.txt`、`obsmat_px.txt`、`students003.txt` 被视为同一 independent source group，不再当成三个独立泛化源。
- `ETH/seq_eth/obsmat.txt` 和 `biwi_eth_10fps.txt` 也被视为同一 ETH source group。
- 去重后 UCY 有 4 个 independent t100 sources，均值收益仍为正，但 max easy degradation = 6.33%，超过 2% gate。
- 这说明 BG 的 UCY positive/easy-safe 是较宽松 source 定义下的支持；BH 更严格后，UCY t100 仍需要 easy/harm guard repair。
- ETH_UCY 只有 1 个 independent source，TrajNet 为 0，global t100 继续 blocked。

## 10. 为什么当前成果不是 true 3D / foundation

不能这么写的原因很具体：

- 没有 verified global homography / scale / meter-per-pixel。
- SDD 与 external 坐标体系不统一。
- t+50/t+100 是 raw-frame，不是 seconds-level。
- 数据仍主要是 top-down trajectory / dataset-local coordinate，不是多源大规模真实 3D scene/video foundation pretraining。
- JEPA 没有稳定 downstream lift。
- Transformer/Hybrid neural 没有无保护超过 Stage37。
- 当前 strongest deployable 仍是 protected policy / safety floor 体系。

因此当前正确论文定位应是：

```text
strict no-leakage protected 2.5D multi-agent world-state modeling on real top-down raw-frame trajectory data
```

而不是：

```text
true 3D foundation world model
```

## 11. 现在最值得做的下一步

1. **Stage42-BI：修复 BH 暴露的 UCY independent-source easy degradation。**
   需要 source-robust easy/harm guard，而不是只看 validation source gain。

2. **继续 t100 source-CV repair，但不 overclaim。**
   只有 ETH_UCY / TrajNet / UCY 至少有足够独立 t100 source support，t100 才能从 diagnostic blocker 变成可部署正 claim。

3. **如果继续冲神经世界模型主贡献，要换更强 graph/scene-rich protocol。**
   当前 ridge/MLP/Conv1D/hand-built graph residual context 都没证明独立增量；不能继续把 history/goal/neighbor 写成已证明主贡献。

## 12. 最终简短 verdict

```text
项目是否跑通：是
当前 best deployable：M3W-Neural v1 protected policy under Stage37 / teacher safety floor
是否 true 3D：否
是否 foundation：否
是否 metric：否
是否 seconds-level：否
是否 Stage5C executed：否
是否 SMC enabled：否
SDD success：是
external t50 success：是
protected source-level full-waypoint success：是
t100 stable success：否，当前仍是 blocker / diagnostic
本轮新发现：4 个 novel t100 candidates 已完成 in-memory schema conversion；Stage42-BG 在宽松 UCY local source-CV 上得到 protected t100 policy positive/easy-safe 支持；Stage42-BH 去重后发现 UCY independent-source t100 仍有 easy degradation blocker，ETH_UCY/TrajNet 支持不足，global t100 stable success 仍未允许
```
