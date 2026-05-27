# M3W 长期目标完整总结：路线、失败、成功与当前质量

更新时间：2026-05-27

工作目录：`/Users/yangyue/Downloads/World`

结果来源：`cached_verified` 汇总既有 Stage18-Stage42 报告、gate、README、`research_state.json`；最近 Stage42-HC / Stage42-HD 为 `fresh_run`，其余旧阶段不伪装成新跑结果。

本文用途：按用户要求，把“在 M3W 这个长期目标内做了什么、尝试了哪些路线、哪些失败了、为什么失败、哪些成功了、当前大概是什么质量”集中写到一个 README。本文是总结文件，不是新训练；不会把 diagnostic 写成 deployable，不会把 not_run 写成完成。

## 1. 最短结论

当前 M3W 最诚实定位是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

它已经从 SDD-only selector scaffold 推进到：

- SDD pixel/raw-frame 上有可部署的 Stage26 cost-aware selector。
- External top-down dataset-local/raw-frame 上有可部署的 Stage37 t+50 safe selector。
- Stage41/42 有 protected neural / full-waypoint / group-consistency / source-domain policy 证据。
- Stage42-HC/HD 进一步证明：不使用 teacher gate 的 floor-free gate 原本不安全，但可以用 validation-selected proximity guard 修复成 teacherless switch gate；不过仍然需要 causal floor fallback，不允许 global floor removal。

它仍然不是：

- true 3D world model
- large-scale foundation world model
- metric / meter-level predictor
- seconds-level long-horizon predictor
- ungated neural dynamics deployable model
- Stage5C latent generative execution
- SMC-ready system

## 2. 当前 best deployable 分层

| 用途 | 当前最强结果 | 是否可部署 | 必须保留的边界 |
| --- | --- | --- | --- |
| SDD pixel/raw-frame benchmark | Stage26 cost-aware expected-FDE selector | 是 | 仅 SDD pixel/raw-frame；不是 metric。 |
| External t+50 dataset-local/raw-frame | Stage37 causal-history + goal-prototype safe selector | 是 | External selector-level success；不是 true 3D。 |
| Protected neural / world-state | M3W-Neural v1 / Stage41-42 protected candidate | 候选可报告 | 依赖 Stage37/teacher/floor safety；不是 ungated neural deployment。 |
| Source/domain/full-waypoint | Stage42-FH/FI frozen protected policy family | 可作为 protected evidence | 允许 source/domain robust claim；不允许 uniform horizon claim。 |
| Floor-free / teacherless switch gate | Stage42-HD proximity-guard repaired gate | 有条件可报告 | teacher gate 不用，但 causal floor fallback 仍必须存在；不是 global floor removal。 |
| h100 / uniform horizon | blocked / partial only | 否 | TrajNet|100、UCY|100 仍需 source/legal/long-horizon support。 |

## 3. 我尝试过的路线总览

| 路线 | 做了什么 | 当前结论 |
| --- | --- | --- |
| 数据发现与许可 | Stage19/20 建 WAM-style registry、web acquisition、license audit、download/user-action plan。 | 框架成立；registry-only 不算 converted；legal terms 未确认不能算可用数据。 |
| SDD official benchmark | Stage21/22 把 SDD 转成 world-state shards、scene packs、episodes、HardBench、FailureBench、GoalBench、baselines。 | 成功建立 SDD pixel/raw-frame benchmark；不是 metric/seconds-level。 |
| Strong causal baseline | 系统比较 constant position、causal velocity、damped velocity、acceleration、turn-rate、scene/goal baseline。 | 成为全项目 safety floor；后续学习组件必须超过它且不能伤 easy。 |
| Hard-class selector | 直接预测 oracle-best baseline class。 | 失败，Stage24 t50 约 `-43.3%`，easy degradation 约 `11.33%`。原因是低 margin 样本被强行硬分类。 |
| Cost-aware selector | 预测 expected FDE/risk，按 confidence/gain/harm fallback。 | 成功，Stage26 在 SDD 上成为 best deployable。 |
| JEPA representation | Stage18/19/22/23/M3W 训练 trajectory/scene/interaction JEPA。 | 多次 non-collapse，但无稳定 downstream lift；不能作为主贡献。 |
| Transformer / Hybrid neural | 训练 Transformer-only、JEPA-only、Hybrid、protected neural dynamics、full-waypoint sequence。 | 无保护 neural 不部署；protected neural 有证据但仍依赖 safety floor。 |
| External zero-shot | SDD-trained selector/latent 直接迁移到 external top-down。 | 大失败，Stage31 all 约 `-92.67%`，t50 约 `-278.57%`。坐标、scale、horizon、agent type、scene/goal context 不兼容。 |
| Domain normalization / latent adapter | per-scene zscore、velocity-scale、relative target、CORAL/linear adapter。 | latent gap 可缩小，但 predictive lift 不稳定；不能写成迁移成功。 |
| External row geometry / goals | Stage34 补 current/past/future-label geometry、train-only goals、scene packs。 | 提供局部 t50/hard 正信号，但 all/easy 不稳，不能部署。 |
| Selective transfer | Stage35 hard/easy/failure/gain/harm selector，低风险 fallback。 | all/hard/easy 过，但 t50 仍 0，因此不可部署。 |
| t+50 history/prototype repair | Stage37 构建 past-only history windows 与 scene-agnostic goal prototypes。 | 成功修复 external t50，gate 16/16。 |
| Bounded correction / residual | 尝试 selected baseline + bounded delta。 | 未安全超过 Stage37，普通 residual 容易伤 easy；不部署。 |
| Safety / physical validity | near@0.05、jagged rate、proximity guard、group consistency、safe switch。 | 成功建立安全评估；多条高收益路线因 proximity/easy 不过而不 promoted。 |
| Source/domain/full-waypoint | Stage42 做 endpoint bridge、full-waypoint shape、group-consistency、source/domain policy、freeze/replay/bootstrap。 | 当前最强 protected source/domain evidence；仍禁止 uniform horizon claim。 |
| Claim guard / paper boundary | module ledger、claim linter、overclaim linter、evidence matrix、source action package。 | 成功锁定哪些能写、哪些不能写，避免把失败包装成成功。 |

## 4. 阶段级关键结果

### Stage18-19：JEPA 与 WAM-style data

- Stage18 SAM-JEPA-2.5D non-collapse，但没有改善 selector、failure predictor、correction、official t+50。
- Stage19 建 WAM-style data registry，把 simulation、top-down real trajectory、human/egocentric video 分清角色。
- 结论：当时最大问题不是继续加模型，而是缺真实 raw scene/video + trajectory + long-horizon data；JEPA 不能当 latent rollout。

### Stage20-21：联网数据与 SDD 接入

- Stage20 做 web dataset acquisition / license audit / user action plan。
- Stage21 完成 SDD 本地转换：
  - SDD scenes = `8`
  - SDD videos = `60`
  - tracks = `10,300`
  - world-state rows = `10,616,256`
  - raw-frame t+50 samples = `10,009,005`
  - raw-frame t+100 samples = `9,497,463`
  - split = train `40` videos / val `4` videos / test `16` videos
  - no-leakage audit = pass
  - velocity = causal finite difference
  - coordinate status = pixel-space
- 结论：SDD 成为 official pixel-space benchmark，但不能说 metric 或 seconds-level。

### Stage22-23：SDD quick / quick-plus

- Stage22 建 SDD scene packs、episodes、HardBench、FailureBench、GoalBench。
- quick windows = `27,600`，t50 episodes = `6,733`，t100 episodes = `6,319`。
- strongest causal baseline = `damped_velocity`。
- SDD FDE raw-frame pixel-space：
  - t10 = `5.7843`
  - t25 = `12.9896`
  - t50 = `29.4944`
  - t100 = `60.5580`
- Stage22 quick selector 没过 gate；failure predictor AUROC 约 `0.6093`；JEPA non-collapse 但无 lift。
- Stage23 因 I/O 慢跑的是 quick-plus，不是 full medium；selector t50 improvement 约 `2.66%`，未达 5%；failure AUROC 约 `0.6498`，未达 0.75；JEPA 仍无 lift。
- 结论：quick-plus 不能替代 medium，下一步必须先修 I/O。

### Stage24：I/O 修复与 true medium

- SDD I/O fast cache 加速约 `12.66x`。
- True medium index 建立：
  - cross_scene train/val/test = `200k / 50k / 50k`
  - within_scene train/val/test = `200k / 50k / 50k`
  - total indexed windows = `600,000`
- Selector oracle headroom 很大：约 `46.2%`。
- Validation-selected hard-class selector 失败：t+50 improvement = `-43.3%`。
- easy degradation = `11.33%`。
- Failure predictor 成功过 gate：AUROC = `0.8715`。
- JEPA 仍无 downstream lift。
- 结论：不是没有 oracle 空间，而是 hard-class selector 目标错了。

### Stage25-26：从 hard classification 到 cost-aware selector

- Stage25 做 selector failure forensics：低 margin、label ambiguity、easy over-switch、horizon/split/agent mixing 是核心。
- 改成 expected-FDE / regret-aware / confidence-gated / conservative fallback。
- Stage26 成功：
  - t+50 improvement 约 `+14.58%`
  - hard/failure improvement 约 `+11.23%`
  - easy degradation 约 `+1.81%`
- 结论：Stage26 是 SDD 当前 best deployable baseline/selector。

### Stage27-30：M3W evidence sprint 与 SDD fresh recheck

- 建 M3W token/schema/dataloader/model/eval/gates/paper package。
- 尝试 JEPA-only、Transformer-only、Hybrid。
- bounded optimization 后仍不能把 M3W small/hybrid 写成 CCF-A candidate。
- Stage30 fresh recheck 支持 Stage26/Stage28 SDD 候选：
  - t50 fresh recheck 约 `+16.86%`
  - hard 约 `+13.36%`
  - easy 约 `+1.93%`
  - bootstrap t50 CI low 约 `+16.01%`，高于 Stage26 `+14.58%`
- 结论：SDD 上成立，但外部验证仍是 blocker。

### Stage31-34：外部迁移失败与诊断

- Stage31 external feature store/no-leakage/latent cache 建立。
- External strongest baseline = `constant_velocity_causal_fd`。
- Zero-shot M3W-LAS external transfer 失败：
  - all improvement 约 `-92.67%`
  - t50 约 `-278.57%`
- Adapted selector 约 0 improvement。
- Stage32/33 normalization、relative target、domain adapter、coordinate-invariant features 仍不足。
- Stage34 补 external row geometry 和 train-only goals，出现局部正信号：
  - t50 diagnostic lift 约 `+6.6%`
  - hard/failure 约 `+18%` 到 `+25%`
  - 但 all-test 为负、easy degradation 高，最终 fallback `0.0`
- 结论：外部不是无空间，而是必须专门建 hard/easy/failure 和 long-horizon safety policy。

### Stage35-37：External selective transfer 与 t+50 修复

- Stage35 selective transfer：
  - all improvement = `+12.13%`
  - hard/failure = `+13.98%`
  - easy degradation = `0.041%`
  - t+50 improvement = `0.0`
  - verdict = not deployable
- Stage36 t50 forensics：
  - external t50 rows = `16,263`
  - t50 oracle headroom 约 `22.98%`
  - 不是没有样本或 headroom，而是现有 features/context 无法安全切换。
- Stage37 修复：
  - past-only history windows K=8/16/32/64
  - scene-agnostic goal prototypes
  - t50 failure/gain/harm/switchability
  - conformal/easy safety
  - all improvement = `+13.48%`
  - t+50 improvement = `+8.46%`
  - t+50 bootstrap CI = `[+7.69%, +9.15%]`
  - hard/failure improvement = `+15.54%`
  - easy degradation = `0.041%`
  - gates = `16 / 16`
  - verdict = `stage37_t50_transfer_repaired_deployable`
- 结论：Stage37 是 external t50 当前 best deployable selector。

### Stage38-40：Correction 与神经动力学未超过 Stage37

- Stage38 冻结 Stage37 policy 并尝试 bounded correction / dynamics head。
- Correction with fallback 没有安全超过 Stage37，故不部署。
- Stage39 训练 Transformer / JEPA / Hybrid：
  - neural_with_fallback 没有超过 Stage37
  - t100 diagnostic 仍没有改善
  - ETH/TrajNet held-out blocker 仍存在
- Stage40 自动诊断/优化 neural world dynamics：
  - 发现 neural 常常是在复制 Stage37 或被 fallback 吃掉
  - 无保护 neural 灾难性失败
  - JEPA non-collapse 仍没有足够 downstream lift
- 结论：神经网络路线不是没价值，但当前不能替代 Stage37；必须在 protected/fallback 条件下报告。

### Stage41-42：Protected neural、full-waypoint、source/domain 和 claim guard

核心成功：

- M3W-Neural v1 protected candidate：
  - all `+21.03%`
  - t50 `+13.65%`
  - t100 raw `+14.69%`
  - hard/failure `+20.38%`
  - easy `0.00%`
  - gates `41 / 41`
  - 仍依赖 Stage37/teacher safety floor。
- Stage42-DL/DM runtime replay：
  - rows `47,458`
  - all/t50/t100raw/hard = `+24.72% / +22.36% / +14.35% / +23.89%`
  - near@0.05 从 `1.94%` 降到 `1.38%`
  - switch exact match true
- Stage42-FE constrained safety composer：
  - all/t50/hard = `26.41% / 23.15% / 24.81%`
  - near@0.05 = `1.32%`
- Stage42-FH UCY-supported source/domain composer：
  - all/t50/t100raw/hard = `34.98% / 28.97% / 20.57% / 33.10%`
  - TrajNet 与 UCY 都 positive-safe
  - gate `20 / 20`
- Stage42-FI freeze/replay：
  - policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`
  - replay diff = `0`
  - 2000-bootstrap CI low all/t50/t100raw/hard = `34.62% / 28.46% / 19.96% / 32.73%`
  - gate `25 / 25`
- Stage42-FU/GJ module claim lock：
  - 可作为主 claim 的模块：history、domain expert、safe switch、teacher floor、group-consistency full-waypoint、full-waypoint shape、endpoint bridge。
  - 不能作为主 claim 的模块：JEPA、Transformer、scene/goal、neighbor/interaction。
- Stage42-GZ/HA claim guards：
  - full-waypoint overclaim linter violations = `0`
  - 防止把 endpoint-only 或 ungated full-waypoint 写成 learned full-waypoint dynamics。

最新 Stage42-HC/HD：

- Stage42-HC floor alternative gate stress：
  - floor-free deployable count = `0`
  - floor-free positive-but-unsafe count = `6`
  - best raw floor-free candidate = `harm_predictor_gate`
  - best raw floor-free all/t50/hard = `+35.95% / +25.20% / +35.86%`
  - failure reason = near-collision delta over 1pp
  - best strict deployable teacher-dependent/current candidate = `current_composite_tail_policy`
  - teacher/current all/t50/t100raw/hard = `+21.03% / +13.65% / +14.69% / +20.38%`
  - decision = keep Stage37/teacher floor globally; allow only validation-backed partial t50 relaxation on `TrajNet|50`, `UCY|50`
  - gate `14 / 14`
- Stage42-HD floor-free proximity-guard repair:
  - pre-guard deployable count = `0`
  - post-guard deployable count = `4`
  - best repaired family = `harm_predictor_gate`
  - teacher gate used = `False`
  - causal floor fallback used = `True`
  - all/t50/t100raw/hard = `+20.74% / +13.82% / +13.68% / +19.99%`
  - easy degradation = `0.00%`
  - collision delta @0.05 = `-0.47%`
  - switch rate = `32.11%`
  - guarded-off rate = `18.44%`
  - by domain:
    - ETH_UCY: all `+18.10%`, t50 `+12.95%`, t100 `+11.53%`, hard `+17.65%`
    - TrajNet: all `+23.16%`, t50 `+17.05%`, t100 `+13.78%`, hard `+22.01%`
    - UCY: all `+24.02%`, t50 `+11.08%`, t100 `+19.68%`, hard `+23.36%`
  - gate `13 / 13`
- HD 结论：teacher gate 可以不使用，但 causal floor fallback 仍不可移除。这是 teacherless switch gate 的安全修复，不是 global floor removal，也不是 ungated neural deployment。

## 5. 失败路线与根因

| 失败/受阻路线 | 失败表现 | 根因 | 当前处理 |
| --- | --- | --- | --- |
| hard-class selector | Stage24 t50 `-43.3%`，easy degradation `11.33%` | oracle label low-margin、高歧义，hard class 迫使 easy cases 切错 | 改 expected-FDE / regret-aware / fallback-safe |
| JEPA 主线 | non-collapse 但 downstream 无稳定 lift | 表征目标与部署收益/风险目标错位 | 只能 auxiliary/diagnostic，不能主 claim |
| SDD->external zero-shot | all `-92.67%`，t50 `-278.57%` | 坐标/scale/horizon/agent type/scene context 不兼容 | 用 external history/prototype/selective transfer |
| Latent adapter | latent gap 缩小但 selector 不提升 | distribution alignment 不等于 task alignment | 不作为 predictive success |
| Stage34 early external policy | t50/hard 局部正，但 all/easy 不稳 | 缺 easy guard 和 t50 专用 switchability | Stage35/37 修复 |
| Stage35 selective transfer | all/hard/easy 过，但 t50=0 | all objective 淹没 long-horizon；缺 past-only t50 context | Stage37 history + prototype repair |
| Correction/residual | 不稳定超过 Stage37，容易伤 easy | 直接改轨迹比选择/回退风险高 | 不部署 correction |
| Unprotected Transformer/Hybrid | neural without fallback 不安全 | dataset-local/raw-frame grounding 不足，floor 太强 | 只报告 protected neural |
| scene/goal 独立主 claim | 不稳定，不能单独解释主要收益 | train-only goal proxy 对 held-out/domain shift 支持有限 | 只能辅助 |
| neighbor/interaction 独立主 claim | scalar features 不足以稳定提升 | interaction 表达太弱 | 用 group-consistency full-waypoint 替代 |
| uniform h100/horizon | TrajNet|100、UCY|100 仍 weak | low-margin ambiguity、source support 稀疏、long-horizon context 不足、legal blocker | 不写 uniform horizon claim，等 source/legal/long-horizon support |
| floor-free raw gate | raw gain 很高但 near-collision unsafe | 无 teacher/floor 保护时切换太激进 | Stage42-HD 加 validation proximity guard，但仍需要 causal floor fallback |

## 6. 成功路线与为什么成功

| 成功路线 | 为什么成功 | 代表数字 |
| --- | --- | --- |
| cost-aware selector | 不再硬预测 best class，而是预测 expected FDE/risk 并 fallback | Stage26 t50 `+14.58%`，hard `+11.23%`，easy `+1.81%` |
| Stage37 external t50 repair | 用 past-only history、goal prototypes、gain/harm/easy guard 修 long horizon | all `+13.48%`，t50 `+8.46%`，CI `[+7.69%, +9.15%]` |
| protected neural candidate | neural 不直接替代 floor，而是在 Stage37/teacher/floor 保护下切换 | M3W-Neural v1 all `+21.03%`，hard `+20.38%`，easy `0%` |
| group-consistency full-waypoint | 用 source/frame/horizon group-level consistency 修全体 rollout，而不只看单 agent endpoint | Stage42-DL/DM all `+24.72%`，near@0.05 `1.94% -> 1.38%` |
| constrained safety composer | 在 validation 上组合高精度候选与 safety fallback | Stage42-FE all/t50/hard `26.41% / 23.15% / 24.81%`，near@0.05 `1.32%` |
| source/domain composer | 增加 UCY train-only support，避免 TrajNet-only robust | Stage42-FH all/t50/t100raw/hard `34.98% / 28.97% / 20.57% / 33.10%` |
| frozen replay/bootstrap | policy hash、exact replay、CI 证明不是偶然 test tuning | Stage42-FI replay diff `0`，CI lows 全为正 |
| proximity-guard repaired teacherless gate | 保留 floor-free gate 的切换收益，同时用 validation proximity guard 关掉不安全切换 | Stage42-HD all/t50/hard `20.74% / 13.82% / 19.99%`，collision delta `-0.47%` |

## 7. 当前不能写的 claim

不能写：

- M3W 是 true 3D world model。
- M3W 是 large-scale foundation world model。
- SDD 或 external 已经是统一 metric / meter-level benchmark。
- t+50/t+100 是 seconds-level long horizon。
- JEPA 是生成式 world model 或 latent rollout。
- Transformer/Hybrid 已经无保护超过 Stage37 并可独立部署。
- Scene/goal 或 neighbor/interaction 是独立主贡献。
- Stage5C 已执行。
- SMC 已启用。
- Floor-free neural 可以全局部署。
- Registry-only、prefill、legal action package 等于 converted/evaluated dataset。

可以写：

- M3W 当前是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate。
- Cost-aware selector、causal history、goal prototype、safe switch、teacher/floor safety、group-consistency full-waypoint、source/domain policy 有实验证据。
- Stage37 在 external t50 上可部署。
- Stage42-FH/FI 在 source/domain protected policy 上有 frozen replay/bootstrap 证据。
- Stage42-HD/HE 证明 teacherless floor-free gate 可以经 proximity guard 修复并通过 2000-bootstrap/per-domain/per-horizon 稳健性审计，但仍需要 causal floor fallback。

## 8. 当前最重要的下一步

1. **继续关掉 h100/uniform horizon blocker。**
   TrajNet|100 与 UCY|100 仍需要合法 source、long-horizon support、row-level history/neighbor/goal context；不能靠调阈值硬写成功。

2. **把 HE 写进 paper-ready evidence package。**
   Stage42-HE 已把 HD 的 teacherless proximity-guard repaired gate 从单次 gate 升级成 robust evidence：all/t50/t100raw/hard bootstrap CI low 全为正，ETH_UCY/TrajNet/UCY 三个 domain robust positive，weak domain-horizon slices 为 none。

3. **保留 Stage37 / causal floor 作为部署底座。**
   当前任何 neural/floor-free/full-waypoint 变体都不能绕过 safety floor。Stage42-HD/HE 的价值是减少 teacher gate 依赖，不是移除 causal floor。

4. **神经网络路线继续，但只做 protected dynamics。**
   重点应是 gain/harm、switchability、group consistency、full-waypoint shape、proximity guard，而不是普通 residual 或无保护 rollout。

## 9. 一句话给用户

我们已经做过很多条路：硬分类 selector、JEPA、Transformer/Hybrid、correction、external transfer、domain adaptation、history/prototype、source/domain/full-waypoint、safety/proximity、claim guard。

真正成功的是：**cost-aware selector + past-only history/prototype + safe fallback + group-consistency/full-waypoint protected policy**。

真正失败的是：**硬分类、单纯 JEPA、无保护 neural、普通 residual、单纯 latent alignment、zero-shot external、把 scene/goal/neighbor 当独立主贡献、uniform h100 claim**。

当前 M3W 已经是一个有实证链的 protected 2.5D multi-agent world-state candidate；但它还不是 true 3D，也不是 foundation，也不能去掉 safety floor。

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
