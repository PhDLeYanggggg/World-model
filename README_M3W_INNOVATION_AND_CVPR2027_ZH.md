# M3W：创新点、证据边界与 CVPR 2027 研究路线

更新日期：2026-09-16。目标会议已确定为 **CVPR 2027 主会长文**，暂不安排 AAMAS。

**同日后续审计更正：** 已确认不是仅有潜在风险。Stage35/37 val 与 test 有 47,223 个逐行几何相同的窗口；Stage43/44 train 与 val 重复 47,223 个窗口、train 与 test 重复 9,540 个窗口，源文件字节哈希也相同。新 test 的 78,270 行来自旧 teacher train。Stage37 最终 variant 选择同样读取 test。这些外部结果降为探索性，不再作为独立部署或泛化证明。已修复 Stage44 后续选择流程并阻断旧缓存训练；尚未重新训练。详见 [数据来源审计](outputs/publication_readiness_2026_09/recording_lineage_audit.md)、[协议修复记录](outputs/publication_readiness_2026_09/protocol_repair.md)。

我希望这篇论文回答一个具体问题：**当强运动基线已经能处理大部分简单样本时，如何让神经预测只在有证据支持的情况下介入，并避免逐个 agent 的改进破坏整个场景的一致性？**

M3W 已经积累了实现基础和初步正结果，但现有证据尚不足以支持一篇完成的 CVPR 论文。最值得发展的贡献是相对基线的风险控制、场景级联合选择和支持不足时的回退。JEPA、Transformer、目标原型、坐标规范化本身都有充分先例，不能把它们的组合或名称当成创新。

## 1. 本次实际核实了什么

本次没有重新训练或重新计算真实数据预测。完成的是当前源码审查、历史报告读取、缓存与 checkpoint 哈希核对，以及两个合成输入诊断。历史真实数据指标仍是旧实验结果。

- 首次审计时 HEAD：`a48b1b7`；Stage44 实现提交：`9e6fa30`。下面的源码缺陷描述指修复前版本，不代表修复后的选择实现。
- Stage44 记录时间：2026-06-04；实际为 small，训练/验证/测试分别为 12,000 / 5,000 / 8,000 rows，七个模型各 3 epochs。
- 当前十二个直接输入缓存的组合哈希与 Stage44 记录一致；七个 checkpoint 的 SHA256 全部与记录一致。这验证了文件身份，**不等于已复现预测或完成端到端无泄露审计**。
- 底层 full-waypoint cache 有 146,809 / 101,446 / 89,736 rows，不能据此把上述 small 结果称为全量结果。
- 源文件 ID 在 split 间没有交集，但后续内容审计已确认不同 ID 下的字节相同录像跨 split 重复；不能把路径互斥当作独立录像互斥。
- 合成反向传播检查证实：Stage44 future target encoder 没有梯度，optimizer step 后参数不变。
- 合成模型排序检查证实：保持 validation 内容不变、只改变 test metrics，`_best_variant` 的选择会变化。

可复现本次检查：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_submission_evidence.py
```

证据：[机器可读审计](outputs/publication_readiness_2026_09/evidence_audit.json)、[审计说明](outputs/publication_readiness_2026_09/evidence_audit.md)。

## 2. 现在有哪些有价值的研究积累

| 路线 | 当前证据 | 创新判断 | 在论文中的合适位置 |
| --- | --- | --- | --- |
| Cost-aware selection：预测每个基线误差、gain/harm 后再切换 | Stage26/37 及后续 protected 系列有历史正信号 | 有实际价值，但单纯 cost-sensitive routing、fallback 并不新 | 方法基座；需要新的风险目标或联合选择机制 |
| Easy preservation 与强基线保护 | Stage37 历史结果 all +13.48%、t50 +8.46%、hard +15.54%、easy +0.041% | “困难样本提升不能以简单样本受损换取”是好的研究问题 | 主要实验设计与目标函数 |
| 完整 waypoint 和 group consistency | Stage42 有 runtime replay 和群体一致性报告 | 比端点预测更合理，但完整轨迹和联合指标已有先例 | 可发展为场景级安全介入贡献 |
| Past-only history 与场景无关目标原型 | Stage37 的 t50 历史正信号；有因果历史缓存 | 可以是有效设计，不宜单独声称首创 | 有严格对照的辅助模块 |
| WorldCore latent dynamics | Stage44 small 的部分 ablation 有神经增益信号 | 尚属探索性；存在选择流程和标签语义问题 | 后续重新验证的 predictor 候选 |
| 数据契约、跨源追溯、无泄露工具 | 代码和报告较完整 | 工程基础强；普通审计不能独立支撑方法创新 | 可复现材料；若发现系统性 benchmark 问题可形成附加研究 |

这些工作并非没有价值；真正缺的是把其中一个问题变成清楚的方法，再用独立实验排除普通 ensemble、阈值搜索、数据泄露和基线口径差异等解释。

## 3. Stage44 的结果应该怎样解读

| 历史 variant | all 改善 | t50 改善 | hard 改善 | easy degradation | t100 diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: |
| hybrid_no_scene | +37.49% | +20.32% | +38.82% | 1.06% | -9.99% |
| 完整 hybrid | +26.75% | +9.57% | +27.30% | 50.91% | -34.90% |
| no JEPA | +32.44% | +6.89% | +35.18% | 9.36% | -4.07% |
| no Transformer | +25.10% | +5.80% | +25.76% | 12.03% | -12.40% |

**这里的改善是逐行 scale-normalized、四个 waypoint 的 ADE，相对于由 selected endpoint 线性插值得到的 floor。** 不是直接对应 Stage37 的 FDE 改善，也不是已按社区标准协议获得的 SOTA。不同阶段的百分比不能相加或直接排成排行榜。

具体问题如下。

1. **Test 参与了最终 architecture 选择。** `src/stage44_worldcore.py` 的 `_best_variant` 读取 `test_eval`，`_needs_repair` 也读取 test。checkpoint 和阈值用 val 选，并不能抵消最终 variant 用 test 选的问题。本次回看发现 val 最优恰好也是 no_scene，但这不能重新创造独立确认集。当前七个模型记录未执行 repair 分支，不能谎称本轮发生过该分支训练。
2. **JEPA 实现不足以代表一个经过正确训练的 JEPA 方案。** target encoder 随机初始化后输出被 detach，当前模块没有 EMA 或另一条更新路径。合成诊断验证参数不变。因此不能由本实验推出“JEPA 方法本身无效”，也不能声称已证明 JEPA 表征创新。
3. **多任务名称大于标签实际含义。** interaction target 是 hard OR failure；density 是过去密度；physical validity 是未来 waypoint 标注是否齐全。这些不是独立的未来交互、occupancy 或物理正确性 ground truth。
4. **Gain/harm 与实际神经介入不匹配。** gain 标签来自候选基线 oracle 相对 strongest 的误差，harm 来自 easy/小 margin；它们不直接回答“这一次神经预测相对 floor 会不会变好/变坏”。
5. **当前 Transformer 处理的是八组聚合特征 token。** 可以叫 token fusion，但还不能仅凭名称声称已建立显式 agent-time 时空动力学。静态图特征与动态图 token 需要分开评价。
6. **16/16 gates 不是论文质量认证。** 有些 gate 只检查模型/文件存在，no-leakage gate 使用声明布尔值，未涵盖 test-selection 问题。
7. **独立统计证据仍不足。** Stage44 JSON 没有 bootstrap，当前 small 只有一个 seed。重叠轨迹窗口不能当作彼此独立的样本来获得很窄的 CI。

这些问题会改变论文可写的 claim，但不表示所有历史预测都是伪造，也不表示项目没有可学信号。

## 4. 与相关工作的差异在哪里

本次是针对性检索，非穷尽系统综述，也不能证明某个提议“从未有人做过”。

| 已有方向及原始文献 | 已经解决/提出的内容 | M3W 不能照搬为创新的内容 | 值得验证的差异 |
| --- | --- | --- | --- |
| [AgentFormer, ICCV 2021](https://ye-yuan.com/agentformer/) | agent-aware 时空预测 | 给历史和邻居加 Transformer | 相对强基线的介入风险与场景级选择 |
| [EqMotion, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html) | 几何等变运动与不变交互推理 | rotation/translation invariant 特征 | 在变换下保持 switch decision 和风险预算一致，而不仅是输出轨迹等变 |
| [SingularTrajectory, CVPR 2024](https://arxiv.org/abs/2403.18452) | 多轨迹任务统一表征与适配 | 统一轨迹空间、跨域 normalization | 不确定单位和历史支持不足时的拒绝介入机制 |
| [Joint Metrics Matter](https://arxiv.org/abs/2305.06292) | JADE/JFDE、collision 与联合评价 | 仅增加 collision metric 或 joint loss | 基线与神经预测混合时，显式避免组合引起的冲突 |
| [Conformal Risk Control, ICLR 2024](https://arxiv.org/abs/2208.02814) | 有条件下的风险控制 | 调一个“conformal”阈值 | 场景级依赖数据、相对 floor 的 excess risk、easy 子组与联合选择同时控制 |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | 用多重检验校准预测算法 | calibration + 超参搜索 | 在多 agent 与 horizon 相关的场景中构造合适风险和采样单位 |
| [SODA-MPC, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html) | OOD 监测和安全 fallback 控制 | 不可靠就 fallback 的大框架 | 本文研究预测相对误差和联合介入，不把预测误差控制写成控制系统安全保证 |
| [I-JEPA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | 非生成式 latent prediction | JEPA + Transformer 名称组合 | 只有修复 target 学习并证明独立 downstream lift 才保留为主模块 |
| [Generalized HCP, 2026 预印本](https://arxiv.org/abs/2608.15500) | 分组样本与新组的 conformal 推断 | “分组 conformal 是新理论” | 必须针对预测介入风险证明差异，不能只换应用名 |

AgentFormer 官方项目还披露了 normalization bug 并要求使用更正版结果。复现对照时必须锁定版本，不能从旧论文直接抄一个最好数字。

## 5. 推荐的创新主线

工作标题：**When to Trust Neural Motion Forecasts: Scene-Level Risk-Controlled Improvement over Strong Baselines**。

中文：**何时应该相信神经轨迹预测：面向场景联合一致性的基线改进与风险控制**。

这是拟议研究题目，尚未证明 novelty 或性能，不是现成结论。

### 创新候选一：直接学习“相对强基线的介入价值”

对同一条样本、同一组未来标签，计算神经模型与固定基线的误差差值，而不是学习 oracle baseline class：

```text
delta_loss = loss(neural_prediction, future_label)
           - loss(frozen_baseline, future_label)
```

训练同时估计增益、正向伤害和不确定性。标签由 train 内 out-of-fold 预测产生，避免把模型在训练样本上的拟合优势当成未来收益。推理只输入过去历史、候选 rollout 与可用场景信息。

区别于普通 selector 的候选贡献：直接优化 protected deployment 的 excess risk，并分别约束 easy 和困难样本，检验这个目标是否比 best-class、raw-FDE regression、ensemble uncertainty 更有效。

**关键反证实验：** 如果简单 gradient boosting + 相同 delta 标签就与神经 head 一样好，不能把神经表示当贡献；仍可能保留风险决策方法。

### 创新候选二：场景级联合介入，而非每个 agent 独立切换

两个预测器分别生成的整场景预测可能都无冲突，但从一个预测器取 agent A、另一个取 agent B 后，组合可能互相碰撞。逐行 FDE gate 看不到这种组合风险。

对当前可见 agent 集合构造邻接图，联合决定各 agent 使用 floor、neural 或有界混合。目标是降低预计误差，同时控制新增接近事件和群体运动不一致。约束只能由当前状态及候选预测计算，不能读取未来 ground truth。

这是最适合 CVPR 主方法的方向：**研究“选哪些预测可以同时成立”**，而不仅是“哪条预测单独更准”。与 Joint Metrics Matter 比较时，需要证明选择机制的价值超出加一个 joint loss。

**最小决定性实验：** identical predictor，比较 no gate、independent gate、整场景同一 gate、joint selection；报告 ADE/FDE、JADE/JFDE、normalized proximity proxy、easy 与干预率。joint 方法必须在相同干预率或相同风险预算下比较。

### 创新候选三：以独立场景为单位的风险校准与支持不足回退

当前窗口高度相关，单纯增加 bootstrap 次数不会增加独立场景数。将 train、model-selection val、calibration 与最终 test 分开，按原始录像/独立场景统计风险。

在预先固定的有限策略集中，用 scene-level risk 的上界挑选可接受策略；没有足够支持的 source/horizon 回退 floor。数学工具可以沿用 LTT 或有界集中不等式，贡献必须来自合适的风险定义、联合选择结构以及实证，不能声称发明了 conformal inference。

**关键限制：** 少量场景会产生很宽甚至无信息的界。不能把几万重叠 rows 当成几万个独立 calibration scenes，也不能宣称对任意 OOD 分布都保证安全。

### 可选增强：可靠性决定是否使用 scene/goal context

现有 no_scene 胜出提示代理场景可能带来噪声。可以学习 context 的增益/伤害，比较真实 scene image/raster、缺失 context、错配 context 和 history-only。在没有真实场景贡献前，不把论文标题写成 multimodal world model。

这条是可选增强，不能挤占主要方法与对照实验。补标准 JEPA 实现属于修复，也不自动等于创新。

## 6. 投稿前必须补齐的证据

1. **冻结评价协议。** 历史 test 作为已暴露的 development evidence；新的 confirmatory holdout 必须审查是否曾用于 teacher、prototype、scene pack 或模型选择。不能简单重命名旧 test。
2. **按真实录像去重。** OpenTraj 是汇总工具，TrajNet 可包含 ETH/UCY 来源；目录名不同不等于独立外部数据集。
3. **强公开对照。** CV/damped 等因果基线、Stage37、一个轻量神经预测器、AgentFormer 更正版或 EqMotion、一个合适的近期模型，以及 risk-control/uncertainty routing 对照。具体运行版本、许可和 compute 必须可核实。
4. **统一比较口径。** 同样的 observation/prediction rows、坐标、缺失 mask、预测样本数。确定性 K=1 不与 best-of-20 混报；新方法输出同样的未来序列长度。
5. **公认协议与项目协议分列。** raw-frame t50 保留为项目诊断；按原始数据可验证协议重跑公开 benchmark。没有时间/尺度证据时继续用 dataset-local/raw-frame，不能强行换算。
6. **匹配独立性做统计。** 至少 3 seeds；paired scene/source-level bootstrap 至少 2,000 次；报告独立 cluster 数、域间差异、leave-one-scene-out。场景太少时明确不确定性。
7. **证明每个主模块有贡献。** relative-risk target、joint gate、calibration、history、scene/goal 可用性各有同预算消融，负消融保留。
8. **衡量风险而非只报均值。** raw ADE/FDE、相对增益、easy harm、worst-scene harm、tail error、switch coverage 与真实/代理交互指标。
9. **完整论文。** 方法、假设、主要结果、统计、局限、训练细节和匿名复现材料均须成稿，不能拿阶段流水账替代论文。

不需要先实现 true 3D、foundation model、Stage5C 或 SMC 才能投稿 CVPR。需要的是在明确而有限的问题上，有新方法和可信的证据。

## 7. 官方截止日期与内部排期

[CVPR 2027 官网日期](https://cvpr.thecvf.com/Conferences/2027/Dates)及[CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)，核查于 2026-09-16：

| 事项 | 官方日期 AoE | Europe/London 换算 |
| --- | --- | --- |
| Paper registration | 2026-11-10 | 2026-11-11 11:59 GMT，按 AoE 当日末换算 |
| Full paper | 2026-11-16 | 2026-11-17 11:59 GMT，按 AoE 当日末换算 |
| Supplementary | 2026-11-23 | 2026-11-24 11:59 GMT，按 AoE 当日末换算 |

执行时按内部更早日期完成，不依赖时区宽限。[CCF 官方 CVPR 条目](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/al/2017-04-25/592032.shtml)列为 A 类。CCF 第七版目录明确不将 Findings/Workshop 等视作目录中的正式长文，不能用这些结果替代本目标。

| 内部完成时间 | 交付物 | 决定继续的证据 |
| --- | --- | --- |
| 09-20 | 独立数据来源与 teacher lineage 审计、metric contract、paper question | 可明确解释每个训练/校准/测试来源 |
| 09-27 | 修复后的 base predictor、强对照初步复现、方法最小原型 | 有效的同协议比较，无明显泄露 |
| 10-04 | Relative-risk 与 joint selection 的首轮对照 | 至少一个可解释、重复出现的正信号；否则缩减 claim 并修目标 |
| 10-11 | 三个 seeds、第二个真正独立来源、关键消融 | 提升不是单一录像或超参选择造成 |
| 10-18 | 主实验、risk-coverage、tail/worst-scene、context 消融 | 对最接近公开方法有可辨别优势 |
| 10-25 | 独立确认评估结束，冻结主结果和表格 | 主要统计支持明确，负结果保留 |
| 10-31 | 完整英文稿、图表、补充材料初稿 | 能从头至尾解释方法与证据 |
| 11-05 | 合作者/独立阅读反馈，复现抽查 | 关键方法和结果可复核 |
| 11-09 | 注册信息与可提交正文完成 | 所有作者账号、匿名性、页数、引用核对 |
| 11-13 | 最终正文内部锁定 | 给官方 11-16 截止留缓冲 |
| 11-20 | supplementary 内部锁定 | 给官方 11-23 截止留缓冲 |

2026-09-16 至全文名义截止有 61 天。以上是资源与证据驱动的交付计划，不是录用保证。10-04 检查方法可行性，10-25 检查是否足以投稿；不足时如实记录，不能改指标凑结果。

本机仍使用 arm64 `.venv-pytorch`、`num_workers=0`、checkpoint/heartbeat/resume。单次任务不因缓慢而中断；超过既定资源或约 12 小时时报告状态并确定续跑方式。云 GPU 或新增付费资源不自动开通。

## 8. 当前结论

当前最强研究资产是**强基线上的安全改进机制及其失败记录**。当前最适合发展的论文贡献是**相对风险学习 + 场景级联合介入 + 独立场景校准**，而不是更换一个大模型名字。

现阶段不能声称已达到 CCF A/B 投稿质量；本次也没有产生新预测增益。下一步先完成数据独立性与评估契约，再修复模型选择与标签目标，运行可证伪的最小实验。

M3W 仍是 dataset-local/raw-frame 2.5D world-state 研究项目；不是 true 3D、不是 foundation，没有未经验证的 metric/seconds claim。Stage5C 和 SMC 保持禁用。
