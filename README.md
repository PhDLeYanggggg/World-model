# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W 是我长期做的一个真实世界多智能体世界模型项目。它从一个简单但很难的问题开始：如果只看过去轨迹、局部场景、邻近行人和一些强因果基线，模型能不能在不偷看未来的前提下，更可靠地理解真实场景里的多智能体运动？

我把这个仓库当成一份公开研究记录，而不是产品宣传页。这里会保留正结果，也会保留失败路线，因为这个方向里很多方法在单个 demo 上看起来很漂亮，但一放到 easy case、跨数据集迁移、长 horizon 或严格无泄露评估里，就会暴露问题。

## 项目现在是什么状态

目前最稳的部分不是一个无保护的端到端大模型，而是一套带安全回退的多智能体预测策略。它会先判断强因果基线是否可能失败、学习模型是否真的有收益、以及切换会不会伤害原本很容易的样本。只有这些条件在验证集上成立时，系统才允许从物理基线切到学习策略。

这条路线已经在 SDD 和外部 top-down pedestrian 数据上给出了一些比较稳定的结果，尤其是 raw-frame `t+50` 和 hard / failure 场景。后面的 latent-state、full-waypoint、group-consistency 和 neural dynamics 还在继续推进，但我不会把它们写成主模型，除非它们在同一套安全标准下超过当前 protected policy。

当前我会把 M3W 描述为：

> 一个受安全回退保护的、dataset-local / raw-frame、2.5D 多智能体 world-state candidate。

它还不是 true 3D world model，也不是 large-scale foundation world model。

## 我主要在研究什么

M3W 的核心不是“换一个更大的模型”，而是让真实评估中的每一次切换都有理由。现在系统里比较重要的组成包括：

- past-only 轨迹历史、速度、加速度、曲率和 stop/go 特征；
- 邻近智能体、局部密度、TTC 和相对运动特征；
- 只从训练 split 构建的 goal / route prototypes；
- constant velocity、damped velocity 等强因果基线；
- cost-aware / regret-aware 的安全选择器；
- 在 safety floor 保护下评估 neural dynamics、latent-state 和 full-waypoint 模型；
- no-leakage、easy preservation、hard/failure、bootstrap 和 replay 审计。

我现在更关心模型是否能在真实切片里稳定增益，而不是架构名字听起来有多复杂。

## 目前比较清楚的结论

有几条路线已经证明比较有价值：

- 直接 hard-classification 地预测“哪个 baseline 最好”不稳，expected-FDE / gain-harm / regret-aware 的策略更可靠；
- conservative fallback 非常关键，没有 safety floor 的学习模型很容易伤害 easy case；
- 外部迁移必须使用 past-only history window、scene-agnostic goal prototypes 和 horizon-specific safety rule；
- raw-frame `t+50` 是目前外部数据里证据最扎实的长 horizon 切片；
- protected full-waypoint 和 group-consistency 有研究价值，但必须在安全策略下评估。

也有一些路线目前不能作为主 claim：

- JEPA 表征没有稳定证明 downstream lift；
- latent distribution alignment 让分布看起来更接近，不等于预测真的变好；
- unbounded residual correction 不安全；
- 无保护 Transformer / Hybrid neural dynamics 还没有替代 protected policy；
- raw-frame `t+100` 仍主要是 diagnostic；
- ETH / TrajNet 等外部 held-out 覆盖还需要继续补齐。

## 我不会夸大的边界

这个项目的边界我会一直写清楚：

- SDD 结果是 pixel-space；
- external top-down 结果是 dataset-local / raw-frame；
- `t+50` 和 `t+100` 是 raw annotation-frame horizon，不是秒级指标；
- self-audited 或 inferred scene labels 不是 human gold；
- Stage5C latent generative 没有执行；
- SMC 没有启用。

除非未来完成逐源 calibration、homography / scale 审计和更大规模跨数据集训练，否则我不会把当前结果写成 metric prediction、true 3D 或 foundation world model。

## 怎么读这个仓库

| 路径 | 内容 |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | 长期实验总账、关键结果、失败路线和 claim 边界 |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | 中文长版路线复盘 |
| `outputs/m3w_neural_v1/` | neural candidate、model card 和相关报告 |
| `outputs/stage42_long_research/` | source/domain、full-waypoint、safety 和 replay 细账 |
| `outputs/stage43_latent_state/` | protected latent-state、tail adapter 和 bounded-residual 证据 |
| [`research_state.json`](research_state.json) | 机器可读研究状态 |

大数据、raw dataset、fast cache、feature store、checkpoint、视频、图像和本地环境都不会提交到 Git。这个仓库只放代码、配置、轻量结果和可审计报告。

## 本地运行

Apple Silicon 上我使用 arm64 PyTorch 环境：

```bash
.venv-pytorch/bin/python
```

训练路径默认使用单进程 DataLoader、checkpoint、heartbeat、resume，以及 CPU / MPS safe runtime。

基本测试：

```bash
.venv-pytorch/bin/python -m pytest tests
```

## 下一步

接下来我会继续把 neural world dynamics 做实，而不是只依赖 selector-level policy。短期重点是补弱 source / weak horizon，扩充合法可用的外部数据支持，做更严格的 ablation，并验证 scene、goal、interaction、latent-state 是否能在安全门槛下带来真实增益。

如果做不到，我会把失败原因写清楚，而不是把它包装成成功。
