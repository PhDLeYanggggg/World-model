# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W 是我正在做的真实世界多智能体世界模型项目。它从一个很朴素的问题开始：

> 在真实 top-down 场景里，如果只看过去的轨迹、邻近行人、局部场景线索和强因果基线，模型能不能在不偷看未来的情况下，更稳地预测多智能体接下来会怎么走？

我不想把这个项目做成一个只展示漂亮样例的 demo。这个仓库记录的是一条比较硬的研究路线：强基线、无泄露评估、安全回退、失败复盘、外部数据迁移，以及每一次“看起来有效”的东西到底能不能经得住更严格的切片检查。

## 项目现在是什么

目前 M3W 最可靠的部分，是一套受安全回退保护的多智能体预测策略。它不会直接相信学习模型，而是先判断：

- 当前强因果基线会不会失败；
- 学习策略是否有足够的预期收益；
- 切换以后会不会伤害原本很容易的样本；
- 这个判断是否在验证集上成立，而不是在 test 上临时调出来。

只有这些条件满足时，系统才允许从物理基线切到学习策略。否则就回退到最稳的 causal baseline。

用更准确的话说，当前的 M3W 是：

> 一个 protected、dataset-local / raw-frame、2.5D 多智能体 world-state candidate。

它还不是 true 3D world model，也不是 large-scale foundation world model。

## 为什么这条路线有意义

在多智能体轨迹预测里，很多学习模型可以在平均指标上看起来不错，但真实部署时会有两个问题：

1. 容易在 easy case 上乱切，反而比简单物理基线更差；
2. 容易在一个数据源上有效，换到另一个场景或坐标系统就失效。

所以 M3W 的重点不是“换一个更大的网络”，而是让模型学会什么时候该相信自己、什么时候该退回强基线。这个方向目前比无保护的 residual、硬分类 selector、单纯 latent alignment 或直接端到端 neural dynamics 更可靠。

## 当前比较稳的发现

这些结论是目前比较清楚、也比较能复现的：

- hard-classification 地预测“哪个 baseline 最好”不稳，cost-aware / regret-aware 的 expected-FDE 选择更可靠；
- conservative fallback 是关键，没有 safety floor 的学习模型很容易伤害 easy case；
- 外部迁移需要 past-only history window、scene-agnostic goal prototypes 和 horizon-specific safety rule；
- raw-frame `t+50` 是目前证据最扎实的长 horizon 切片；
- full-waypoint、group-consistency、latent-state 和 neural dynamics 有研究价值，但必须在同一套安全门槛下超过当前 protected policy，才会被写成主贡献。

## 还不能夸大的地方

这些边界我会一直保留在公开说明里：

- SDD 结果是 pixel-space，不是 metric prediction；
- external top-down 结果是 dataset-local / raw-frame，不是统一物理尺度；
- `t+50` 和 `t+100` 是 raw annotation-frame horizon，不能写成秒级预测；
- self-audited 或 inferred scene labels 不是 human gold；
- Stage5C latent generative 没有执行；
- SMC 没有启用；
- 目前的结果不能写成 true 3D，也不能写成 foundation world model。

## 仓库怎么读

| 路径 | 内容 |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | 实验总账：主要结果、失败路线、claim 边界和长期状态 |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | 中文长版复盘：做过什么、哪些失败、哪些成功 |
| `outputs/m3w_neural_v1/` | neural candidate、model card 和相关报告 |
| `outputs/stage42_long_research/` | source/domain、full-waypoint、safety、replay 和 paper-claim 细账 |
| `outputs/stage43_latent_state/` | protected latent-state、tail adapter、bounded residual 和 reviewer replay 证据 |
| [`research_state.json`](research_state.json) | 机器可读研究状态 |

大数据、raw dataset、fast cache、feature store、checkpoint、视频、图像和本地环境不会提交到 GitHub。仓库里只保留代码、配置、轻量结果和可审计报告。

## 本地运行

Apple Silicon 上我使用 arm64 PyTorch 环境：

```bash
.venv-pytorch/bin/python
```

基本测试：

```bash
.venv-pytorch/bin/python -m pytest tests
```

训练和评估脚本默认走单进程 DataLoader，并支持 checkpoint、heartbeat、resume，以及 CPU / MPS safe runtime。

## 下一步

短期我会继续做三件事：

1. 补弱 source 和弱 horizon，尤其是 raw-frame `t+100` 的 source-stable 证据；
2. 把 scene、goal、interaction、latent-state 的贡献从 proxy-heavy evidence 推到更严格的 retrained ablation；
3. 继续训练受安全回退保护的 neural dynamics，但只有它真正超过当前 protected policy，才会进入主模型叙述。

如果做不到，我会把失败原因写清楚，而不是把它包装成成功。
