# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W 是我长期推进的一个真实世界多智能体世界模型项目。它从一个很朴素的问题开始：只看过去轨迹、局部场景、邻近智能体和强因果基线，一个模型能不能在不偷看未来的情况下，更可靠地预测真实场景里的多智能体运动？

我不想把这个仓库写成产品宣传页。它更像一份持续更新的研究记录：我会把一个想法做出来，和强基线比较，找它在哪些场景失败，再围绕失败点继续修。正结果会保留，负结果也会保留，因为很多路线看起来合理，但一到 easy case、跨域迁移或无泄露评估就会露出问题。

## 现在做到哪一步

目前最可靠的结果不是一个无保护的端到端大模型，而是一套带安全回退的多智能体预测策略。它会先估计强因果基线是否会失败、切换是否可能带来收益、以及切换会不会伤害 easy case；只有验证集证据足够时，才允许学习组件覆盖物理基线。

这个方向已经在 SDD 和外部 top-down pedestrian 数据上给出了一些稳定证据，尤其是 raw-frame `t+50` 和 hard/failure 场景。后续的 latent-state、full-waypoint、group-consistency、neural dynamics 方向也在推进，但我不会把它们写成可部署主模型，除非它们在同一套安全门槛下超过当前 protected policy。

一句话说，M3W 现在是：

```text
protected dataset-local / raw-frame 2.5D multi-agent world-state candidate
```

它还不是 true 3D world model，也不是 large-scale foundation world model。

## 我在做什么

M3W 主要研究 top-down 多智能体运动预测和 agent-scene world modeling。当前系统里比较核心的东西包括：

- 只使用过去信息的轨迹历史特征；
- 局部密度、邻近智能体和相对运动特征；
- train-only 的 goal / route prototype；
- constant velocity、damped velocity 等强因果基线；
- cost-aware / regret-aware 的安全选择器；
- 在 Stage37 这类 safety floor 保护下测试 neural dynamics 和 latent-state 模型；
- 严格的 no-leakage、easy preservation、hard/failure 和 bootstrap/replay 审计。

我更关心一个模型能不能在真实评估里安全地带来增益，而不是它在单个 demo 上看起来多复杂。

## 已经比较明确的结论

几个方向是有用的：

- hard classification 形式的“选哪个 baseline”不稳，cost-aware expected-FDE / gain-harm 选择更可靠；
- conservative fallback 很关键，没有安全 floor 的学习模型很容易伤害 easy case；
- external transfer 必须有 past-only history window、scene-agnostic goal prototypes 和 horizon-specific safety rule；
- raw-frame `t+50` 是当前最有实际证据的外部长时程切片；
- protected full-waypoint / group-consistency 路线有研究价值，但仍要放在安全策略下评估。

也有一些路线目前不能作为主 claim：

- JEPA 表征没有稳定证明 downstream lift；
- latent distribution alignment 缩小距离，不等于预测变好；
- unbounded residual correction 不安全；
- 无保护 Transformer / Hybrid neural dynamics 还没有替代 protected policy；
- raw-frame `t+100` 仍主要是 diagnostic；
- ETH / TrajNet 等外部 held-out 覆盖还需要继续补齐。

## 不能夸大的地方

这个项目里我一直保留这些边界：

- SDD 结果是 pixel-space，除非另有逐源 calibration 证据；
- external top-down 结果是 dataset-local / raw-frame，不能直接写成 metric 或 seconds-level；
- `t+50` / `t+100` 是 raw annotation-frame horizon；
- self-audited 或 inferred scene labels 不是 human gold；
- Stage5C latent generative 没有执行；
- SMC 没有启用。

我希望这个项目最终走向更强的真实世界多模态多智能体 world model，但现在的证据还不能支持 true 3D、foundation 或 metric-world-model 这类说法。

## 结果和代码怎么读

| 路径 | 内容 |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | 长期实验总账、关键结果、失败路线和 claim 边界 |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | 中文长版路线复盘 |
| `outputs/m3w_neural_v1/` | neural candidate、model card 和相关报告 |
| `outputs/stage42_long_research/` | 长期 ablation、source/domain、replay 和安全审计 |
| `outputs/stage43_latent_state/` | protected latent-state / bounded-residual / tail-adapter 证据 |
| [`research_state.json`](research_state.json) | 机器可读的当前研究状态 |

大数据、raw dataset、fast cache、feature store、checkpoint、视频、图像和本地环境都不会提交到 Git。这个仓库只放代码、配置、轻量结果和可审计报告。

## 本地运行

Apple Silicon 上我使用 arm64 PyTorch 环境：

```bash
.venv-pytorch/bin/python
```

训练路径默认使用单进程 DataLoader、checkpoint、heartbeat、resume，以及 CPU/MPS-safe runtime。

基本测试：

```bash
.venv-pytorch/bin/python -m pytest tests
```

## 下一步

下一步我会继续把 neural world dynamics 做实，而不是只依赖 selector-level policy。重点是修弱 source / weak horizon，补 source-level geometry 和合法可用数据，做更严格的 ablation，并证明 scene、goal、interaction、latent-state 至少有一部分能在安全门槛下贡献真实增益。

如果做不到，我会把失败原因写清楚，而不是把它包装成成功。
