# 对称风险头实验运行说明

## 实验边界

本轮只将神经风险头的 underharm4 损失改为对称 MSE。它估计事件中的
基线误差量和正伤害量，不是概率，也不是已经校准的安全上界。
收益头固定为上一轮对称收益头，轨迹、标签、特征、抽样和 2% 预测风险预算
全部不变。神经与阻尼两种候选都重训；线性风险对照直接核验复用。

36 个小风险头，每个 2,000 次更新，总计 72,000 次。不是重训端到端世界模型。
试跑 100 次更新后恢复，计入首个模型预算。全部训练完成才做对照读数。

## 环境与命令

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 `.venv-pytorch`。
入口在导入 Torch 前拒绝 macOS x86_64；计算线程4，inter-op1，数据进程0。
不使用默认 Intel Conda，不探测 GPU，不用 NumPy fallback 冒充训练。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_risk.py --prepare --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_risk.py --train --resume --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_risk.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_symmetric_risk.py
.venv-pytorch/bin/python scripts/plot_m3w_european_symmetric_risk.py
```

首次运行需要前序本地数据与已核验模型。GitHub 不含第三方原始轨迹、逐行
缓存和权重。缺少这些文件就报告缺失，不造数据、不忽略哈希。

## 恢复与核验

私有目录为 `data/stage_cvpr2027_experiments/european_symmetric_risk_v1/`。
心跳和事件日志记录 PID、UTC、模型名与步数。每200步原子保存模型、优化器、
随机数和抽样状态。中断后确认原进程结束，再使用 `--train --resume --evaluate`；
已完成模型核验后跳过。不要删除锁文件来绕过活跃进程，也不要重写已冻结配置。

`verification.json` 是完整指标重算证据；`checkpoint_replay.json` 是每个新头
4,096 行的精确预测复算，以及新旧抽样和预处理一致性证据。
`accounting_audit.json` 独立复算逐点决策并核查联合预算和匹配干预数量。
`risk_reliability.json` 及其表格明确区分总体与被选中样本的预测/实际风险。
没有事件支持时比值为 undefined，不能当作零风险。训练 loss 不等于测试准确度。

本轮小风险头可在本地完成，不提交 CREATE 作业。远程认证最近可核实记录是
2026-09-24 的只读连接失败；它不能说明远程任务是否停止。本轮不修改凭据、
simulation 环境或作业，也不把历史调度器截图当当前状态。

## 如何理解结果

2% 预测约束未变，不等于实际伤害满足2%。如果新损失提高精度却伤害 easy 或
零误差基线样本，仍然不能升级部署。保留全部48配置，不根据结果挑阈值或种子。
只使用已打开的12个源场景，独立选择/校准/确认组保持关闭。
3,000次场景 bootstrap 是开发性条件区间，不是独立确认。
图像像素、观察8步、预测12步、原始帧间隔12；不是秒、米、t50、human gold、
true3D或foundation。Stage5C和SMC不执行。
