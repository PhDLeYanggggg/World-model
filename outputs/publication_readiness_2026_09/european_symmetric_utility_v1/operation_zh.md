# 对称收益头实验：运行和核查

## 本轮改变了什么

只把收益头的 underharm4 损失改为对称 MSE。神经轨迹候选与固定阻尼候选
都重训收益头，不能只给神经模型更有利的目标。72 个风险头、轨迹模型、
特征、标签、抽样、划分、2% 预测风险预算和支持不足时的回退保持不变。
18 个小型神经收益头不等于重训 18 个端到端世界模型。

## 环境

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 的
`.venv-pytorch/bin/python`。入口在导入 Torch 前拒绝 macOS x86_64/Rosetta。
CPU 计算线程 4，inter-op 1，数据加载进程数 0；不探测 GPU 资源，不用
默认 Intel Conda，不把 NumPy fallback 当作 Torch 训练。

本轮只训练小型收益头，本地资源足够，不提交 CREATE 作业，也不修改
simulation model 的环境或运行中作业。远程 M3W 目录未核实，不猜路径。

## 首次运行

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_utility.py --prepare --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_utility.py --train --resume --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_symmetric_utility.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_symmetric_utility.py
.venv-pytorch/bin/python scripts/plot_m3w_european_symmetric_utility.py
```

需要前序已核验的本地数据和模型；GitHub 不包含第三方数据、逐行特征或权重。
文件缺失时报告前置条件缺失，不造数据、不绕过哈希检查。

## 中断恢复

私有目录：`data/stage_cvpr2027_experiments/european_symmetric_utility_v1/`。
`heartbeat.json` 和 `events.jsonl` 记录 PID、UTC 时间、阶段、模型名和训练步数。
每 200 次更新原子保存模型、优化器、随机数状态与抽样记录。
首个 100 步试跑从断点继续，计入每个头固定的 2,000 步，不能追加成 2,100 步。

确认原进程已退出后，使用上面的 `--train --resume --evaluate`。
已完成模型先校验后跳过，不重复训练；单实例锁阻止重复进程。
不要在运行时改动配置、已绑定代码、注册协议或输入文件。
只因慢而终止或换成小样本不属于本实验协议。

## 结果如何判断

- `analysis.json` 是较大的本地完整指标，Git 忽略；公开轻量摘要绑定其哈希。
- `verification.json` 确认完整指标由相同产物重算一致。
- `checkpoint_replay.json` 对每个新头复算 4,096 行并与保存预测精确比较；
  同时验证新旧头的抽样、已知标签支持、初始化常量和预处理一致。
- `accounting_audit.json` 独立复算全部逐点决策，并核查原风险头、联合决策
  预算、实际干预数量和支持不足时的回退。
- `results.md` 保留全部 48 个配置及联合对照；不能只挑最好种子作结论。
- `training_losses.md` 是训练小批次损失，不是测试准确度。

指标属于已打开的 12 个源场景开发实验。3,000 次场景 bootstrap 是以这些
数据和模型为条件的区间，不是独立风险校准。保留组没有打开。
观察 8 步、预测 12 步、原始帧间隔 12、图像像素；不能换称 t+50、秒或米。
检测器轨迹不是 human gold。没有部署晋级，没有 Stage5C 或 SMC 执行。
