# 固定发生概率实验：运行与恢复

## 本轮检验什么

两组都从同一个已核验的模型开始，都拆分为发生概率分支和代价分支。
对照组继续训练两个分支；实验组固定发生概率，只训练代价分支。
两组的初始化、样本顺序、优化器设置、2,000 次更新预算完全一致。
因此，本实验检验的是“固定概率分支”的作用，不是拆分网络本身的作用。

108 组配对、216 个模型、432,000 次更新保留不变。最终使用规定的最后
一次更新，不按开发集结果挑模型。这里只使用已经开放的 12 个开发来源；
独立模型选择、风险校准和最终确认来源保持关闭。

## 环境与进度

工作目录为 `/Users/yangyue/Downloads/World`，使用原生 arm64 环境。
CPU 计算线程为 4，跨算子线程为 1，DataLoader workers 为 0。
不要使用默认 x86_64 Conda，也不要并行启动同一个实验的第二份训练。

心跳和事件位于：

```text
data/stage_cvpr2027_experiments/european_fixed_occurrence_v1/heartbeat.json
data/stage_cvpr2027_experiments/european_fixed_occurrence_v1/events.jsonl
```

每个模型每 500 步保存检查点。先检查心跳、PID 和已有完成记录，再决定
是否恢复。进程仍正常推进时不重启。磁盘低于 10 GiB 保护线时保留全部
已有产物，查清资源状况后继续；不减实验规模，也不删除无关文件。

## 固定执行顺序

所有命令都在上述工作目录执行。`register` 和 `pilot` 已完成，不重复运行。
训练尚未完成且原进程已退出时才运行恢复命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/train_m3w_fixed_occurrence.py train --resume
```

完整训练完成后，重训第一对模型检查确定性，再核验全部训练记录：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/train_m3w_fixed_occurrence.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_fixed_occurrence.py training
```

先提交 `training_freeze.json` 和 `fit_replay.json`，再计算动作：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_fixed_occurrence_policy.py decide --resume
```

全部动作冻结后，先提交 `decision_freeze.json`，再做完整动作重放及评价：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_fixed_occurrence_policy.py replay_decide
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_fixed_occurrence_policy.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_fixed_occurrence_policy.py replay_evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_fixed_occurrence.py readout
```

已经存在的最终记录不会被静默覆盖。评价重放是核验同一结果，不是新一轮
模型选择。训练只重训第一对检查确定性；动作和数值评价重放覆盖全部组。

## 如何解释结果

- 训练误差下降不是泛化、校准或安全证据。
- 主开发比较为固定分支与可训练分支，逐个查询匹配介入数。
- 原始控制、独立选择、联合选择和回退基线都保留。
- 未定义风险、超标来源和不利比较不能删除。
- 3,000 次 locality bootstrap 先合并重复视图，不把窗口当独立样本。
- 12 个来源已用于开发；其置信区间不是独立确认或选择校正区间。
- 即使开发筛查通过，也不自动改变部署或打开独立确认来源。

观察 8 步、预测 12 步，步距为 12 个原始帧；坐标仍为图像局部坐标，
标签为自动检测 silver。不能声称米、秒、人工 gold、真实物理安全、
true 3D、foundation model 或已经达到投稿质量。Stage5C 和 SMC 保持禁用。

Git 仅保存源码、配置、报告和轻量指标。检查点、原始数据、缓存及大数组
留在本地忽略目录；提交时显式指定本实验路径，不夹带其他已暂存工作。
