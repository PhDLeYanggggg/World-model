# 本轮实验操作与结果说明

## 本轮实际完成了什么

这次没有重新训练轨迹 Transformer，也没有训练新的 JEPA。复用并核验了此前
18 个冻结预测器，重新训练 9 个岭回归风险头和 9 个真实 PyTorch 风险头，
只修复一个问题：回退参照与 gain/harm 监督都改为因果恒速 CV。

神经风险头共完成 18,000 次更新，三个种子、三个嵌套源场景折全部跑完。
第一次 100 次更新的试跑保存在 checkpoint 中，后续从它继续，不额外增加预算。
训练、对照评估、完整指标复算和 checkpoint 推理复核都已结束，不是在后台
继续运行。缓存复用和本轮新训练已在英文报告中分别标记。

## 结果应如何理解

- 相对 CV 的 ADE 改善为 4.18% 到 4.43%，三个条件性场景 bootstrap 区间均为正。
- easy 平均退化从上一版约 12.81% 到 13.38%，降到 1.90% 到 2.38%。
- 但最差场景的 easy 仍退化 8.71% 到 12.54%，每个种子仍伤害 1 个零误差样本。
- 相对强固定阻尼基线，改善仅 0.09% 到 0.36%，区间都跨过零，不能宣称稳定超过。
- 场景联合决策未证明比相同介入数量的独立决策更好。缺乏支持的比较保留为
  undefined，不能当成零或删除对应场景。

因此结论是“参照修复有部分作用，但仍不可部署”，不是“已完成世界模型”。
保持原有部署状态；不据此提升任何新模型。旧 Stage37 分数在后续来源审计中
发现重复、teacher 暴露和 test 选型问题，仍属于探索性历史结果，不能拿旧数值
充当新的独立验证。

四个 CV 零误差样本都在同一源地点，每个只有 12 步中的 2 步标签，缺最终端点。
所以“零误差”只描述已有标签，不代表整个未来都正确。它们所在评估折的训练
部分没有这种事件，联合决策小样本也没有它们。不能删掉这些样本，不能将缺少
事件写成安全已验证，也不能使用未来标签做推理过滤。

## 本地复核

工作目录：`/Users/yangyue/Downloads/World`。
使用原生 arm64 的 `.venv-pytorch/bin/python`，不要换成 x86_64 Conda。
当前配置计算线程 4、inter-op 线程 1、DataLoader workers 0。

下面的操作核验现有实验，不重新训练轨迹模型：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cv_reference.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_cv_reference.py
.venv-pytorch/bin/python scripts/plot_m3w_european_cv_reference.py
```

第一步会验证冻结身份、复算完整指标，并对每个风险头重新推理 4,096 个抽样行。
已有结果为 18 个头全部一致、最大差异 0。这是抽样 checkpoint 推理复核，
不是所有轨迹重新训练一遍，也不是独立数据上的确认。

首次执行该冻结版本时使用：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cv_reference.py --prepare --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cv_reference.py --train --resume --evaluate
```

现有完整结果不会因这两条命令被算作“又训练成功一次”。如运行中断，第二条
命令从已保存的模型、优化器和随机采样状态恢复；身份不匹配时应停止检查，
不要覆盖旧 checkpoint。改变科学目标、数据、损失或阈值必须建立新版本，
不能修改本轮冻结文件后继续沿用原结果。

记录位置：

- 公共证据：`outputs/publication_readiness_2026_09/european_cv_reference_v1/`。
- 私有日志与 checkpoint：`data/stage_cvpr2027_experiments/european_cv_reference_v1/`。
- 最新进程状态：私有目录的 `heartbeat.json`；事件记录为 `events.jsonl`。
- `results.md` 是全部对照；`training_losses.md` 是实际训练 loss。
- `analysis.json` 含场景分解、标签支持、尾部误差和成对区间。
- `accounting_audit.json` 保存额外逐项核对和零事件支持结论。

本轮规模适合本地，未提交 CREATE 作业，也未修改 simulation 项目的任务或环境。
需要更大实验时仍先核对准确的远程项目路径与调度限制，不扫描无关目录。

## 下一步的明确依据

先测试针对 easy/零误差事件的条件风险目标和基于过去信息的支持不足回退。
保持同一源数据、同一冻结预测器、原有 2% easy 限制和零额外伤害限制，
让一次实验能区分“风险目标有问题”还是“预测器本身没有可用增益”。
独立 selection、calibration、confirmation 和 DroneCrowd 保持关闭。

这些仍是 detector tracks 的图像像素与 raw frame 实验；不是米、秒、true 3D、
foundation model、人工 gold 或物理安全证明。Stage5C 和 SMC 均未执行。
