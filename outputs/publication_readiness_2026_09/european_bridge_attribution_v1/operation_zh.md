# 对照实验复现说明

## 环境与数据

在 World 项目根目录使用本机 arm64 `.venv-pytorch/bin/python`。
不要使用 x86_64 Conda/Rosetta。入口在导入 Torch 前检查架构；计算线程4，
interop线程1，DataLoader workers0。源数据、父实验模型和私有缓存必须仍在
原位置且通过哈希检查。代码和轻量报告在 GitHub，原始数据、模型检查点及
缓存不在仓库中，因此只有 Git clone 不足以直接复现。

本实验目录为 `data/stage_cvpr2027_experiments/european_bridge_attribution_v1/`。
`events.jsonl` 保存进度，`heartbeat.json` 保存 PID 和最近完成步骤；
`heads/` 保存检查点和成本分数。低于10GiB磁盘余量会停止并保留检查点。

## 执行顺序

下面的命令用于原始注册执行。注册和决策冻结文件均需先提交并同步远端，
程序才允许进入依赖阶段。这不是读完结果后再补注册。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_attribution.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_attribution.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_attribution.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_attribution.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_attribution.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_bridge_attribution.py
.venv-pytorch/bin/python scripts/verify_m3w_european_bridge_attribution.py
```

`pilot` 是首个头的100步，不是额外模型；`train --resume` 接着训练到2000步。
首次完整评价结束后用 `verify` 重放与核查，不覆盖不可变评价文件，也不要
删除已有结果以重新挑选种子。若要从头重训，需要新的实验命名空间、注册
和结果来源标记，而不是改写已完成的记录。

## 结果核查

核对36个新神经头、36个新 ridge 对照、72,000总更新，以及36组配对决策文件
和396个策略视图。完整配对的36个旧神经头与36个 ridge 对照为
`cached_verified`，不是本轮重新训练。核验在独立进程恢复72个新旧神经头，
逐项检查训练场景排除、分数、介入决定、每查询匹配数量和汇总指标。

损失记录在 `training_metrics.json`：它使用训练源CV尺度，不是原始FDE。
随机小批次损失不要求单调。测试通过、训练完成、平均误差下降和得到独立
统计证明是不同层级，不能混写。

本轮 CREATE 仅进行了已授权只读队列检查，未提交新任务或修改远端。
本地真实训练试跑支持在CPU4完成成本头预算；不干扰现有其他研究任务。
正式确认集仍封存，所有结果均属于已打开模型选择场景上的开发证据。
