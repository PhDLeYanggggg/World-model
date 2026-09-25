# 因果拒绝介入实验操作记录

本轮只运行固定预测器上的新支持检查与决策评估，没有重新训练神经网络。
上轮的网络、分数和轨迹预测是校验后复用；本轮支持范围、停止状态保护、同帧对照和指标是新计算。

## 本轮比较

保留原预测和回退基线，比较三种拒绝介入方式：最近一步停止、训练来源支持不足、二者组合。
每种方式都加入“同一录像同一当前帧、介入人数完全相同”的风险排序和固定随机排序对照。
这样可以区分“选得更准”与“只是少介入所以少犯错”。

两个目标版本、两个归一化版本、三个来源折、三个训练种子、两个风险事件，全部保留。
结果来自已经打开的开发场景，不是独立确认结果；不能把 720 个相关视图当作 720 个独立实验。

## 环境和恢复

使用项目的原生 arm64 `.venv-pytorch/bin/python`。四个计算线程、一个 inter-op 线程，
不启动 DataLoader 多进程。本轮不依赖 CUDA，不提交 CREATE 训练作业。
CREATE 只读队列核查的原始信息保留在私有目录，不上传账户和调度详情。

私有目录为 `data/stage_cvpr2027_experiments/european_causal_abstention_v1/`。
`heartbeat.json` 与 `events.jsonl` 记录 PID、阶段、时间和完成组。
`batch/decisions`、`fitting/decisions` 保存不可变决策；两者全部完成后才可读新结果。
`evaluation` 按组保存结果，只有完整组才能进入汇总；中断后使用 `--resume` 继续。
不通过重命名或覆盖结果重复挑选最佳阈值。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode batch --phase evaluate --resume
.venv-pytorch/bin/python scripts/run_m3w_european_causal_abstention.py --mode fitting --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_causal_abstention.py
```

这些命令需要本机已校验的数据和分数缓存；Git 上只有代码、配置和轻量聚合结果。
独立模型选择、风险校准和最终确认数据仍未打开。Stage5C 和 SMC 均不执行。
坐标仍是像素；观察 8 步、预测 12 步、原始标注步长 12，不是秒或米。
检测器轨迹不等于人工金标准，不能称 true 3D、foundation 或物理安全保证。
