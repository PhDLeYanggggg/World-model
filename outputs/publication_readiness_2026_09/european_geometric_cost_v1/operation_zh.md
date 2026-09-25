# 因果几何上界收益／风险头操作说明

## 实验目的

本轮不更换轨迹预测器，而是重训收益／风险头。由两条过去信息生成的预测轨迹，
计算各请求步位置差的最大值D，用它约束预测收益和伤害。不能用未来有效标签
掩码计算D；未来标签可能不完整，简单平均轨迹差不一定是合法上界。

数学上界不等于校准保证。风险估计依旧可能错，2%门限依旧可能在另一场景失效。
全部old、utility_only、risk_only、both组合都报告，不根据这些评价结果选赢家。

## 运行与恢复

使用已有原生arm64环境，CPU线程4，inter-op1，workers0。不要用x86_64 Conda。

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_european_score_scaling.py
.venv-pytorch/bin/python scripts/run_m3w_european_geometric_cost.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_geometric_cost.py --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_geometric_cost.py --train --resume --replay --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_geometric_cost.py --verify
.venv-pytorch/bin/python scripts/report_m3w_european_geometric_cost.py
.venv-pytorch/bin/python scripts/plot_m3w_european_geometric_cost.py
.venv-pytorch/bin/python scripts/complete_m3w_european_geometric_cost.py
```

100次更新的试跑写入检查点，完整训练从100继续到2000，不重新获得额外训练预算。
每200次更新原子保存模型、优化器、随机数状态和采样次数。中断后先查进程是否
仍在运行；确认停止再用`--train --resume --replay --evaluate`继续。完整且校验
通过的头不会重复训练。不要删除已有缓存或在活跃进程上重复启动。

本地检查点、分数缓存和日志在：
`data/stage_cvpr2027_experiments/european_geometric_cost_v1/`。
查看`heartbeat.json`里的PID、状态、head和step；再核实该PID及日志更新。
单独存在心跳文件不证明进程还活着。所有日志保留在`events.jsonl`。
原始数据和上游检查点不是仓库的一部分，仅克隆代码不能完成真实实验复现。

## 核查与解读

54个头各2000次更新，共108000次更新。与原头逐一核对相同输入、标签、标准化、
参数数量和采样次数。54个检查点分别重放前4096个held-index行，不是随机抽样
或所有行重推理。之后全部144个策略视图和指标重算，旧36个视图必须一致。

`training_losses.csv`和训练图记录loss。loss是对应批次、对应任务下按训练CV误差
缩放的MSE，不可将不同任务的loss直接比较，也不能把loss下降当作held-out提升。
观察ADE/FDE、最差场景easy、零CV伤害、介入率、oracle机会及被拒绝的收益。
保留所有负结果。区间按场景重采样，不把重叠窗口当独立样本。

本轮使用本地小模型训练。CREATE只复用此前已授权只读队列观察记录，不表示本轮
重新查过队列，更不表示已找到远程M3W资产。没有提交、修改或终止远程任务。

仍是检测轨迹、图像像素、观察8步／预测12步、raw stride12开发实验。不是t50、
秒级、metric、人工gold、物理安全、true3D、foundation或独立确认。
不执行Stage5C，不启用SMC，不根据本轮开发结果升级部署。
