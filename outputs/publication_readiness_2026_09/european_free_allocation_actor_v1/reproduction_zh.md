# 分配网络实验复现说明

## 这次实际做了什么

冻结七种预测轨迹、回退基线和18个连续成本网络，只训练36个小型动作分配网络。
每个网络2,000次更新，两个上下文分支具有相同参数量、初始化、输入归一化、抽样和预算。
另做两个完整重复训练，总计72,000次主更新和4,000次重复更新。
训练、完整分配评估、固定动作重评分和公开报告重建是不同的步骤，不能混称。

本次数据为EuropeanSquares斜视摄像头的检测器silver轨迹，不是SDD。
观察8步、预测12步，采样间隔12个原始帧，不声称米、秒、人工gold或true3D。
结果只代表已经暴露的SOURCE拟合，不是独立测试，也不是全流程折外预测。

## 本地运行

工作目录为仓库根目录，使用原生arm64的`.venv-pytorch/bin/python`。
计算线程4、interop1、DataLoader workers0。不要用x86 Conda/Rosetta环境。
入口在导入Torch前检查运行架构。数据、旧权重、上下文索引及其哈希必须与登记一致。

```bash
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py pilot
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py train
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py eval
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py verify
.venv-pytorch/bin/python scripts/report_m3w_free_allocation_actor.py
```

同一登记目录的已完成结果不可覆盖。上列训练命令需要尚未完成的登记副本及合法的本地
数据/模型资产，不能在当前已完成目录重复执行并称为新实验。中断后只恢复未完成阶段：

```bash
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_free_allocation_actor.py eval --resume
```

检查点每100步、训练心跳每50步。分配进度按query保存，互斥锁防止同一实验重复运行。
每阶段12小时硬上限、10GiB磁盘保留、1GiB压缩权重上限。健康但缓慢的任务不降级。
`data/stage_cvpr2027_experiments/european_free_allocation_actor_v1/heartbeat.json`
和对应日志记录PID与阶段；读取实际进度后再决定是否恢复，不能仅凭SSH失败或旧日志重启。

## 结果和核验

`training_freeze.json`绑定36份训练凭据；两个`first_full_refit_*.json`绑定完整重复训练。
`complete.json`绑定72组汇总和完整摘要；`metric_groups.zip`只包含逐组标量统计。
`verification.json`证明重新推理后，对归档动作的评分完全一致，**不代表重新优化MILP**。
实际分配器中断恢复另由18个真实query的工程试验验证，不是完整72组重优化。

公开报告重建只读取代码、登记、公开训练凭据和标量压缩包，不需要读取原始轨迹或权重：

```bash
.venv-pytorch/bin/python scripts/report_m3w_free_allocation_actor.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_free_actor_evidence.py
```

这不等同于匿名包从原始数据开始的全流程复现。完整仓库测试未被上述定向测试替代。
实际门槛以`gates.md`和`readout.md`为准，负结果不能被报告程序或工程测试通过覆盖。

## CREATE边界

本次真实训练和分配试跑的内存、速度支持本地完成，未提交新的CREATE训练任务。
本轮只读SSH检查发生连接重置，不代表旧作业失败。迁移后环境与队列需重新核实，
不能把此前历史作业完成状态当作当前GPU环境验证。以后确需远程训练时仍经调度器，
先核对现有作业、保留job ID、环境、日志、检查点和恢复命令，避免重复提交。

Stage5C和SMC始终关闭；本说明不授权正式投稿或部署。
