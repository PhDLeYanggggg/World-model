# 当前选择区域训练：运行与结果核查

## 本轮改变什么

沿用同一28229参数神经代价网络、428维因果输入、已冻结轨迹候选、
三个种子和每模型2000次更新。第一组18个网络将额外的录像聚合误差损失
固定在旧老师的选择区域；第二组18个网络只把该区域改为当前网络的独立选择，
选择操作停止梯度。没有重新训练轨迹预测器，也没有开启Stage5C或SMC。

这不是普通类别选择器，也不是证明物理安全的新定理。两组网络都保留原来的
逐点损失，固定额外损失权重为1；不根据H结果选择权重、模型或阈值。
触发第二组训练的证据来自C训练区域，不能称为独立验证或风险校准。

## 数据和环境

从项目根目录执行，使用原生arm64环境，不用x86Conda/Rosetta：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_adaptive_record_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_adaptive_record_cost.py train
```

已完成阶段不要重复执行。只有确认原进程已退出、阶段尚未完成且绑定未变化时，
才能使用同一个阶段加`--resume`恢复。不要为重复试验删除checkpoint或完成凭据。
CPU计算线程4、interop1、DataLoader workers0，checkpoint和日志每100步保存。
不因训练慢而缩减目标。每阶段有12小时界限和5GiB磁盘保留。

```sh
cat data/stage_cvpr2027_experiments/moving_adaptive_record_cost_v1/heartbeat.json
tail -n 5 data/stage_cvpr2027_experiments/moving_adaptive_record_cost_v1/train.log
```

heartbeat中的PID、步骤和时间必须结合实际进程判断；单独看到旧日志不代表仍运行。
完整训练必须有18份fit凭据、training_freeze.json和train_execution.json，
并且进程正常退出。模型权重、原始行、特征和求解器缓存都留在忽略的本地目录。

## 先冻结两组，再评估

必须先完成两组各18个网络及完整首模型重训核对，再运行下面顺序。
第一组评估完成并重放通过，第二组评估才会接受执行。评估定义已在新H读取前提交。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_record_excess_cost.py
.venv-pytorch/bin/python scripts/evaluate_m3w_moving_adaptive_record_cost.py evaluate
.venv-pytorch/bin/python scripts/evaluate_m3w_moving_adaptive_record_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_adaptive_record_cost.py
```

未完成的评估可在原进程已退出后使用原命令接续原绑定求解器缓存。
完成后的verify是重新推理及固定决策数值核对，不宣称重新求解MILP得到逐位相同结果。
第二组重放54个神经代价数组；树的代价不重新推理，树决策做hash及数值核对。
不要把cached_verified控制写成fresh训练。

## 读什么结果

- `readout.md`：已知标签效用和直接completion-lower对照区间。
- `per_source.md`：完整23个支持来源，不挑有利场景。
- `gates.md`及`decision.json`：原风险条件与研究结论，执行成功不等于通过科学gate。
- `secondary_metrics.json`：ADE、可用终点FDE、尾部和hard/easy；缺失完整终点的FDE保持不可估。
- `cost_diagnostics.json`：两个代价网络在三种固定选择区域上的误差，属于事后诊断，不能调参。
- `zero_CV_diagnostic.json`：零CV误差行的补充诊断，不替代原风险条件。
- `verification.json`：推理与政策数值重放状态，不是独立泛化证明。

每个arm/policy保留348个重复录像视图及所有空对照。缺支持不能当安全通过，
easy净退化低也不能掩盖正伤害。三个种子和3000次来源组bootstrap不是独立新场景确认。
原24组主要终点仍不可估；现有23组都已经作为开发数据暴露。
原生观察8步预测12步是给定标注顺序，不是已验证的真实传感器时序、米或秒。

## Git和论文

提交代码、配置、完整轻量凭据和正负报告；不提交数据、缓存、checkpoint或第三方文件。
仓库有无关暂存文件时只提交本轮明确路径，不能`git add -A`或清理他人修改。
这些命令依赖已授权的本地数据和私有权重，尚不等于匿名公开冷启动复现已完成。
新报告与旧v9PDF分开，未重新编译并逐页查看就不能宣称PDF已更新。
