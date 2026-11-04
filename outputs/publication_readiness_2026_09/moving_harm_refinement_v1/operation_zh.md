# 伤害代价修正实验操作说明

## 范围

本实验固定已有的18个神经代价头和预测轨迹，另外拟合54个小型树修正器。
不是54个新神经网络，不是轨迹残差训练，不是独立风险校准。
只使用原有64来源录像范围；23个有支持的来源组仍是已暴露开发评价。
原24组主要结果不可估，未恢复为独立测试。Stage5C、SMC保持关闭。

## 运行顺序

在项目根目录使用原生arm64环境。实验定义已登记并提交，不改动旧版本。

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_harm_refinement.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_harm_refinement.py tests/test_m3w_harm_refinement_reporting.py
```

这些是带不可变回执的研究命令，不是每次都覆盖重跑的演示。已完成阶段会拒绝
重新执行。训练中断时用同一阶段加`--resume`，不能更换配置、权重或数据哈希。
恢复会从保存的树序列重建拟合预测，并检查哈希；不是仅从最后一个计数器继续。
每一轮保存检查点并更新心跳。CPU计算4线程、interop1、DataLoader workers0。
树拟合为单进程；没有调用MPS/CUDA或声称这些后端已验证。

## 监控

```bash
tail -n 5 data/stage_cvpr2027_experiments/moving_harm_refinement_v1/training.log
cat data/stage_cvpr2027_experiments/moving_harm_refinement_v1/heartbeat.json
```

心跳包含实际PID、时间、阶段、轮数和产物大小。超过12小时的阶段通过原定义
恢复；健康慢任务不改成quick结果。磁盘保留9.5GiB，私有产物上限100MiB。
首次继承10GiB保护的启动在登记前退出，没有训练；资源设置的前瞻性修订已写入
protocol，不改变2%科学要求。磁盘继续不足时保留现有结果，不清理无关数据。

## 结果解释

主要对比是动态区域减固定区域的joint-free直接配对completion-lower差，
不是从三种方法中挑一个最好看的测试结果。全部六种策略、来源组、重复录像视图、
缺失标签和空同切换数对照都必须保留。空对照是不可估，不代表测得零风险。
记录级完整2%要求不变；汇总正收益或更少切换不能代替它。
保存的求解记录可以核验绑定和可行性，不声称限时MILP逐比特重新求解一致。
图中训练损失使用各方法自己的权重，不是相同评价分布上的性能对照。

坐标仍是数据集内pixel-space；native8步观察/12步预测不等于已验证视频帧或秒。
没有新的米、秒、true3D、foundation或可部署声明。
