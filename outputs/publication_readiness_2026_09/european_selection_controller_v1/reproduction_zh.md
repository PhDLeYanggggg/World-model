# SOURCE 控制器比较的复现边界

本实验训练的是真实 Torch 代价控制器，不是新的轨迹预测器。轨迹由已冻结的9个
history 预测器给出；18个新控制器分别使用逐行或同帧上下文，均只在对应预测器未见过的
8个 SOURCE 场景训练。模型选择集标签不能训练权重或归一化，也不用于调阈值。

## 当前本地资产上的执行入口

使用原生 arm64 `.venv-pytorch/bin/python`，CPU4线程，单进程。不要使用默认x86 Conda。
执行前检查现有进程和完成回执，不重复启动已经运行或已经完成的阶段。

```bash
.venv-pytorch/bin/python scripts/run_m3w_selection_controller.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selection_controller.py train --resume
mkdir -p data/stage_cvpr2027_experiments/european_selection_controller_v1/allocation_progress
.venv-pytorch/bin/python scripts/run_m3w_selection_controller.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_controller.py eval --resume
.venv-pytorch/bin/python scripts/run_m3w_selection_controller.py verify
.venv-pytorch/bin/python scripts/export_m3w_selection_controller_traces.py
.venv-pytorch/bin/python scripts/report_m3w_selection_controller.py
```

这些是流程入口，不意味着应该覆盖已完成的不可变阶段。完整回执存在时脚本会拒绝重写。
只有中断的同定义阶段才能`--resume`，不得删除回执来伪造一轮fresh结果。
checkpoint保存优化器、采样器排列/计数、RNG和绑定哈希。日志和heartbeat在私有实验目录。
每阶段最多12小时，超过时保留原预算续跑，不缩小数据或改为quick。

原始注册后的SOURCE预测缓存已经完成。首次真实恢复pilot因极小浮点差异失败，原日志、
checkpoint及registration.json仍保留。registration_v2.json改为固定membership矩阵求和，
不放宽精确相等要求；新的400步pilot和首个预测器下两版完整重训均要求精确恢复。
首次推理仅因checkpoint父目录缺失退出；补目录后按同一注册和模型恢复。

## 公共复核与完整重训不同

报告脚本只从公开的54组标量JSON、训练回执和复核回执重建汇总与bootstrap，不读取私有
轨迹、标签、预测或checkpoint。私有文件拒绝读取测试单独验证这一点。
训练曲线导出器则读取已核验的SOURCE checkpoint，仅导出标量loss、梯度和训练覆盖计数，
不导出权重、轨迹或标签。这个导出不是重新训练。

完整重训需要合法的原始数据、既有SOURCE预处理与冻结预测器权重。注册和修复的时间顺序
保存在Git和两份注册中。当前仓库还不是从空环境下载数据到重训的完整匿名复现发布包；
不能把公共标量复核说成独立训练或已完成端到端发布。这仍是论文交付的待补项。

## 研究评价

主比较为context matched-joint减其independent策略的全体目标配对completion lower。
逐query匹配介入数量；空matched对照标不可估，只能做效用比较，不能证明安全。
每个原始录像仍须满足原定2%正伤害/参考误差约束和easy保护；不以平均增益抵销正伤害。
全部38,102个目标保留，包括未来完全缺失和部分缺失的目标。

本实验没有优化成对碰撞或组合冲突，没有读取风险校准集、最终确认集或DroneCrowd。
obs8/pred12、rawstride12、local-image pixel、detector-silver；不是完整12步标签指标、
米、秒、人工gold、true3D或foundation证据。不执行Stage5C，不启用SMC。
