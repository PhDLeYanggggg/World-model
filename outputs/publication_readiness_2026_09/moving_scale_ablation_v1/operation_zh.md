# 尺度输入消融：运行与核查

## 实验边界

这轮只改变风险头的一个输入：屏蔽标准化后的绝对像素尺度字段354。
其余运动、邻居、候选轨迹、标签、采样、预算和原2%规则不变。
主要比较为MSE下的消融；signed-cost版本为预先指定的次要比较。
不是端到端坐标不变模型，也不是新轨迹动力学训练。

六个未屏蔽对照复用上一轮已核验checkpoint，标为cached_verified。
六个屏蔽版本重新训练，才标fresh_run。所有新模型冻结后才读已开放V。
这里的V已反复用于开发，不是独立测试集；保留域与H不开放。

## 顺序

在项目根目录使用原生arm64环境：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_scale_ablation.py
```

这些是干净产物目录下的执行顺序，不要在已有完成产物上重新跑同一阶段。
已完成阶段拒绝覆盖。权威注册为registration_v2.json；首版保留预检记录。
训练前要求代码与注册已提交Git。不得删除产物来绕过冻结检查。

## 断点续训

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py train --resume
```

仅用于未完成阶段；配置、源码、注册和上游资产必须不变。
恢复检查模型、优化器、采样顺序、随机状态和标签/输入hash。
CPU4线程、interop1、DataLoader0，不使用默认x86_64Conda，不探测Torch资源。
每100次更新写checkpoint和heartbeat。正常慢训练可继续，12小时保护。
保留至少5GiB磁盘；本实验私有产物上限512MiB。

本地进度在data/stage_cvpr2027_experiments/moving_scale_ablation_v1/heartbeat.json。
先核查PID、更新次数和时间戳是否推进，不因一次读取失败启动重复进程。
process.lock防止同一实验并发写入。此规模已在本地验证，无需另提HPC作业。

## 如何读结果

- pilot.json：真实100步断点/连续训练是否精确一致，不是泛化成绩。
- training_freeze.json：六个模型是否完整完成，以及是否先于V冻结。
- analysis.json：全部原规则指标及预先指定比较，不能只摘最好场景。
- verification.json：重算预测/动作/指标是否精确一致，不是独立验证。
- full_fitting_objectives.json：完整拟合数据上的损失和决策错误，不是测试集提升。
- gates.md：安全、支持和部署状态；空对照仍不可估，旧失败不修改。

缺失标签completion上下界不是bootstrap置信区间。多个重叠窗口不是多个
独立场景。原24组主结果仍不可估，不能把17个有条件视图冒充完整主结果。
当前数据是观察8步、预测12步的原生像素坐标；不声称米、秒、true3D或foundation。
Stage5C与SMC保持关闭。原始数据、checkpoint、缓存和大型产物不提交Git。
