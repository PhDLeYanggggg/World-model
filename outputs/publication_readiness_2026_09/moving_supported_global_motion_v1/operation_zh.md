# 四格神经模型实验操作

在项目根目录使用原生arm64 `.venv-pytorch/bin/python`。不使用Intel Conda、
多进程DataLoader或Torch资源探测。固定CPU4线程、inter-op1、workers0。

定义已提交后，顺序运行以下阶段。每步以真实退出与产物核验为准：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py infer
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_supported_global_motion.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_supported_global_motion.py
```

中断后对原阶段追加`--resume`，不删除checkpoint、不改变定义或另启重复进程。
训练每128次更新保存，256次更新写心跳。各阶段单独12小时上限，5GiB磁盘
保留空间和新私有产物3GiB上限。先核查心跳、PID和checkpoint是否增长；慢不
等于卡死，不据此降级目标。没有新提交CREATE作业，也不推断远端旧作业状态。

训练与完整重训各96模型、393,216次更新。verify核查同实现数值可重现，
不是新的独立科学重复。所有36组因果选择固定后才评分，不用评估调阈值。
replay重算预测和指标，并检查旧求解器选择的可行性；不重新进行限时整数
优化，也不保证另一次优化获得相同路径。不要与报告写入同时执行。

公开执行收据在本目录executions；私有心跳、事件、checkpoint、数组与图在
`data/stage_cvpr2027_experiments/moving_supported_global_motion_v1/`。
只提交代码、配置、报告、哈希与轻量指标，不能提交这些私有大文件。

结果必须同时读全部四格、固定小幅度强对照、72次录像检查、NA和尾部；
训练loss下降不算风险通过。保留原2%规则、空支持不可估、旧失败结论。
不执行Stage5C，不启用SMC，不宣称秒、米、true3D或foundation成功。
