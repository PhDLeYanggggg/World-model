# 风险头多种子复验

使用项目原生arm64 `.venv-pytorch/bin/python`；CPU4、inter-op1、workers0。
此实验复用冻结teacher17和上一轮donor/own-B缓存，不复制原始数据。
种子17模型是cached_verified；只新增29、43，共48个模型，另完整重训48个。
不要把它说成三个预测器种子或完整世界模型三种子主实验。

新实验的历史执行顺序如下。每个阶段结束后再运行下一个，不要并发写报告与重放：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py verify
```

首次新增种子评价前已经单独注册并提交mean/unanimity辅助诊断：

```sh
.venv-pytorch/bin/python scripts/evaluate_m3w_donor_seed_consensus.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py evaluate
.venv-pytorch/bin/python scripts/evaluate_m3w_donor_seed_consensus.py evaluate
.venv-pytorch/bin/python scripts/report_m3w_moving_donor_seeds.py
.venv-pytorch/bin/python scripts/evaluate_m3w_donor_seed_consensus.py replay
.venv-pytorch/bin/python scripts/run_m3w_moving_donor_seeds.py replay
```

现有完成目录只需重放，不要覆盖注册记录或重新运行pilot。所有定义须先提交。
模型中断后先检查PID与心跳，健康进程不重复启动。主训练/完整重训可用
`train --resume` / `verify --resume` 在相同代码配置下恢复原子检查点。
每128步保存、每256步写心跳；健康任务允许12小时，磁盘保留5GiB。
心跳路径为本地 `data/stage_cvpr2027_experiments/moving_donor_seeds_v1/heartbeat.json`。

新主训练196,608updates，完整重训另196,608，独立试跑4,608。每个模型4,096步，
每步512次有放回抽样，不能把抽样次数当独立数据量或32epochs。所有三种子保留。
统计按8个来源组分层成2/3/3组，3000次配对bootstrap，不对重叠窗口bootstrap。
这些区间以冻结teacher和已暴露来源为条件，不是新的独立测试结论。

mean/unanimity均用全部三头，阈值固定0；一致同意不是安全证明或恒等回退认证。
新完整策略的2%/easy/未知标签/支持检查尚未完成，部署不变。空对照NA、旧失败
与主24组NA继续保留。Stage5C/SMC不执行。缓存、checkpoint和图像不提交Git。
