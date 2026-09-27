# 场景分离校准：操作说明

本实验不重新训练轨迹预测器或风险网络。先核验已有原生 Torch 权重及其历史
校验记录，再重新推理。权重属于 cached_verified；本轮推理、校准和评价属于
fresh_run。不能把重复使用权重写成本轮训练。

在项目根目录使用原生 arm64 的 `.venv-pytorch/bin/python`。计算线程4、互操作
线程1、DataLoader workers0，不进行硬件资源探测。缓存及权重保留在被 Git
忽略的 `data/stage_cvpr2027_experiments/` 下，至少保留10GiB空闲磁盘。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase infer --resume
# 校准前，先提交 score_freeze.json。
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase calibrate --resume
# 检验结果计算前，先提交 decision_freeze.json。
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_infer
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_calibrate
.venv-pytorch/bin/python scripts/run_m3w_european_score_support_calibration.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_score_support_calibration.py
.venv-pytorch/bin/python scripts/plot_m3w_score_support_calibration.py
```

`heartbeat.json` 和 `events.jsonl` 记录进程PID、阶段及组进度。恢复时先核验已完成
的组，只重算中断组，不重训冻结网络。文件锁阻止同时写入；不能仅凭时间戳变旧
就删除锁或重启，须先检查实际进程。不可变记录会拒绝被改动的代码、权重、分数
或结果。

首次注册出现三个历史文件名冲突，已在任何推理开始前按提交15768a5d逐字节恢复。
新实验使用独立的 `score_support_calibration` 名称。旧预测器心跳已从原来的最后
终止事件恢复；误写的两条注册事件仍留在追加式事件日志中，便于追溯。旧权重、
预测和数值结果没有被替换。

本说明依赖已有本地数据、冻结预测库及完整上游记录，不代表已完成从原始下载开始
的重建或匿名复现。没有打开独立评价角色，没有提交HPC作业，也没有执行Stage5C
或启用SMC。每个校准视图仅两个场景，所得规则是经验性筛选，不是安全保证。
