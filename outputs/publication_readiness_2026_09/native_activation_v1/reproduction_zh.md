# 启动策略对照：本地复现

这是同一批 34 段开发录像的实验。每个来源独立留出，三组真实预测器种子复用已冻结的预测结果。观测 8 步、预测 12 步，单位是原图像坐标和标注步。此入口不读取 DroneCrowd。

## 环境与运行

在项目根目录运行：

```sh
uname -m
.venv-pytorch/bin/python -c 'import platform, sklearn; print(platform.machine(), sklearn.__version__)'
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py score
.venv-pytorch/bin/python scripts/run_m3w_native_activation.py replay
```

Apple Silicon 应使用原生 `arm64` 环境；实验登记绑定 scikit-learn 版本。
运行器沿用数值库四线程配置，不启动多进程 DataLoader，也不探测 MPS。
源码和登记文件须与提交一致。公开仓库不包含原始轨迹、预测库和模型权重；
仅重读公开 JSON 不等同于跑上述三个阶段。

私有目录是 `data/stage_cvpr2027_experiments/moving_record_joint_actions_v1/native_activation_v1/`。
`heartbeat.json` 记录当前 PID、种子和已完成的留出来源。中断后先用
`ps -p PID` 检查原进程是否仍在运行；确认退出后重跑原阶段即可从已封存
的种子检查点恢复。未完成的种子重跑，不改配置或清空已验证缓存。

## 核查

`fit` 冻结决策和每个来源的阈值；`score` 和 `replay` 在同一份
决策上独立重算。两份运行回执的 `artifact.sha256` 应一致。
回顾性等次数对照使用整段录像的分数；实时对照只用当前查询和开发来源
阈值，所以后者在留出来源上的次数可以不同。不要把前者的结果写成实时策略性能。

这轮无需 CREATE。上游多 GPU 实验才迁移；先用 `squeue -j JOB_ID`
查是否已有运行，再用 `sacct -j JOB_ID` 核查终态和退出码。连接超时
不代表训练停止。恢复沿用相同数据和配置，并核对检查点、日志及指标。
[之前已核验的 CREATE 复算示例](../european_boundary_diagnostic_v1/create_recovery_20261001.md)
不是本轮远端训练证据。
