# 本轮诊断的运行与恢复

## 环境和数据

在 `/Users/yangyue/Downloads/World` 中使用原生 arm64 环境
`.venv-pytorch/bin/python`。不要使用 x86_64 Conda 或 Rosetta 下的训练环境。
入口在导入数值库前检查架构；运行配置为 CPU4、inter-op1、DataLoader workers0。
本轮没有新增 CREATE 作业，也没有把本地历史记录当成当前调度器状态。

这不是从 GitHub 空仓库即可复现的数据产品。还需要获得授权的原数据、原划分、
已冻结的预测器与风险头检查点。注册文件记录其哈希；数据和大权重不上传 Git。
不能用任意新数据或不同检查点替代后仍称为同一复现。

## 阶段命令

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_full_objective_errors.py
.venv-pytorch/bin/python scripts/diagnose_m3w_full_objective_errors.py --verify
.venv-pytorch/bin/python scripts/report_m3w_moving_full_objective_risk.py
```

这些命令有固定依赖顺序。已完成阶段拒绝覆盖原结果，并不是运行失败。
注册的代码、配置和协议必须与提交版本一致。`evaluate` 在训练内部筛查失败时
只写 `not_run`，不读取新的验证预测；该命令退出0不能被称为完成验证评价。
`verify` 重算完整训练指标，若确实执行过验证评价，再重放其预测和决策。

## 中断恢复

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py train --resume
```

仅在原阶段被中断、且没有完成回执时恢复。不要改配置后复用同名检查点。
每5个完整外层步骤保存模型、优化器历史和随机状态，每10步写心跳。
线搜索中途退出不保存临时试探权重，恢复时重算最后一次保存后的工作。
同一实验的进程锁防止重复启动。先检查 PID、心跳和日志，不因等待输出就重启。

运行记录位于 `data/stage_cvpr2027_experiments/moving_full_objective_risk_v1/`。
检查点在其中的 `fits/`；`heartbeat.json` 和运行事件记录可用于确认是否推进。
保留至少5GiB磁盘余量；本轮私有产物上限512MiB，健康运行上限12小时。
达到200步只是预定预算完成，不等于数值收敛；应同时查看梯度和错误代价。

## 如何读结果

`training_freeze.json` 证明三个训练结果已冻结，`fitting_screen.json` 决定
是否值得再读已暴露的开发集。它不是新的安全规则。`analysis.json` 明确区分
`fresh_run` 与 `not_run`。本轮损失是风险回归误差，不是轨迹 ADE/FDE。
新增归因仅使用各自训练集，分别报告放行有害动作与拒绝有益动作的代价。
训练损失下降、恢复成功、程序退出0，都不能替代原2%安全和支持要求。

旧失败不改，空对照仍不可估，未使用恒等回退认证。保留测试数据继续关闭。
当前为观察8步/预测12步图像坐标实验，不是米、秒或新的真实3D证据。
Stage5C和SMC均未启用。
