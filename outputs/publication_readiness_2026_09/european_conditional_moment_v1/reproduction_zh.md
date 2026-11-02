# 条件代价实验复现

本实验是 EuropeanSquares 已暴露 SOURCE 拟合对照，不是独立测试，也不是
SDD t+50。观察 8 步、预测 12 步，采样间隔为 12 raw frames。坐标为局部图像坐标，
标注为 detector-silver；不声明米、秒、真实 3D 或 foundation 能力。

## 环境与执行

使用原生 arm64 `.venv-pytorch/bin/python`。训练为 CPU 4 线程、inter-op 1 线程、
DataLoader workers 0。不要改用 x86_64 Conda，也不要同时启动同一实验的第二个进程。

从仓库根目录运行，各阶段必须等待上一阶段成功退出：

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_conditional_moment.py train
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_conditional_moment.py eval
.venv-pytorch/bin/python scripts/export_m3w_conditional_traces.py
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_conditional_moment.py verify
.venv-pytorch/bin/python scripts/verify_m3w_conditional_public_report.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_conditional_moment.py \
  tests/test_m3w_conditional_moment_evidence.py tests/test_m3w_conditional_public.py
```

训练输入和权重留在本机，不随公开仓库发布。缺少私有输入时只能重建已发布的标量报告，
不能宣称重新训练成功。已有 `training_freeze.json`、`complete.json` 等完成标记的阶段
不可覆盖；先检查对应记录，不重复训练。

## 恢复

每 50 次更新保存 checkpoint 和 heartbeat。先读取
`data/stage_cvpr2027_experiments/european_conditional_moment_v1/heartbeat.json`
核对 PID、阶段、上下文、seed、arm 和 step，再确认原进程是否仍在运行。进程健康时继续等待。
只有确认原进程中断后，才对中断阶段增加 `--resume`；不得改预算、划分、输入或注册文件。
所有 36 个主拟合仍须完成原预算，并检查两次完整 refit 数值一致。

## 结果含义

完整结果应包括 36 个拟合记录、208,320 次主优化更新、14,336 次完整 refit 更新、
72 个原始策略评估组和单独数值复核。`training_traces.json` 导出真实训练曲线，不含权重。
固定拟合监控损失不是 validation loss。报告重建测试禁止读取私有数据，但不重新执行优化。

主要对照为 identified 减 marginal 的 full completion-lower utility；不能只挑已知标签收益。
同切换数量比较仅用于效用，不代替所有原始录像的 2% 风险要求。空分母是不可估，
不是安全通过。未知未来仍为未知，不因发生概率预测而被填成真实标签。

三个代价模型 seed 共用 forecaster seed 43，不是三个独立世界模型训练 seed。
本实验没有访问独立保留角色；Stage5C 和 SMC 均不执行。
