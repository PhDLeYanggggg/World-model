# 原始与对齐双视图轨迹模型复现

需要已有 native 数据、过去图像变换代理和 pinned EqMotion。使用 arm64
`.venv-pytorch`，CPU4/inter-op1/workers0。不要重复启动活跃进程。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py admit
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py control --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py verify --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_dual_view.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_dual_view.py
```

已完成pilot/evaluate收据不可覆盖；replay重算12个新预测及全部指标。
原18个对照本轮只校验hash与逐行ID，不说成新推理。原始几何始终保留，
模型输入不含未来相机外推。中途信号在完整更新后停止；异常的半步状态
不会覆盖最后一致checkpoint。不要复用旧单边实验的半步checkpoint。
loss字段mean_past_normalized_ADE沿用旧名称，实际是batch未加权
available-label ADE的packed图像坐标数值，不是米或统一跨域尺度。
两个完整重训核验只覆盖每个arm固定第一design。不是独立测试或部署认证。
