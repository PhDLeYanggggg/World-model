# 本轮视角条件化轨迹模型复现

使用 arm64 `.venv-pytorch`，CPU4/inter-op1/workers0。需要现有已校验
native轨迹、过去图像变换代理和 pinned EqMotion 源码。原始数据、缓存、
第三方源码及checkpoint不在Git。不要新建重复训练进程。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py control --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py verify --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py forecast
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_camera_frame.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_camera_frame.py
```

已完成的 pilot/evaluate 收据不能覆盖；replay 复算预测和原始指标。
训练使用同定义checkpoint恢复，不减少4000步，也不改变风险门槛。
完整重训核验仅一个固定design，不能称六个都完整重训。
Camera extrapolation只来自过去七对图像，未来真实相机位姿不作输入。
loss CSV 的 mean_past_normalized_ADE 是沿用字段名，数值实际为未加权
batch available-label ADE的packed图像坐标值，不是米或统一跨场景尺度。
这是训练侧来源排除组件实验，不是新独立测试或完整部署认证。
