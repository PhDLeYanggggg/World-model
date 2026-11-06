# 本轮真实轨迹训练复现

使用现有 arm64 `.venv-pytorch`，CPU4/inter-op1/workers0。
需要本地已校验的来源轨迹、过去图像派生特征及 pinned EqMotion 代码。
数组、缓存、第三方源码和 checkpoint 不在 Git 中。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_actor.py control --resume
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py train --resume
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py verify --resume
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py forecast
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py evaluate
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_visual_actor.py
```

已完成的 pilot/evaluate 收据不可覆盖重跑；核验已有产物即可。
训练在同定义断点上恢复，已完成模型校验后复用；不得以 quick 替代4000步。
resource_amendment 只将磁盘工程预留由5GiB改为2GiB，保留1GiB产物上限。
原模型、训练终点和2%科学评价规则不变，不删除旧数据。
replay 重算所有预测和原指标，不进行新优化。
loss CSV 的 mean_past_normalized_ADE 是旧字段名，实际为未加权 batch
available-label ADE 的原始 packed 图像坐标值，不是米或不同场景统一尺度。
当前是训练侧来源排除组件实验，不是新的独立测试或部署结果。
