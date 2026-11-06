# 复现本轮图像上下文实验

使用项目现有 arm64 `.venv-pytorch`。单进程 CPU4/inter-op1/workers0。
大数组、图像缓存和模型权重只保存在本机，不随 Git 上传。
当前结果是已经暴露的开发集，不是新的独立测试。原 2% 及支持要求不变。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py verify --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_visual_risk.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_visual_risk.py
```

已完成的 pilot/prepare/evaluate 产物保持不可变，不覆盖重跑。
`train` 和 `verify` 可恢复同定义 checkpoint；不降低模型训练步数。
`infer --resume` 恢复原分配过程；每个 query 的限时求解保留可行回退。
`replay` 重算预测和指标，核验冻结决策的可行性与同数量对照；
不要求重新运行限时求解器复现同一搜索路径。
报告脚本只读取公开标量结果，不读取私有未来标签，也不调阈值。
本轮是原生观察 8 步预测 12 步的图像坐标任务，不是米、秒、true3D。
