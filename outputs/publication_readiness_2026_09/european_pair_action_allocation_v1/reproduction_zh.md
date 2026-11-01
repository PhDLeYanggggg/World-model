# 复现说明

本轮是冻结预测器后的真实联合决策实验，没有重新训练神经轨迹模型。重算预测需要授权数据和权重；公开汇总报告生成不需要这些私有素材。
使用原生 arm64 `.venv-pytorch/bin/python`；CPU 4 线程、interop 1、DataLoader workers 0。
## 只核验公开汇总并生成报告

以下两条只依赖已公开、校验通过的汇总指标包和执行记录，不需要读取私有轨迹、缓存或权重；不是重新训练或重算预测。
绘图会在忽略目录创建一个小 PNG 预览，不会生成假的数据缓存。
```bash
.venv-pytorch/bin/python scripts/export_m3w_pair_distance_evidence.py
.venv-pytorch/bin/python scripts/report_m3w_pair_action_allocation.py
```

## 当前工作区可以重复执行的核验

下列命令需要现有授权素材和缓存。第一条会重新求解全部查询；后两条只核验已保存的标量结果及生成报告，不是重新训练。
必须检查本次进程退出码和新日志，不能拿旧 verification.json 冒充本次成功。
```bash
.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py verify
.venv-pytorch/bin/python scripts/export_m3w_pair_distance_evidence.py
.venv-pytorch/bin/python scripts/report_m3w_pair_action_allocation.py
```

## 原始执行记录

这是本轮的执行顺序，不是覆盖当前冻结产物的重跑按钮。当前目录已完成，pilot/eval 和首次距离核算会拒绝覆盖。
从头复算需要注册时的隔离工作区及上游素材映射；通用隔离输出接口仍待补齐，不能声称公开仓库已具备无素材从零复现。
```bash
.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py register
.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py eval
# 距离诊断在主实验运行中完成登记，未据主结果调参
.venv-pytorch/bin/python scripts/run_m3w_pair_distance_diagnostic.py register
.venv-pytorch/bin/python scripts/run_m3w_pair_action_allocation.py verify
.venv-pytorch/bin/python scripts/run_m3w_pair_distance_diagnostic.py
.venv-pytorch/bin/python scripts/export_m3w_pair_distance_evidence.py
.venv-pytorch/bin/python scripts/report_m3w_pair_action_allocation.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_pair_action_allocation.py tests/test_m3w_pair_action_safety.py -q
```
已完成的结果不可覆盖。中断后用 `eval --resume`；只有确认进程已终止才恢复，不能重复启动。
完整 readout PID：59783；复核 PID：64032。
日志、心跳与断点在 data/stage_cvpr2027_experiments/european_pair_action_allocation_v1/；不会进入 Git。
两秒是每个 MILP 的求解预算，不是中止慢实验的总时限。求解失败保留原策略并计入所有评价。
结果只属于已暴露 SOURCE 拟合数据。不要称独立验证、米/秒预测、人工 gold、true 3D 或 foundation。
