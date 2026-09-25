# 本轮运行与复现

本轮只重新推理源域 C、拟合校准映射和评价冻结策略，不训练新的神经网络。
既有模型必须通过文件哈希、来源划分和预测重放核验，不能将缓存模型写成新训练。

## 环境与恢复
使用仓库内原生 arm64 `.venv-pytorch/bin/python`。计算线程为 4，interop 为 1，
DataLoader workers 为 0；入口在导入 Torch 前拒绝 macOS x86_64。
保留 10 GiB 磁盘余量。心跳、PID、锁、分组结果和不可变回执保存在
`data/stage_cvpr2027_experiments/european_bridge_calibration_v1/`。
运行中先检查原 PID 与心跳，不能重复启动。中断后同一阶段重跑会核验既有产物，
不删除已有检查点，不把已经完成的缓存重放称为 fresh training。

## 顺序
```sh
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_calibration.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_calibration.py --phase calibrate
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_calibration.py --phase decide
```
注册必须先提交并推送，才运行拟合；72 个映射、288 个决策视图必须先冻结并推送，
才进行下面的评价。冻结回执绑定私有预测文件，不能修改回执来绕过顺序检查。
```sh
.venv-pytorch/bin/python scripts/run_m3w_european_bridge_calibration.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_bridge_calibration.py
.venv-pytorch/bin/python scripts/verify_m3w_european_bridge_calibration.py
```
最后一步在独立进程恢复模型、重算校准和逐行决策，并运行本方法依赖的回归测试。
它不是未经筛选的全历史测试套件。公开文件必须小于 1 MiB。

## 结果解释
六个评价地点已用于模型选择，只能作为开发证据。置信区间以地点为单位，不能
将重叠窗口或多种子重复视图当作独立样本。保留的 12 个校准地点与 6 个确认地点
仍不打开；不更改 2% 上限，不执行 Stage5C 或 SMC。

CREATE 本轮只读检查队列，没有提交作业，也没有声称远端 M3W 资产清点完成。
本地已有模型和数据，本轮的 CPU 推理/校准成本不足以支持迁移或重复远端训练。
Git 只提交代码、配置、轻量报告和哈希；不提交数据、特征、模型权重，也不处理
其他工作预先暂存的文件。
