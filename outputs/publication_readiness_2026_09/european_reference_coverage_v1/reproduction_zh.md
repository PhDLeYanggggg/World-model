# 参考轨迹覆盖实验：复现和恢复

工作目录：`/Users/yangyue/Downloads/World`。使用原生arm64的
`.venv-pytorch/bin/python`，不要用x86_64Conda。固定CPU4线程、interop1、
DataLoader workers0。代码不做GPU资源探测；本轮也没有新的HPC训练作业。

## 已完成的真实工作

9组神经模型、36,000次主要更新已完成；不是NumPy替代训练。
真实80步试跑验证中断恢复，完整4,000步中性对照精确一致。
模型6组未改变半径，完整状态及预测与原模型相同；3组改变输出半径，
但训练行、抽样次数和随机状态相同。模型终点先提交，再读取结果。

主结果是观察8步、预测12步的SOURCE开发比较，不是t+50。
相对上一轮模型仅改善0.063559个百分点，easy退化仍8.345904%。
不能据此升级部署。数据是pixel/raw-frame和detector tracks，不是米、秒、
人工gold、true3D或foundation结果。

## 固定实验命令

```bash
.venv-pytorch/bin/python scripts/run_m3w_reference_coverage.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_reference_coverage.py --pilot --neutral --train
```

同名实验已完成后，这些命令只验证/复用已有终点，不产生一轮新的独立训练。
不要删除旧checkpoint来伪造fresh_run。需要独立重训时，应另建实验身份并
保留原控制和协议。当前代码/配置修改会触发身份校验，不能悄悄覆盖。

真正中断而尚未完成时：

```bash
.venv-pytorch/bin/python scripts/run_m3w_reference_coverage.py --neutral --train --resume
```

每200步原子保存checkpoint，每50步写heartbeat和events。当前运行产物在
`data/stage_cvpr2027_experiments/european_reference_coverage_v1/`，不提交Git。
不能把进程存在或出现loss当作完成；检查全部9个终点及退出状态。

## 固定读数和精确重放

先提交`training_freeze.json`，再运行：

```bash
.venv-pytorch/bin/python scripts/report_m3w_reference_coverage.py
.venv-pytorch/bin/python scripts/verify_m3w_reference_coverage.py --readout
.venv-pytorch/bin/python scripts/analyze_m3w_reference_coverage_support.py
.venv-pytorch/bin/python scripts/verify_m3w_reference_coverage.py --support
.venv-pytorch/bin/python scripts/diagnose_m3w_remaining_support.py
.venv-pytorch/bin/python scripts/diagnose_m3w_remaining_support.py --verify
.venv-pytorch/bin/python scripts/diagnose_m3w_multiaction_support.py
.venv-pytorch/bin/python scripts/diagnose_m3w_multiaction_support.py --verify
```

核验命令重新计算并核对已有科学产物，不改模型、不调阈值、不生成独立测试。
新的`verify_m3w_reference_coverage.py`入口保留首次执行收据，只在科学hash
和检查结论完全相同时接受新的PID/耗时；原冻结读数代码不改。重复核验时
使用这个入口，避免原始一次性收据因运行时间不同被误判为结果变化。
支持诊断的首次读取会只读获取已冻结的CREATE SOURCE TRAIN包；逐包验证原
byte count和SHA后缓存。缓存仅约420MiB，最多4GiB，须保留10GiB磁盘空间。
重放仅用已核验本地字节，无需网络。SSH255不等于远程训练失败，不得因此
重提、重启、取消作业或更改认证。首次中断和恢复经过保留在
`transport_recovery.md`。原始包、模型、特征和预测缓存均不入Git。

原neural-only支持检查仍147/148；扩展physical action bank的148/148只是
标签oracle。它使用未来标签选择离线证据，不能作为inference policy。
完全等于floor的动作被排除，不用虚假介入获得安全通过。下一轮多动作
学习模型尚未训练。旧失败、2%要求及空对照不可估规则均不改变。

## 证据检查

70项相关测试通过，覆盖80个小型多动作问题的穷举对照；没有重跑完整
历史测试库。所有真实读数的重放是同环境验证，不声称跨平台bitwise保证。
独立selection/calibration/confirmation未打开，论文没有提交。
当前官方时间和格式核验见`submission_checkpoint_20261029.md`。
Stage5C与SMC始终关闭。
