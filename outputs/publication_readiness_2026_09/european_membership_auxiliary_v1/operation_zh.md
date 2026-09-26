# 辅助监督实验操作说明

本实验比较直接成本预测与加入 easy-membership 辅助监督的模型。
这不是新的轨迹策略，不打开独立选择、校准或最终确认数据。
是否完成以 completion_checks、compute_receipt 和 verification 为准；
计划数量不是实际训练结果，测试通过也不是研究假设成立。

## 环境与恢复

项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。
CPU 线程4、interop1、DataLoader workers0；不使用 Intel Conda/Rosetta。
每200次更新记录心跳和 checkpoint，保留优化器及随机状态。
私有目录的 heartbeat.json 记录 PID；events.jsonl 保存详细训练日志。
磁盘可用空间低于10GiB时停止新折，保留已完成模型，不删除旧结果。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_membership_auxiliary.py --phase train --resume
```

注册必须已经提交；全部288模型完成后，先核验并提交 prediction_freeze.json。
冻结推送成功之前不能做新的留出结果读出。

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_membership_auxiliary.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_membership_auxiliary.py
.venv-pytorch/bin/python scripts/plot_m3w_european_membership_auxiliary.py
.venv-pytorch/bin/python scripts/verify_m3w_european_membership_auxiliary.py
```

验证重放 checkpoint 前4096行，重算完整适用行的成本和分类诊断，检查输入、
标签、抽样、损失尺度、原始 reference 数组，并运行相关测试。完整旧测试集
不在本轮默认范围内。所有角色、种子和负结果都保留，不按留出表现选最好模型。

本轮适合本地运行，未提交 CREATE 新任务，也不修改其他研究任务。
Git 只上传代码、配置、报告和轻量汇总，不上传原始数据、行缓存或 checkpoint。
上述命令依赖本机已有、经过 hash 校验的源数据和上游模型；仅克隆公开仓库
不能直接重现全部结果。本轮验证是本机资产可重放，不等于已交付独立的匿名
复现包，也不代表已经完成论文提交准备。
像素和 annotation-step 不是米或秒；辅助分类提升不能冒充世界动力学提升。
Stage5C、SMC 继续关闭。
