# 本轮运行与复现

这轮训练新的风险代价头，不重新训练轨迹预测器。训练仅用源域 B，
源域 C 用于冻结后的评价。六个开发评价地点本轮不评价，预留校准/确认地点不打开。

## 执行顺序
注册代码和协议先提交并推送，再运行真实数据试跑和完整训练：
```sh
.venv-pytorch/bin/python scripts/run_m3w_european_selected_risk_learning.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_selected_risk_learning.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_selected_risk_learning.py --phase train --resume
```
100 步试跑在第一个头的 2,000 步预算内续跑，不另算一个完成模型。
训练完成会冻结 72 个头与 432 个决策视图；先提交、推送冻结回执，再评价：
```sh
.venv-pytorch/bin/python scripts/run_m3w_european_selected_risk_learning.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_selected_risk_learning.py
.venv-pytorch/bin/python scripts/plot_m3w_european_selected_risk_learning.py
.venv-pytorch/bin/python scripts/audit_m3w_selected_query_budget.py
.venv-pytorch/bin/python scripts/verify_m3w_european_selected_risk_learning.py
```

## 运行安全
原生 arm64 环境，计算线程4、interop1、workers0；macOS x86_64 在 Torch
导入前拒绝。单进程采样，不做资源探测或多进程 DataLoader。保留10GiB磁盘。
检查点、PID、心跳、锁、固定检查批次损失、随机状态和样本计数保存在
`data/stage_cvpr2027_experiments/european_selected_risk_learning_v1/`。
先检查原进程和心跳，不因一次观察超时重复启动。中断后使用相同配置和
`--resume`，不可删除旧检查点来伪造 fresh run。完成回执会校验文件哈希。

当前数据与模型在本地，真实试跑用于估计完整计算时间；推理和源域重建耗时
另计。CREATE 本轮只读检查队列，不改其他任务，不重复提交。远端 M3W 资产
清点仍须单独证据，队列成功不等于远端模型验证完成。

## 解释结果
固定批次 loss 下降只说明拟合进展。必须同时查看留出源域准确率、简单样本
净退化和正伤害约束。场景预算是预测值，不是物理安全或分布无关保证。
Bootstrap 以地点为单位，不能把窗口或重复源域设置当独立样本。
无独立确认时不升级部署，不执行 Stage5C/SMC，不写 metric 或 seconds。
只提交代码、配置、轻量统计与报告；不提交数据、缓存、权重或第三方资源，
不碰其他工作已暂存的文件。
