# 风险预算目标：训练与复现

## 目标和区别

本轮只把两项误差的独立平方损失，改为“正伤害减去2%基线误差”的平方损失。
预测器、355维输入、网络、初始化、采样、三个种子和训练预算保持一致。
新网络两项输出只是有符号分数的内部计算量，不能当作分别校准的误差预测。
这不是新部署策略；独立选择、校准和最终确认集没有打开。

## 运行

项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。
计算线程4，interop线程1，worker0；不使用 x86_64 Conda 或资源探测。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_risk_excess.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_risk_excess.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_risk_excess.py --phase verify_eval
.venv-pytorch/bin/python scripts/report_m3w_risk_excess.py
.venv-pytorch/bin/python scripts/plot_m3w_risk_excess.py
.venv-pytorch/bin/python scripts/verify_m3w_risk_excess.py
```

每200步保存检查点和心跳，恢复时检查身份、优化器、随机数和采样记录。
每个头2000步，首个100步试跑在此预算内。保留10GiB磁盘空间。
日志位于 `data/stage_cvpr2027_experiments/european_risk_excess_v1/`，
`heartbeat.json` 记录PID和当前模型，`training.log` 记录完整过程。
不要删除已有检查点或因运行较慢重复启动。

评估要求 prediction_freeze 已提交。不能看留出结果后改损失、挑种子、
挑检查点或改阈值。固定首个模型的完整重新训练由
`scripts/replay_m3w_risk_excess_training.py` 执行，属于核验开销，不是新候选。
已有重训记录时应核验，不再重复运行。

## 证据边界

先区分损失预测更准、切换子集是否风险失准、完整部署策略是否有效。
这三者不等价。2%的风险阈值没有放宽，技术测试通过不代表安全证书。
完整历史测试套件和冷启动原始数据重建是否运行，见 verification.json。
逐行数据和权重不提交GitHub；只提交代码、配置、汇总、报告和统计图。

本批小风险头本地运行。CREATE只读查询不等于远程训练或资源保证；
继续遵循 simulation model 项目提供的连接和保护限制，不修改其作业。
没有 Stage5C、SMC、metric 或 seconds-level 声明。
