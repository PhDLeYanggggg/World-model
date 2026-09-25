# 本轮复现与恢复

本轮实际训练 36 个 PyTorch 风险预测头，共 72,000 次更新。它不是新的轨迹
生成模型，也不是端到端世界动力学训练。使用原生 arm64 环境、4 个计算线程、
1 个 interop 线程、0 个数据加载进程；不使用默认 x86 Conda。

## 执行顺序

在项目根目录使用 `.venv-pytorch/bin/python`。顺序如下：

```bash
.venv-pytorch/bin/python scripts/audit_m3w_easy_harm_training_support.py
.venv-pytorch/bin/python scripts/run_m3w_european_easy_harm_sampling.py --phase register
# 先提交并推送注册方案，之后才能训练。
.venv-pytorch/bin/python scripts/run_m3w_european_easy_harm_sampling.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_easy_harm_sampling.py --phase train --resume
# 先提交并推送 decision_freeze.json，之后才能读取 C 结果。
.venv-pytorch/bin/python scripts/run_m3w_european_easy_harm_sampling.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_easy_harm_sampling.py
.venv-pytorch/bin/python scripts/plot_m3w_european_easy_harm_sampling.py
.venv-pytorch/bin/python scripts/audit_m3w_easy_harm_transport.py
.venv-pytorch/bin/python scripts/verify_m3w_european_easy_harm_sampling.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_easy_harm_sampling.py
```

已有产物会校验身份和哈希，不会默默重新训练或覆盖原结果。模型每 200 步保存
优化器、随机数状态、采样计数和训练轨迹，恢复使用 `--phase train --resume`。
保留 10 GiB 空闲磁盘；低于此值时保留检查点并停止新增训练。发生版本或身份
不匹配时先诊断，不删除旧产物绕过限制。完成后重复训练入口属于缓存验证，
不是 fresh_run；完成后的重放验证不等于重新拟合全部模型。

私有日志、PID、心跳、检查点在
`data/stage_cvpr2027_experiments/european_easy_harm_sampling_v1/`。
不要把该目录提交 Git。不要启动重复训练进程，也不要因读取日志超时终止训练。
本轮 CREATE 只进行了只读队列检查，没有提交任务或修改其他项目作业；远程
M3W 资产清单没有重新核验，不能声称已有远程训练结果。

## 结果含义

`training_metrics.json` 记录统一固定 B 批次损失和重要性加权训练批次损失。
两者不是相同样本上的配对测量，不直接比较单步数值。`results.md` 记录全部
C 结果，`diagnostic_summary.json` 记录暴露率和选择后伤害估计误差，
`verification.json` 记录重放和测试范围。不把平均 easy 保持误写为条件正伤害
控制，也不把源场景开发结果误写为独立验证。

观察 8 / 预测 12 个标注步，raw stride12，图像像素，检测器衍生标签。
没有米、秒、人工 gold、物理安全、true3D、foundation 或投稿就绪声明。
预留校准与确认场景未打开，Stage5C 和 SMC 未执行。
