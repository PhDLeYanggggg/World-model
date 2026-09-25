# 参考成本保护实验操作记录

## 范围
本实验对已训练的风险预测头进行同预算继续训练，不重新训练轨迹预测器。
两组初始权重、抽样顺序、B 域预处理与损失尺度一致。保护组固定参考成本
预测，只更新伤害预测；继续训练组更新整个共享预测头。结果不自动部署。

## 环境和恢复
在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`，CPU 四线程，
interop 一线程，DataLoader workers=0。不要使用 x86_64 Conda。checkpoint
包含优化器、随机数状态、累计更新数和抽样计数；中断后沿用相同注册配置：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_reference_protection.py --phase train --resume
```

本地私有目录为 `data/stage_cvpr2027_experiments/european_reference_protection_v1/`。
其中 heartbeat.json 记录 PID、UTC、组别和更新进度；events.jsonl 保留事件流。
保留至少 10 GiB 空闲磁盘。只观察到超时不能判定卡死，也不能重复启动训练。
已完成模型按清单复核后复用，不重新拟合。

## 固定后读出
训练完成须先核查 72 个预测头、144,000 次新增更新、36 组决策文件和
576 个策略视图。提交并推送 decision_freeze.json 后才运行：

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_reference_protection.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_reference_protection.py
.venv-pytorch/bin/python scripts/plot_m3w_european_reference_protection.py
.venv-pytorch/bin/python scripts/verify_m3w_european_reference_protection.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_reference_protection.py
```

复核会检查模型预测、受保护参考分量、两组抽样和随机数状态、全部策略决策、
风险质量统计以及相关回归测试。正式结论以最终 verification.json 为依据；
训练成功不代表科研假设成立，相关测试也不代表整个历史测试集已运行。

## 数据边界
C 是本次 A/B 拟合之外但历史已使用的发展数据，不是独立确认集。本轮不打开
六个模型选择地点、十二个预留校准地点或六个确认地点。不使用 test 调阈值。
仅报告图像像素和注释步，标签为 detector-derived；不能写米、秒、人工 gold、
物理安全、true3D 或 foundation 成功。Stage5C 和 SMC 关闭。

## CREATE 与版本控制
本轮只读检查 CREATE 队列，没有提交、取消或修改作业；远程 M3W 数据资产
清点仍未运行。队列能连接不等于远程资产已验证。Git 只提交本实验显式文件
清单，不提交私有数据、checkpoint 或大缓存，也不改动其他任务已暂存文件。
