# 风险矩交叉验证操作说明

## 本轮在验证什么

将允许切换的风险比拆成两部分：分子是切换造成的正伤害，分母是基线误差。
即使分子预测偏保守，只要分母明显高估，风险比仍可能显得过于安全。
本轮用三个 controller 来源训练、一个来源留出，分别检查两项的误差。
这是训练来源内诊断，不是独立测试，也不是替换现有部署策略。

## 环境和恢复

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`。计算线程4、
interop线程1、DataLoader worker0；不使用旧 x86_64 Conda 或资源探测。
每200步保存模型、优化器、随机状态、采样计数和日志。
每个风险头固定2000步，首个100步试跑计入预算，不额外增加候选模型。
写锁防止重复运行；开始新模型前保留至少10GiB可用空间。

恢复已经开始的训练：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_risk_moment_crossfit.py --phase train --resume
```

进度、PID和恢复状态位于
`data/stage_cvpr2027_experiments/european_risk_moment_crossfit_v1/heartbeat.json`。
同目录 `training.log` 与 `events.jsonl` 保留过程日志。
不要删除已有检查点，不要因运行较慢重复启动。

## 核验

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_risk_moment_crossfit.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_risk_moment_crossfit.py --phase verify_eval
.venv-pytorch/bin/python scripts/report_m3w_risk_moment_crossfit.py
.venv-pytorch/bin/python scripts/verify_m3w_risk_moment_crossfit.py
```

评估必须在 prediction_freeze 已提交后执行，不可用留出结果改阈值。
首次完整训练复现由 `scripts/replay_m3w_risk_moment_training.py` 执行；
已经存在复现文件时应核验哈希和记录，不要再造一个训练候选。
结果文件 `summary.json`、`results.md` 和 `gates.json` 区分训练完成、
预测能力和部署结论。`verification.json` 是技术核验，不是效果证书。

## CREATE 和数据边界

这批小风险头在本地完成，没有提交 CREATE 训练。
最近一轮 CREATE 只读查询是父实验的带时间记录，不能冒充本轮资源查询。
后续确需 HPC 时仍遵守 simulation model 项目的授权入口和限制：
只读检查现有作业，计算经调度器，不触碰其模拟作业、环境或认证。

原始数据、几何缓存、逐行预测和权重都不提交 GitHub。
只提交代码、配置、轻量汇总和报告，并保护其他任务的暂存内容。
独立选择、风险校准和最终确认集本轮保持关闭。
没有 Stage5C、SMC、metric 或 seconds-level 声明。
