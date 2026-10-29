# 本轮复现与恢复

本轮检验的是训练集上的介入决策学习，不是新的轨迹预测器，也不是独立测试结果。
安全要求仍为 2%；locality082/51 的候选动作限制保留，不通过删除录像绕过。

## 环境

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_train_feasibility.py tests/test_m3w_cost_sensitive_policy.py tests/test_m3w_cost_sensitive_readout.py tests/test_m3w_cost_sensitive_report.py
```

入口会在加载 PyTorch 前阻止 Apple Silicon 上的 x86_64/Rosetta 路径。
训练使用 CPU 计算线程 4、线程间并行 1，无 DataLoader 子进程。不要改用默认
x86_64 Conda。40 次真实更新试跑已验证模型更新与断点恢复，不仅是 import 检查。

## 已有数据上的验证

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_train_feasibility check
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy check
.venv-pytorch/bin/python -m scripts.report_m3w_cost_sensitive_policy --check
```

后两个命令需要训练及固定读数已完成。检查会核对数据、配置、模型、旧对照和结果
哈希；不重新训练，不选择新的阈值。私有数据与权重不在公开 GitHub 仓库，所以
只有代码不足以从零重现真实结果。必须使用本机已授权且哈希匹配的资产。

## 未完成训练的恢复

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_cost_sensitive_policy train --resume
```

先确认原 PID 已结束；不能因为一次观察超时就启动第二份任务。脚本还有进程锁。
状态保存在 `data/stage_cvpr2027_experiments/european_cost_sensitive_policy_v1/`：
`heartbeat.json`、`events.jsonl` 与 `heads/*.pt.gz`。权重每 100 次更新保存，心跳
每 50 次更新记录。恢复保留模型、优化器、约束乘子及随机状态，仍完成原定 18 个
模型和每个 2,000 次更新。已完成的阶段不会覆盖，应使用 `check`。

## 核查结果

先查看 `training_freeze.json` 中全部 18 个最终模型，再查看 `complete.json` 的
72 组训练读数与 `readout.md`。需要同时看收益、伤害、未标注结果上界、逐录像
安全条件与空对照；不能用 loss 下降代替这些指标。本轮不生成新的留出集结果，
更不把训练集指标包装为泛化提升。

## CREATE

这次约 3 GB 级内存需求适合本机，没有新提交 HPC 任务。只读队列观察没有远程
执行标记，空 stdout 和认证提示不足以确认队列为空或旧任务完成。需要 HPC 时
先确认已有作业，再走既有调度器流程；不要通过重新训练来替代无法取得的状态。

Stage5C 和 SMC 均不执行。原始数据、缓存、模型权重不提交 GitHub。
