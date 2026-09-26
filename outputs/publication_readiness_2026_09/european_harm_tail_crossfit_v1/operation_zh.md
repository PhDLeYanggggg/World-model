# 本轮实验操作与恢复

本轮只训练风险诊断头，不改预测器、部署阈值、主指标或 2% 风险容忍度。
所有命令在项目根目录运行，使用本机原生 arm64 环境。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_harm_tail_crossfit.py --phase train --resume
```

训练为单进程，计算线程 4，interop 1，DataLoader workers 0。检查点保留
模型、优化器、采样随机数及已完成更新数。中断后用上述命令恢复；观察超时
不是训练失败，不要据此重启或创建重复进程。私有目录中的 heartbeat.json
记录 PID、当前模型、步骤和时间，events.jsonl 保留过程。

只有全部 144 个头完成、产物哈希一致，并将 prediction_freeze.json 提交
到远程后，才运行诊断读出：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_european_harm_tail_crossfit.py
.venv-pytorch/bin/python scripts/report_m3w_european_harm_tail_crossfit.py
.venv-pytorch/bin/python scripts/plot_m3w_european_harm_tail_crossfit.py
.venv-pytorch/bin/python scripts/verify_m3w_european_harm_tail_crossfit.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_harm_tail_crossfit.py
```

核验会重放检查点预测、训练采样和评分分箱，核对逐坐标误差，以及用另一套
算法核对 AUROC、AP、尾部伤害捕获。最后运行当前链路相关测试，不把范围化
测试说成所有旧代码都通过。JSON 和报告中的未运行项必须保留其实际状态。

磁盘自由空间须保留至少 10 GiB。不要删除用户无关数据以凑空间。
本轮 CREATE 仅有只读队列检查，没有新提交作业；这不代表远程模型资产已
核实。后续若需要迁移，应遵循 simulation model 项目确认的调度和目录限制。

只提交当前实验代码、配置、报告和轻量聚合结果。不要提交 data 下的私有
模型或缓存；不要带入已有无关暂存项。结论须区分训练完成、核验通过、研究
假设被支持，不能互相替代。Stage5C 和 SMC 不执行。
