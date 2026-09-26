# 本轮运行与恢复

当前是一个固定辅助损失的受控实验，不是新的轨迹预测器或部署策略。
项目根目录使用原生 arm64 Python，保持计算线程4、interop1、workers0。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_support_fractional.py --phase train --resume
```

100步真实试跑属于首个模型的固定2000步预算，恢复不会多加训练步数。
私有目录 data/stage_cvpr2027_experiments/european_support_fractional_v1/
保存逐模型检查点、优化器、采样随机数、heartbeat.json及events.jsonl。
观察超时不是训练失败，先核对同一PID，不要重复启动。保留至少10GiB磁盘。

全部144个新头完成后，将prediction_freeze.json推送远程，再读取当前结果：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_european_support_fractional.py
.venv-pytorch/bin/python scripts/report_m3w_european_support_fractional.py
.venv-pytorch/bin/python scripts/plot_m3w_european_support_fractional.py
.venv-pytorch/bin/python scripts/verify_m3w_european_support_fractional.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_support_fractional.py
```

核验包括旧/新采样一致、预处理一致、检查点预测重放、训练分箱重放、
逐坐标误差和另一套排序/尾部指标计算。最后运行当前链路相关测试，
不把这些范围化测试说成整个旧项目都已通过。

当前训练成本适合本机，没有提交CREATE作业。引用的队列记录是带时间的
历史只读观察，不代表当前空闲资源。尚未核实M3W远程项目目录，不得把
simulation model的目录或运行任务当成可修改的M3W资源。

只同步代码、配置、报告和轻量聚合结果；不提交私有数据、模型权重、
历史缓存、图片或.venv-pytorch，不碰已有无关暂存项。无论结果正负，
保留全部来源安排和种子。Stage5C、SMC保持关闭，不开预留确认集。
