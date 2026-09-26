# 本轮执行和恢复

在项目目录使用原生 arm64 环境，CPU线程4、interop1、workers0。
不使用默认 x86_64 Conda；不启用 DataLoader 多进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_easy_membership.py --phase train --resume
```

100步真实试跑已纳入第一个线性模型的2000步预算。恢复读取模型、优化器、
随机数、采样计数和预处理状态，不额外重跑预算。每200步保存checkpoint和
heartbeat。私有目录 data/stage_cvpr2027_experiments/european_easy_membership_v1/
保存权重、逐行预测和日志。观察超时不是进程失败；先核查原进程，不重复启动。
保持10GiB磁盘余量，出现硬错误保留checkpoint后诊断。

训练结束后先提交并推送 prediction_freeze.json，之后运行：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_easy_membership.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_easy_membership.py
.venv-pytorch/bin/python scripts/plot_m3w_european_easy_membership.py
.venv-pytorch/bin/python scripts/verify_m3w_european_easy_membership.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_easy_membership.py
```

先保留全部分组结果，再写结论。标签是“基线正误差处于训练easy范围内”，
不是“发生正伤害”；零基线误差按原定义不算该标签的正例。主对照仅使用
训练prevalence，另报训练条件prevalence敏感性，不用留出标签重新定常数。

CPU本地足以处理这些小模型。最近CREATE只读观察属于cached_verified，
不是当前空闲GPU证明，也不是已确认M3W远程目录。不要改动其他项目作业。

只提交代码、配置、报告、SVG和聚合指标。不要提交原数据、逐行概率、
缓存、权重、视频图像或环境；不碰已有无关暂存项。常规审计已授权执行，
无需反复要求用户确认。独立确认集、Stage5C、SMC和正式投稿边界不变。
