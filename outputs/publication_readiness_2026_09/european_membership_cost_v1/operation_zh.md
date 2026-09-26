# 本轮运行与恢复

使用原生 arm64 `.venv-pytorch/bin/python`，CPU 4线程、interop 1、workers 0。
不要切换到 x86_64 Conda，也不要因观察超时重启已有进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_membership_cost.py --phase train --resume
```

100步试算计入原定2000步预算。checkpoint 保存模型、优化器、随机状态、
采样计数和预处理；每200步保存并写 heartbeat。中断后先确认旧进程已终止，
再使用 resume，不追加训练步数。保留10GiB磁盘余量。

所有模型训练完成后，先提交推送 prediction_freeze.json，再统一读取留出结果：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_membership_cost.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_membership_cost.py
.venv-pytorch/bin/python scripts/plot_m3w_european_membership_cost.py
.venv-pytorch/bin/python scripts/verify_m3w_european_membership_cost.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_membership_cost.py
```

固定概率对照不是重新训练的模型。条件专家额外使用上一轮分类器，因此必须
报告总模型成本差异。两条路线损失尺度不同，不能直接把较小训练loss当作更好。
推理不接收真实简单样本标签，分类器概率也不参与新成本头的训练。

本轮 CREATE 仅只读查看队列；没有提交或修改其他项目任务。M3W 远程产物目录
仍未验证，不能把队列可访问写成远程训练已完成。

只提交代码、配置、聚合指标、报告和矢量图，不提交原数据、逐行结果、权重、
缓存、图像或环境。不动其他暂存文件。常规审计已授权，无需用户逐项审核。
独立数据角色、既定风险容忍度、Stage5C/SMC禁令和正式投稿确认边界不变。
