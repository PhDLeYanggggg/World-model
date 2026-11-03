# 解码监督续训：运行与复现

## 当前范围

这是同一批 18 个冻结成本头的匹配续训，不是新的轨迹网络或大模型。
两个版本均从真实 2,000 步 checkpoint 的模型、Adam、采样器及随机状态
开始，再训练 2,000 步。共 36 个续训任务、新增 72,000 步；旧步数不重复
计入新增计算。输入、归一化、容量及 2% 风险要求均不改变。

原目标继续训练是主对照；额外监督只作用于现有解码 logits，不增加参数、
不输入未来。辅助标签是成本分量，不是部署时可见的最佳 baseline 类别。
它改变损失几何和样本权重，不能仅归因为浮点修复，也不宣称 log-loss 新颖。
此前失败的 log-loss／hurdle 实验保持失败。

## 执行顺序

在项目根目录使用 arm64 `.venv-pytorch`，CPU 4 线程、interop 1、workers 0。
不用 Rosetta／x86 Conda，不做 Torch 资源探测。先登记并提交完整代码和
registration.json，再试跑，之后完整训练。当前这两步已经完成：真实
6 单元 50+50 对照直接 100 步试跑退出 0，耗时 72.122 秒，峰值内存约
7.43GB；正式训练已开始。以下不是“所有阶段已完成”的声明。

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py register
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py pilot
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py train
.venv-pytorch/bin/python -u scripts/audit_m3w_moving_link_fit.py evaluate
.venv-pytorch/bin/python -u scripts/audit_m3w_moving_link_fit.py verify
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py evaluate
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_link_repair.py
```

固定区域检查在本轮新留出策略读出前完成。三个评分版本分别对同一批
原选择区域、继续训练选择区域、新选择区域及全神经区域计分，不能只报
最有利的区域。未来标签只在因果选择向量形成后用于诊断。它不挑 checkpoint，
不改阈值或停止规则；完整预算始终保留。

## 监控和恢复

私有目录：`data/stage_cvpr2027_experiments/moving_decoder_link_repair_v1/`。
查看 heartbeat.json、events.jsonl、training.log 和实际 PID。每 100 步保存
原子 checkpoint 并更新心跳。只有确认旧进程已结束后，才恢复未完成任务：

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_decoder_link_repair.py train --resume
```

已完成阶段会拒绝覆盖；不要删除有效 checkpoint 或收据。保持原 10GiB
磁盘余量和单次 12 小时边界。慢不构成减预算的理由。真实资源不足时，
核对已有 CREATE 任务与自己目录后再迁移，避免重复作业。本轮实测资源
适合本机；未新提交 HPC 作业。

## 如何读结果

主要固定开发对比是 auxiliary 与等预算 continuation 的 joint-free 配对
completion-lower，另列已知标签、全部六种策略、easy／hard、逐录像风险、
切换率与同数量效用。旧冻结 2k 策略是参考，不是主等预算对照。训练图用
两版本共有的原损失，不能把不同总损失曲线直接比较。

23 个支持组只提供已暴露来源的开发二级结果；原 24 组主结果仍不可估。
重复种子／producer／录像视图不是独立地点；同数量空对照不可估，不能
算安全通过。均值改善、loss 降低或训练通过都不能替代原完整风险要求。
校准／最终确认仍未打开；不声称可投稿、metric／seconds／true3D／foundation。
不执行 Stage5C，不启用 SMC。

模型、采样状态和预测大数组只保存在本地。Git 只保存代码、配置、报告、
自绘图和轻量指标；完整重跑需要已核验的数据与私有 checkpoint。
