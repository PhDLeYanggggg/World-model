# 相对实际回退策略的训练：复现与恢复

## 这次训练什么

轨迹预测器保持冻结。本轮训练的是判断神经轨迹是否值得替换实际回退轨迹的增益和伤害头，
不是重新训练一个大型世界模型。四种对照共享过去信息、预测轨迹、容量和训练预算；
分别改变增益目标、伤害目标或两者。旧的离线回退修正也是必须保留的对照。

训练样本上的回退策略由另外两个场景训练产生，不能使用见过该样本标签的模型。
每组八个外部开发场景从整条生产链排除。它们仍属于已经打开过的开发集，不能重新叫作独立测试集。

## 环境与文件

项目根目录：`/Users/yangyue/Downloads/World`。
使用原生 arm64 的 `.venv-pytorch/bin/python`，不要使用 x86_64 Conda。
计算线程4、交互线程1，DataLoader worker0。CPU 上的真实 Torch 训练不等于 NumPy 回退。

配置：`configs/m3w_european_floor_relative_v1.json`。
本地检查点、得分数组、运行日志：`data/stage_cvpr2027_experiments/european_floor_relative_v1/`。
其中 `heartbeat.json` 记录最近状态，`events.jsonl` 保留 PID、更新步和阶段事件。
`inner_shared/` 是两组配置共用的内部增益头；各组的 `inner_risk/`、`heads/` 保存对应模型。

每个头2,000次更新，每200次保存检查点。已有完整头会验证身份和文件哈希后复用；
中断的头从模型、优化器和采样器状态继续，不从零重跑。磁盘低于10GiB会停止并保留产物。
不要删除检查点来解决慢的问题。源码或绑定数据变化会被身份检查拒绝，不应强行绕过。

## 按顺序运行

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode batch --phase inner --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode fitting --phase inner --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode batch --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode fitting --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode batch --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode fitting --phase decide
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode batch --phase evaluate --resume
.venv-pytorch/bin/python scripts/run_m3w_european_floor_relative.py --mode fitting --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_floor_relative.py
```

两组完整决策都冻结之前，评价入口会拒绝运行。已有评价只允许验证后恢复，不能修改阈值重做。
同一个工作目录有运行锁，不要并发启动两组训练。首次部署到另一台机器还需要合法获取
本地前置数据与冻结预测器；GitHub 上的轻量报告不包含这些大文件。
报告入口要求已有成功的针对性测试回执，测试文件清单见 `completion_checks.json`。

## 阅读结果

先读 `conclusions.md`，再读 `results.md` 和 `summary_metrics.json`。
`view_metrics.json` 包含每组条件置信区间；按折保存的 CSV 包含场景级均值、尾部和最差场景。
`training_inventory.csv` 保留每个训练头的更新数、参数数、损失和重放核查。
固定训练批次的风险损失与随机批次的增益损失不能混作验证损失，更不能单靠损失下降宣称泛化。

工程核验、开发集改善、独立风险校准、最终确认是四件不同的事。
重叠窗口不是独立样本，36个配置视图也不是36次独立实验。
本协议为像素坐标、观察8步预测12步、原始标注步长12，不与历史raw-frame t+50直接等同。
Stage5C和SMC关闭，当前部署不变。常规核查由项目流程执行，不需要用户逐项审计。
