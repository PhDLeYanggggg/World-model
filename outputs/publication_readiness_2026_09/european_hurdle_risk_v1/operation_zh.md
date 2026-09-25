# 发生概率与伤害程度分解实验：操作说明

## 范围

本轮仅重训风险头，不训练新轨迹预测器，不替换部署模型。输入、原始收益头、
预测轨迹、场景角色与2%风险预算固定。比较相同三输出网络下的两个目标：
直接拟合平均成本，以及平均成本加伤害发生概率、发生后的伤害程度监督。
后者属于已有两部分模型思想的应用，不能仅凭结构命名宣称方法创新。

所有12个European Squares locality均已用于开发。每次拟合使用4个，另外8个
从完整预测器训练链中排除，但这些结果仍不是独立最终测试。检测轨迹是像素、
观察8步预测12步、raw stride12，不是t50、秒、米、人工gold或物理安全。

## 原生环境与运行

在项目根目录使用原生arm64环境；CPU计算线程4、inter-op1、num_workers0。
不要使用x86_64 Conda/Rosetta。入口在导入Torch之前检查架构。

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_event_support.py
.venv-pytorch/bin/python scripts/run_m3w_european_hurdle_risk.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_hurdle_risk.py --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_hurdle_risk.py --train --resume --replay --evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_hurdle_risk.py --verify
.venv-pytorch/bin/python scripts/report_m3w_european_hurdle_risk.py --complete
.venv-pytorch/bin/python scripts/plot_m3w_european_hurdle_risk.py
.venv-pytorch/bin/python scripts/audit_m3w_hurdle_zero_reference.py
```

pilot的100次更新计入每个头总共2000次，不额外增加预算。两个目标各36个头，
总共72个、144000次更新。每200次保存模型、优化器、随机状态与采样计数。
恢复前先核查PID是否仍活跃；不要重复启动。确认原进程停止后，用同一条
`--train --resume --replay --evaluate`继续。完整且哈希一致的头会跳过。
入口文件锁防止两个同实验训练进程同时写入。磁盘不足10GiB时停止并保留产物。

缓存和日志位于本地Git忽略目录：
`data/stage_cvpr2027_experiments/european_hurdle_risk_v1/`。
检查`heartbeat.json`、`events.jsonl`及记录的PID，不把陈旧心跳当作仍在运行。
`heads/`包含模型、完整训练收据和逐行分数，不上传GitHub。

## 如何解读

- `training_losses.csv`记录总loss、平均成本MSE、BCE、正样本条件MSE和梯度。
- 两种目标的总loss不同，不直接比较总值；训练图比较同一平均成本MSE分量。
- `factor_reliability.csv`分别记录每场景总体和实际介入子集的Brier、ECE、
  正伤害率及条件程度误差。均为事后诊断，不是校准器或风险保证。
- `score_reliability.csv`记录预测与实际参考误差质量、正伤害质量。
- `results.md`保留所有144视图和全部对照，不只展示最优种子或最优场景。
- easy退化超过2%或损害零CV误差样本的视图不能因平均精度提高而算成功。
- `zero_reference_audit.json`是完成后的诊断：只有4个不同的零CV误差行，
  全部在一个场景。另6个通过组合检查的视图没有这类评估样本，不能称其
  零误差保护已经泛化。审计没有修改阈值、容忍度或模型。

本轮所有72个头、144000次更新及144视图重算完成，34个文件共218项测试通过，
并非全仓库历史测试。正easy样本的最坏退化从同结构控制的17.25%降到0.67%，
但含零CV样本的12个新分解监督视图均造成额外误差；神经模型相对同保护的
阻尼基线没有正的all-ADE置信区间。两个hard子集正信号局限于同一个划分，
其余多数反而偏向阻尼。保留局部改善，不升级部署或论文主结论。

每个检查点重放前4096个excluded-index行，不是随机抽样或整库重新推理。
另行重算全部视图指标，并复核72个旧/几何控制视图。原始数据、上游预测器、
本地缓存不在GitHub；只克隆仓库不能完成真实训练复现。

CREATE本轮只做已授权只读队列查询；未提交、修改或取消任务。模型很小且输入
在本地，用实际试跑决定运行地点，不因为队列查询成功就声称远程训练成功。
不得把simulation项目目录当作M3W环境。未执行Stage5C，未启用SMC。
