# 当前预测器损失对照：复现与恢复

## 范围

这轮直接比较当前无量纲、按智能体历史组织的预测器。另一个
`european_reference_harm_forecast_v1` 是早期模型的匹配对照，不能混成同一结果。
所有数据仍是已打开的 SOURCE 开发数据，不是独立确认集。主任务是观察8步、
预测12步，间隔为12个原始帧。没有秒、米、真实3D或foundation声明。

## 环境

工作目录：`/Users/yangyue/Downloads/World`。
使用原生arm64的 `.venv-pytorch/bin/python`，不用x86_64 Conda。
训练固定CPU4计算线程、1个interop线程，DataLoader workers为0。
输入、权重、预测缓存都不提交Git。至少保留10GiB磁盘空间。

## 已验证的运行检查

当前模型的真实数据80步工程试跑完成，10+10与20步连续训练精确一致；
关闭新损失项时，与原训练代码精确一致。随后完整重跑4,000步中性对照，
模型、优化器、随机状态、训练行及抽样次数均与旧checkpoint精确一致。
这验证实现可比，不是科学效果或安全通过。

## 首次运行

```bash
cd /Users/yangyue/Downloads/World
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_harm_forecast.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_harm_forecast.py --pilot --neutral --train
```

代码、配置和协议须先提交。不要删除已有checkpoint重跑，也不要改变原协议
后覆盖同名实验。中断时恢复同一预算：

```bash
.venv-pytorch/bin/python scripts/run_m3w_dimensionless_harm_forecast.py --neutral --train --resume
```

每200步原子保存checkpoint，每50步更新本地heartbeat和events。
真实数据在本地；无需提交新的CREATE训练任务。训练结果以完整9个固定终点
的 `training_freeze.json` 为准，不以进程存在、日志有loss或单个checkpoint为准。

## 冻结与评价

先将 `training_freeze.json` 提交Git，再运行：

```bash
.venv-pytorch/bin/python scripts/report_m3w_dimensionless_harm_forecast.py
.venv-pytorch/bin/python scripts/report_m3w_dimensionless_harm_forecast.py --verify
.venv-pytorch/bin/python scripts/analyze_m3w_forecast_support.py
.venv-pytorch/bin/python scripts/analyze_m3w_forecast_support.py --verify
```

最后两个命令只读流式获取CREATE上已有、带hash的旧SOURCE TRAIN数据包，
不提交、重启或取消HPC作业。若连接失败，保留本地结果，记录外部数据包未核验，
不能把访问失败说成任务结束或重复提交任务。

支持诊断先重现旧148个录像/上下文的标签感知oracle，再替换新候选轨迹。
旧CV/damping.97floor、旧资格mask、录像集合和2%预算不变。oracle不是模型预测。
新预测不能直接沿用旧的候选相关安全分数；仍需新训练和独立风险校准。

## 结果解释

`readout.md` 和 `analysis.json` 报告三种子、各场景及配对bootstrap。
`record_diagnostics.csv` 保留每个录像，包括缺标签和零参考误差情况。
bootstrap按locality抽样，不把重叠窗口当独立样本；这些区间仍属开发性证据。
净误差改善不能抵消正伤害。空对照风险记不可估，而非安全通过。

日志中的 `loss` 为训练目标；`native_loss` 是有源尺度权重的ADE项，
`selected_positive_harm`、`cv_positive_harm` 为新增惩罚。
继承键名 `mean_past_normalized_ADE` 实际记录未加权坐标ADE，不能按名称误读单位。
不同损失的总loss数值不直接比较，科学结论来自匹配的预测误差和风险读数。

Stage5C和SMC始终关闭；本轮不会自动部署新模型。
