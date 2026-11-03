# 新来源神经预测对照：运行与结果边界

## 做什么，不做什么

本实验在已接纳的 MovingDroneCrowd++ 训练来源上真正拟合 PCT 和固定头
EqMotion，不是 NumPy fallback，也不是旧 SDD 分数重述。两模型族、三来源
折、三个真实种子，合计 18 个模型，每个完整 4,000 次更新，共 72,000 次
主训练更新。用于后续同一预测器的增益／伤害学习，不因预测器跑通就宣称
选择策略安全或世界模型成功。DUT、DroneCrowd 的预测结果不在本轮读取。

采用原生 annotation step 下观察 8、请求 12，不是已经验证的秒级时域。
坐标是 head-box center 像素，不是米制、true3D 或 foundation world model。
未来标签和有效性 mask 只进入 loss／评价，不进入模型输入。

## 空支持问题怎样保留

64 个接纳录像全部处理，但其中 6 个没有可用 8/12 query，实际 58 个录像
贡献数据。24 个来源组中，scene2 对应组的三个短片只有 17／10／16 帧，
整个组没有请求窗口。原入口因此在训练前退出，原代码、日志、完整 RAM
数据统计仍保留。前向修复不删任何一行、不改折、不缩短预测任务。

195,286 个历史完整目标行仍全部保留：136,295 完整未来、54,275 部分未来、
4,716 全未知未来。原 24 组完整主对比记为不可估，不能补成零误差；23 个
有预测支持的组只计算明确标注的二级结果。来源组不是已证实 IID 地点，
重叠窗口也不是独立样本。

两个预测器采用相同的最近 8 个过去历史完整邻居规则。原始当前可见输入
先完整构建，部分过去历史者没有因未来标签被移除；另外冻结的 48 维全体
已观测运动特征包含这些人，但本轮不喂给预测器，也不宣称它已带来贡献。
后续 critic 实验才比较它的增量价值；它不是物理相机位姿。

## 环境与命令

在 `/Users/yangyue/Downloads/World` 使用 arm64 `.venv-pytorch`，不要使用
x86_64 Conda／Rosetta。入口在导入 Torch 前拒绝错误架构。本轮显式使用
CPU 4 计算线程、1 interop、0 DataLoader workers，无 multiprocessing 和
Torch 资源探测。现已通过两个模型族的真实 50+50／直接 100 步一致性试跑，
比较模型、optimizer、RNG、采样覆盖和实际输出，不只是 import 成功。

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_source_forecast_supported.py pilot
.venv-pytorch/bin/python -u scripts/run_m3w_moving_source_forecast_supported.py train
.venv-pytorch/bin/python -u scripts/run_m3w_moving_source_forecast_supported.py evaluate
.venv-pytorch/bin/python -u scripts/run_m3w_moving_source_forecast_supported.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_source_forecast.py
```

这些命令有阶段顺序，不能把尚未生成的评价／验证结果写成已完成。已完成
pilot／训练阶段会拒绝覆盖，不必为复现删除它们。`evaluate` 在 18 个完整
训练模型之前拒绝读出留出结果。

## 监控与中断恢复

私有任务目录：
`data/stage_cvpr2027_experiments/moving_source_forecast_supported_v1/`。
检查 `heartbeat.json`、`events.jsonl`、`training.log` 和真实 PID。每 50 次
更新写 heartbeat，每 200 次原子保存 checkpoint，完成模型另写 complete
收据。第一主模型可复用其完全同身份的 100 步试跑前缀，预算仍是 4,000。

若旧进程已确认结束且训练尚未完成：

```sh
.venv-pytorch/bin/python -u scripts/run_m3w_moving_source_forecast_supported.py train --resume
```

恢复会检查模型身份、训练行、loss 因子、optimizer／随机状态，跳过已完成
的同身份模型，续跑未完成者。观察超时不等于进程退出；先检查同一 PID，
不要重复启动。阶段边界异常需要先核对已完成的收据，不要删除有效工作。
12 小时是单次边界，不是允许缩小实验的理由。此前不足 200 步的未落盘
更新可能需重算，应保留中断事实，不假称所有实际计算均被恢复。

完整 551,487,664 字节特征矩阵仅在 RAM，重建约十几秒，避免写巨型缓存。
checkpoint 和压缩预测留在本地，不入 Git。保留原 10 GiB 磁盘余量；真实
资源不足时先保留产物，再核对已授权 CREATE 的队列／自己目录后转移，
不改另一个 simulation 项目的任务或环境。本轮本地成本适合，无新 HPC 作业。

18 个正式模型完成后，为保留完整重训核验的磁盘余量，4 份已结束的 100 步
试跑副本移入本地无损归档。所有原始字节和路径信息都保留，48,838,756 字节
压缩为 14,842,420 字节；每个成员的 SHA 与原件一致，正式 18 个模型未改变。
原始试跑单文件路径需还原后再直接打开，完整 4k 重训并不依赖这些副本。
完整收据见同目录 `storage_compaction.md` 和 `pilot_storage_archive.json`。

```sh
.venv-pytorch/bin/python scripts/archive_m3w_moving_pilot.py verify
.venv-pytorch/bin/python scripts/archive_m3w_moving_pilot.py restore
```

还原要求保留原 10 GiB 余量，且不覆盖有变更的现存文件。无损存储校验不是
重新训练或新的科学结果；归档及模型大文件仍不入 Git。

## 核查结果

需要同时看到所有主训练完成、原始留出推理完成、两个首模型完整重训一致、
全部保存预测独立重算一致，才能把控制实验记为完成。loss 曲线是拟合目标，
不是留出 ADE／安全。旧 heartbeat 字段 `mean_past_normalized_ADE` 沿用了
原工具命名；本数据在 loss 前已恢复像素单位，该原始 ADE 字段不能冒称米
或无量纲误差，真正训练目标另有按拟合来源基线归一化的 loss。

分组 bootstrap 为 3,000 次，先平均实际训练种子和排除来源 producer，再
按支持组重采样。正的 23 组二级对照不能使原 24 组主结论通过；也不证明
全 2% easy／operational risk、scene/goal/interaction 贡献或独立确认。
原风险阈值、负结果、Stage5C-off 和 SMC-off 限制保持。
