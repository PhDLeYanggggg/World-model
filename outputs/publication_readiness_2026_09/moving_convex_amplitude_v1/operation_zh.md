# 凸幅度代价头操作说明

## 范围

本实验不重新训练轨迹骨干，也不训练普通 residual。它在同一七幅度候选
上约束 H、EH 及非 easy 伤害的曲线形状。原来的 18 个幅度头是核验后的
缓存控制；新增 18 个约束头，共 36,000 次正式更新，不称为 36 个新训练。
试跑与两个首任务完整重训另计。数据角色、参考分母和原 2% 要求不变。

原生 arm64 `.venv-pytorch`，CPU4、interop1、workers0，不做 Torch 资源
探测。真实三折试跑用时 58.45 秒，峰值内存约 9.31 GB；恢复结果及原
控制采样流一致。纯拟合投影约 1,173.10 秒，不是完整流程完成时间。
本地资源足够，未启动新 HPC 作业，也不据此声称核验了远程队列状态。

## 顺序命令

在仓库根目录执行。定义及注册记录须先提交；已完成阶段不会被覆盖。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_convex_amplitude.py
```

最后的报告命令只在训练、完整评价及重算结束后运行。所有 18 个新头和
完整重训核验完成前，不评价新留出策略。公开代码不包含第三方数据或
权重；需要原父实验的数据、预测、18 个控制权重及求解器缓存。

## 监控和恢复

本地日志与 checkpoint 位于
`data/stage_cvpr2027_experiments/moving_convex_amplitude_v1/`。
检查 `heartbeat.json`、`events.jsonl`、`training.log`，并核对 PID 是否
存活与步数是否推进。只有锁文件或旧 heartbeat 不能证明正在运行。
每 100 更新写 checkpoint 和 heartbeat；日志包含 loss、分项 loss、
梯度范数、有效标签与未知边界监督行数、采样覆盖。

若原进程确已终止而阶段未完成，先确认环境及注册哈希未变，再恢复：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_convex_amplitude.py train --resume
```

未完成的 pilot/evaluate 也接受 `--resume`。完整阶段是不可变结果，不要
删除回执强行重跑。10 GiB 磁盘保留量、100 MiB 私人产物上限或 12 小时
阶段边界触发时保留 checkpoint，修复硬问题后恢复原预算，不因正常慢跑
自动缩小数据或更新数。

## 解释边界

相同原始参数、容量、输入、loss 和采样预算，不表示初始中间幅度输出
完全相同；这正是解码器改变的内容。线性 hinge 使用稳定双精度运算，
训练末端仍转回原 FP32 loss，推理保留曲线精度。此数值设计已披露，
不是一个单独精度消融，不能把任何差异全部归因于“凸性”一个因素。

必须同时读固定主对照、缺失结果下界、所有录像风险、easy/hard、
尾部、终点诊断、介入幅度和空的同数量对照。空对照不可估，不是安全零；
形状正确也不代表代价大小准确。求解器超时保留预测可行的独立父策略，
不是实际安全证明。重算不重新优化 MILP，不称为逐位求解器重演。

全部结果仍是已暴露开发来源的辅助证据，不是独立最终测试。原 24 组
主要结果不可估，不能改称通过。像素与 native annotation step 不等于
米、秒或 SDD raw t50。Stage5C execution 与 SMC 始终关闭。

## 本次实际完成

18 个新头与 36,000 次正式更新全部完成，训练用时 1,627.62 秒；两个
首任务的完整重训一致。完整留出评价 841.28 秒，独立重算 180.76 秒，
均观察到退出码 0。全部 36 个新旧成本输出与策略指标一致，保留全部
138 个来源/模型/种子视图及 348 个重复录像视图。没有因慢缩减。

结构检查通过，但科研要求未通过。相对匹配原控制，已知标签效应约
-0.02 个百分点，区间跨零；配对缺失结果下界约 -1.83 个百分点，区间
为 [-3.62,-0.42]。完整风险支持仍为 10/348，最差已知 easy 净退化
428.91%。新头不部署，也没有证明新的轨迹动力学提升。

报告中的三个图已经目视核查。`per_source.md` 保留所有 23 个支持来源，
不只选改善切片。代码检查、真实训练完成、重算一致与科研门槛通过是
不同结论；当前只完成前三类，不能写成独立泛化或投稿候选成功。
