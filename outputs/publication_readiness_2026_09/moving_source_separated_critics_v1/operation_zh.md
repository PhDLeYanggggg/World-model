# 本轮训练与恢复

本轮遵从“保持原规则，先改模型”。不增加恒等回退认证，不放宽2%阈值，
不删除无支持录像，不重写旧失败。当前使用 MovingDroneCrowd++ SOURCE，
观察8步、预测12步原始标注序列，不是已验证的秒级或米制任务。

真正的修改是神经成本头的训练数据隔离。沿用原有 C 内层三折，每一折
单独计算自己的特征归一化、监督目标尺度和初始偏置，再训练两个种子。
同一来源换种子与真正不同来源的风险头分开比较。H 不参与本轮训练或调参。

使用本机 arm64 环境，CPU4线程，单进程，不启动新的远程作业。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_source_separated_critics.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_source_separated_critics.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_source_separated_critics.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_source_separated_critics.py verify
.venv-pytorch/bin/python scripts/report_m3w_source_separated_critics.py
```

这些是首次运行命令。完整阶段完成后不会覆盖旧结果。日常重新复核请使用：

```sh
.venv-pytorch/bin/python scripts/replay_m3w_source_separated_critics.py
```

这个入口重新推理全部216个成本数组，并核对108组策略和1620个成本统计。
它只更新私有运行心跳和日志，不修改现有公开实验文件，不训练，不读取新H
结果。缺失产物或数值不一致会报错，不会自动“修正”旧结果。需要本机保留
的合法数据与检查点；公开Git仓库没有原始数据和权重，不是免数据的演示。

每100次更新保存检查点。心跳、事件记录和模型权重位于 Git 忽略的
`data/stage_cvpr2027_experiments/moving_source_separated_critics_v1/`。
一次只运行一个阶段；运行中不要重复启动。若进程确实中断且退出后，使用
同一条命令增加 `--resume` 恢复，不改变预算、配置或模型绑定。
已完整完成的阶段不会覆盖。程序保留5GiB磁盘空间，单阶段最长12小时。

完整训练包括108个新成本头、216000次主训练更新；试跑和首次完整重训
检查另外统计，不能充作主训练规模。完整训练全部冻结后才读取新的内层
验证结果。慢但健康就继续等待；试跑的预计时间不是已完成时间。

风险合约仍检查每个完整策略的所有必要录像视图。空的同数量效用对照
仍不可估，预测等于基线也不自动把缺失支持变成通过。不论平均值多好，
没有满足原合约就不部署，不进入 Stage5C，不启用 SMC。
