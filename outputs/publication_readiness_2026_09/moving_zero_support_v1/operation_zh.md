# 移动零参考误差案例：因果邻域与标注来源诊断

本轮不训练模型、不重新调阈值、不更换部署策略。使用已核验的 SDD 源数据、
三个种子和三类冻结预测器，分析七个移动零参考误差案例及固定元数据抽样对照。
四个场景已参与开发，不能称为独立确认。

## 运行

```sh
.venv-pytorch/bin/python scripts/audit_m3w_moving_zero_support.py --view coupa_seed17 --action transformer
.venv-pytorch/bin/python scripts/audit_m3w_moving_zero_support.py --resume
.venv-pytorch/bin/python scripts/audit_m3w_moving_zero_support.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_moving_zero_support.py tests/test_m3w_causal_motion_support.py tests/test_m3w_zero_atom.py -q
```

使用 arm64 环境、CPU4、inter-op1、单进程读取。单个真实视角试跑已完成，
计算邻域约 5.14 秒，另需源身份校验和原始标注读取。该成本适合本机完成全部
36 组，不需要占用 CREATE。最近保存的 CREATE 尝试认证失败，当前远程任务
状态未知，不重启、不取消也不推断远程任务已停止。

私有目录为 `data/stage_cvpr2027_experiments/moving_zero_support_v1/`。
`heartbeat.json`、`events.jsonl` 记录实际 PID 和每组完成时间；每组 JSON
原子保存，`--resume` 复用同身份记录。运行锁禁止同时启动两个写入者。
`--verify` 完整重算并比较不可变结果。进程健康但慢时继续等待。

## 科研边界

邻域排序只依赖源训练特征与待诊断的过去特征。未来标签只用于回顾性分组及
邻域标签构成统计，不是推理特征。普通移动对照按录像、帧和 track 元数据的
固定哈希选取，不按未来误差挑选。训练均值和标准差保持冻结，不能用查询
场景重新估计。

原始标注的 generated、未来控制点和密集帧误差仅用于标注来源审计。
它们不进入模型，不改变观察 8 步、预测 12 步、stride12 的主指标。
邻近不等于完全相同，缺少近邻零事件不等于事件不可能。
不得把邻域标签比例叫作校准概率，也不得把多个重叠窗口叫作独立事件。

DroneCrowd 确认保持关闭，IMPTC 仍隔离。仅提交代码、配置和轻量报告；
不提交输入特征、完整邻域行、模型权重、视频或第三方数据。不执行 Stage5C 或 SMC。
