# 联合决策损失实验：运行与恢复

## 本轮改动

只改变神经风险头的一个附加训练损失。旧损失分别学习 all/easy 通道；
新损失学习两个头、两个通道都通过的联合决定。原风险回归、未知标签的
保守监督、2% 风险与 easy-net 门槛、支持要求都不变。没有恒等回退认证。

这不是新的轨迹生成器，也不是新的 JEPA 训练。输入仍是过去历史、邻居、
冻结的候选轨迹等因果特征；监督标签和未来有效性不能进入预测接口。
当前任务是原生 obs8/pred12 图像坐标，不是 t+50，也没有秒或米的声明。

## 环境

在 `/Users/yangyue/Downloads/World` 使用原生 arm64 环境：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_joint_decision.py
```

已经完成的 phase 不应重新启动；其完成记录不可覆盖。登记文件、代码和
协议在训练前已提交。默认 CPU 4 线程、interop 1、DataLoader workers 0，
不做运行时资源探测。不要换回 x86_64 Conda。每 100 个完整更新保存一次。
本轮无需提交新的 CREATE 作业，避免重复训练。

## 监测与恢复

本地私有目录：
`data/stage_cvpr2027_experiments/moving_joint_decision_v1/`。
`heartbeat.json` 记录 PID、当前分区、loss_arm、step、耗时和损失；过程日志
及各 fits 检查点也保存在该目录。先核对 PID 是否仍存活，再考虑恢复。
耗时长但更新正常不是卡死，也不意味着应该缩减实验。

```sh
cat data/stage_cvpr2027_experiments/moving_joint_decision_v1/heartbeat.json
pgrep -fl run_m3w_moving_joint_decision
.venv-pytorch/bin/python scripts/run_m3w_moving_joint_decision.py train --resume
```

仅在进程已经退出、该阶段尚无完成记录时运行恢复命令。恢复会检查数据、
标签、提案、配置和登记哈希，接续模型、优化器、采样器与 RNG；不会保存
更新了一半的参数。不得更改配置后继续旧检查点。保留 5GiB 磁盘余量，
本轮私有文件上限 512MiB。超过资源限制要报告硬错误，不得伪称完成。

## 如何判断完成

- `training_freeze.json`：六个实际模型及固定更新预算完成，不代表安全。
- `fitting_screen.json`：原来的内部训练诊断；失败则不增加 held 读出。
- `analysis.json`：明确 fresh_run 或 held_readout_not_run。
- `verification.json`：训练侧指标、归因和已执行读出的精确重放。
- `gates.md`：区分工程完成、原安全要求、独立科学证据。

仅已暴露的 SOURCE/nested-C 开发角色可用。reserved Moving、DUT、
DroneCrowd 与 H 不打开。原 24-group 主指标仍可能因支持不足为 NA，
不能把空对照或没有参考事件写成通过。所有旧失败保留。

Git 只保存代码、配置、聚合指标和报告，不上传原始数据、缓存、模型权重、
图像或第三方文件。Stage5C 与 SMC 均不执行。论文准备完成仍须独立证据。
