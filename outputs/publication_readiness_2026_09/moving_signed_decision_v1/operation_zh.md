# 本轮训练、恢复与结果核查

## 本轮改变什么

只改变风险头的训练损失。对照仍用 MSE；新模型按实际风险代价学习
应该放行还是拒绝。两个模型共用输入、初始化、训练数据、采样顺序和
2,000 次更新预算。不是重新训练轨迹预测器，也不是继续 JEPA。

保持原规则：2% 风险门槛、easy 净退化要求、参考样本支持要求全部不变。
缺少分母仍不可估，不引入恒等回退认证，不把空效用对照判为通过。

## 环境

在项目根目录使用原生 arm64 环境，不用 x86_64 Conda：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_signed_decision.py
```

配置固定为 CPU 4 线程、interop 1、单进程、workers 0；入口拒绝 Rosetta。
每 100 次更新保存检查点和心跳。有进展的训练可继续运行；最长 12 小时，
磁盘保留至少 5 GiB。私有产物上限 512 MiB。不要另起一个同目录训练进程。

## 中断后恢复

只对尚未完成的阶段使用 `--resume`，例如：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py train --resume
```

检查点保留模型、优化器、随机状态、采样状态、更新次数和绑定哈希。
代码、配置和输入必须与登记版本一致。完整阶段拒绝覆盖；不要删除旧结果
再当成新实验运行。进程暂时没有打印不等于卡死，先查心跳、PID 和检查点。

私有检查点和运行记录位于
`data/stage_cvpr2027_experiments/moving_signed_decision_v1/`；
轻量报告位于本目录。数据、缓存、检查点和环境不上传 GitHub。

## 怎样判断是否成功

1. `pilot.json` 检查两种损失的恢复结果是否逐点相同。
2. `training_freeze.json` 检查六个真实模型是否完成、是否在评价前冻结。
3. `full_refit.json` 检查新损失模型的完整重复训练是否相同。
4. `verification.json` 检查三组预测、动作、指标是否复现，哈希是否匹配。
5. `analysis.json` 和 `README.md` 检查收益、伤害、缺失支持和最差场景。
6. `full_fitting_objectives.json` 只解释训练行为，不能代替泛化结果。

新的分数不是校准风险概率。训练损失降低、能恢复、程序退出成功，都不等于
原安全门槛通过。缺失标签补全区间不是 bootstrap 置信区间。
七个已暴露 SOURCE 单元、一个预测器种子不是独立确认。

本轮使用观察 8 步、预测 12 步的原生标注步长及图像像素。不能替换成历史
SDD 的 raw-frame t+50，也不能写成秒或米。保留域和 H 不打开。
Stage5C 不执行，SMC 不启用。
