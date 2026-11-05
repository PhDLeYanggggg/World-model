# 候选预测与历史运动关系：运行与恢复

本实验保持原 2% 风险、easy、参考与支持要求，不加入恒等回退认证。
训练的是神经介入风险模型，不是新轨迹生成器，也不是部署命令。

## 环境与运行

在 World 根目录使用原生 arm64 `.venv-pytorch/bin/python`，不要使用
x86_64 Conda/Rosetta。CPU4、inter-op1、workers0，无资源探测。
真实试跑已比较两个版本各自的50+50与连续100步，并复核旧对照。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_rollout_history.py
```

已完成阶段不能覆盖或重复启动。训练完成并通过预先固定的拟合检查，才允许
读取已经曝光的开发集；若失败，开发评价标 not_run。所有六个模型均保存，
不能根据开发结果删除失败版本。拟合检查不是科学安全门槛。

整个实验完成后，使用只读复现入口，保留原结果和执行收据：

```sh
.venv-pytorch/bin/python scripts/replay_m3w_moving_rollout_history.py
```

## 恢复与资源

私有检查点、heartbeat、事件日志位于
`data/stage_cvpr2027_experiments/moving_rollout_history_v1/`。
每100个完整更新保存模型、优化器、RNG和采样器。先检查原 PID；观察超时
不代表进程退出。确认原进程退出且尚无完成收据后，用原配置恢复：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_rollout_history.py train --resume
```

文件锁阻止重复任务，半步更新不会覆盖原检查点。保留5GiB磁盘空间，私有
产物上限512MiB。本轮本地数据和资源足够，不提交 CREATE 重复作业。

## 证据边界

新训练与诊断记 fresh_run；旧权重、划分、原始特征按哈希复核后记
cached_verified；未运行的独立确认记 not_run。新32维特征只从过去轨迹和
已有因果预测计算，不读取真实未来。未来标签仅作训练监督与评价。
own-B 用于训练，已曝光 nested-C 用于开发，不是独立测试。
原24组主评价仍不可估；保留 Moving20、DUT、DroneCrowd、H 不开启。
obs8/pred12 图像坐标不解释为米或秒，不声称 true3D、foundation 或人工gold。
Stage5C/SMC 保持关闭。Git只保存代码、配置、报告和轻量指标。
