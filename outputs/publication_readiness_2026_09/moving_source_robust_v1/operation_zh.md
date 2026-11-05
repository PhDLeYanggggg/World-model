# 来源稳健神经风险训练：运行与恢复

本轮只修改训练目标，不修改原 2% 风险、easy、参考与支持要求，不加入恒等
回退认证。对照组继续使用原损失；实验组使用 75% 原损失加 25% 最差训练
来源损失。它是介入风险头训练，不是新轨迹动力学模型或部署命令。

## 环境

在 World 根目录使用原生 arm64 `.venv-pytorch/bin/python`，不用 x86_64
Conda/Rosetta。CPU4、inter-op1、workers0，不进行 Torch 资源探测。
来源编号仅在训练时分组损失，不是推理特征，也没有按来源调整门槛。

每个模型从已经冻结的 relational 神经风险头继续训练 1000 步。两组使用
相同初始化、归一化、样本顺序和参数量；优化器也对称重置。六个模型全部
训练完才判断预先固定的拟合检查是否允许下一次已曝光 nested-C 开发评价。
未通过时评价为 not_run，不修改检查标准，不把拟合改善写成安全通过。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_source_robust.py
```

注册文件必须先提交，才能执行真实训练；已完成阶段不能覆盖。以上完整
运行命令用于理解依赖关系，不应对已有完成收据的阶段重复执行。
整个实验完成后复核现有证据使用只读入口：

```sh
.venv-pytorch/bin/python scripts/replay_m3w_moving_source_robust.py
```

## 恢复

私有目录为 `data/stage_cvpr2027_experiments/moving_source_robust_v1/`。
`heartbeat.json` 保存 PID、当前组和进度，`events.jsonl` 保存事件记录。
检查点每 100 个完整更新保存模型、优化器、RNG 和采样器。
先检查原 PID 和日志；一次观察超时不代表进程退出。确认原进程已退出且
尚无完成收据后，使用原配置恢复：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_source_robust.py train --resume
```

文件锁阻止重复任务，半步异常不会覆盖上一次完整检查点。保留 5GiB
磁盘空间，私有产物上限 512MiB。本轮资源适合本地，无需重复 CREATE 作业。
真实数据 pilot 比较两种损失各自的 50+50 恢复与连续 100 步；完整训练
还重复一次 1000 步拟合检查确定性。复现一致不等于独立科学验证。

## 证据边界

- 新训练、来源损失和错误归因记 fresh_run。
- 旧权重、原始因果数据、schema 和 own-B 拆分按哈希复核后记 cached_verified。
- 未执行的独立确认记 not_run；不读取保留的 Moving20、DUT、DroneCrowd 或 H。
- 未来标签只用于训练监督和评价，缺失标签保留原上下界，不伪造已观测标签。
- 三个拟合上下文共享一个预测器种子，不是三个独立训练种子。
- 原 24 组主评价仍不可估；旧失败不被本轮工程检查覆盖。
- obs8/pred12 为图像坐标，不声称米、秒、true 3D、foundation 或 human gold。
- Stage5C/SMC 关闭。Git 只保存代码、配置、报告与轻量指标。
