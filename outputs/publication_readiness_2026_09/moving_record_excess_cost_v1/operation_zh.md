# 聚合风险误差训练与恢复

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_record_excess_cost.py verify
```

已完成阶段不会覆盖。中断的train使用同一命令加`--resume`；不得改配置、数据、
模型或目标后冒充同一恢复。每100步保存模型、优化器、采样器和随机状态。私有
heartbeat包含PID、当前模型、步数、loss及耗时；`process.lock`防止本任务重复启动。
用实际进程与持续推进的日志判断是否仍在运行，不把旧heartbeat或观察超时当作卡死。

原始首次试跑日志保留。日志字段修复及从step0恢复的原因见runtime_patch_note.md。
实际六组50/100恢复与直接训练一致，三个point模型与原训练循环一致后才启动正式训练。
正式18模型每个2000步；原point控制是18个已核验模型，不伪称新训练。另执行第一个
point完整重训与第一个新模型完整重训核对。所有模型冻结后才计算新H预测。

需要本地已授权的Moving数据、冻结预测器、成本模型和分配缓存；Git不包含这些大文件。
arm64原生环境、CPU4线程、interop1、workers0，无资源探测或多进程DataLoader。
保留5GiB磁盘和256MiB本任务缓存上限，单阶段12小时界限。慢但健康的任务继续运行。
本次模型和数据适合本机；没有新建或重启CREATE任务，旧远程状态不冒充实时查询。

未来标签只用于训练loss和评估。教师选择、录像分组只参与训练loss，不增加推理输入。
C教师是同C监督拟合，不是来源OOF验证或独立校准。保留所有空对照与无支持视图，
原2%风险合同不变。只训练神经成本头，不训练轨迹生成器，不执行Stage5C或SMC。
