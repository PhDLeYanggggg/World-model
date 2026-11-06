# 共享专家上下文参考分支实验

本实验继续遵守原逐录像 2% 风险与参考支持规则，不增加恒等回退认证。
它只使用已暴露 MovingDroneCrowd++ own-B 和已验证的来源隔离训练 donors，
观察8步、请求预测12步，图像坐标；不是旧 SDD t50，也不是米或秒的结果。

冻结上一轮成本模型、轨迹预测器、伤害与标签支持输出。新分支使用同一 agent
冻结神经预测的因果描述，不是未来真实轨迹。两个同容量模型只比较屏蔽/接入
这份共享上下文。参考量按 agent 计算一次，再广播给该 agent 的候选动作。

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py infer
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py replay
```

执行前需已核验的私有父级数据缓存和权重。公开 Git 不包含这些数据，下载代码
不等于已复现输入。源代码、配置和协议注册提交后才执行真实准备和训练。
入口拒绝 x86_64/Rosetta；CPU4/inter-op1/workers0，不探测 MPS 资源。

真实数据先比较128+128恢复与连续256步，随后48次参考模型训练、48次完整精确
复训；每模型4096步、每128步保存、每256步记录心跳。健康运行不因慢缩为 quick。
私有目录 `data/stage_cvpr2027_experiments/moving_expert_reference_v1/` 保存心跳、
事件、缓存和权重。磁盘保留5GiB，新私有产物上限2GiB，每阶段12小时。

中断后使用原定义恢复，不删缓存或重新开始：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py verify --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_expert_reference.py infer --resume
```

先冻结所有18组新策略选择，再运行评价。冻结旧模型的指标必须逐项重现。
主比较固定为 expert_context 减 history_refit 的 joint-free 配对保守增益，
按8个来源分量进行3000次 bootstrap，不把重叠窗口或72个重复视图当独立场景。
原24来源组主结果继续 NA。replay 重算预测和指标、验证冻结选择，不重新拟合，
也不重新运行带时间限制的整数搜索。运行 replay 时不要改写公开实验文件。

未来有效性只用于监督/评价，不进入输入或 query 筛选；不用 future endpoint、
central velocity、test goals 或 test 统计归一化。不读取新 C/H、reserved 数据；
不执行 Stage5C，不启用 SMC。平均预测改善不自动等于部署安全或投稿就绪。
