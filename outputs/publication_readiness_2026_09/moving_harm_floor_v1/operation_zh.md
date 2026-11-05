# 本轮训练与恢复

这是固定的三种损失对照，不是部署命令。原 2% 风险、easy 和支持要求不变。
不得改阈值后覆盖现有结果；保留测试数据不在本轮读取范围内。

## 环境与阶段

在 World 项目根目录使用原生 arm64 的 `.venv-pytorch/bin/python`。
不要切到 x86_64 Conda。入口在数值库导入前检查架构；CPU 4 线程、
inter-op 1、单进程数据加载，未使用 NumPy 训练替代或新的 HPC 作业。

按顺序运行以下阶段；已完成阶段的报告不可覆盖，不要重复启动：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py train
.venv-pytorch/bin/python scripts/verify_m3w_harm_floor_weights.py
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_harm_floor.py
```

训练包括 9 个模型，每个 2,000 步，另有首次处理模型的完整重训和三组恢复试跑。
训练侧检查失败时，evaluate 只记录未做额外开发集评估，不会将失败当成功。
所有结果都是历史图像坐标下 obs8/pred12 的风险头实验，不是新轨迹预测器。

## 中断恢复

本地状态在 `data/stage_cvpr2027_experiments/moving_harm_floor_v1/`。
每 100 个完整优化步骤保存模型、优化器、随机数状态和采样器状态。
`heartbeat.json` 与 `events.jsonl` 记录 PID、阶段和实际进度。日志中的 PID
应再用进程状态确认；单独存在锁文件或旧 heartbeat 不证明进程存活。

连接观察超时不等于训练结束。先检查同一个会话/PID，不启动副本。
只有确认进程已结束且该阶段没有完成报告，才按原配置恢复，例如：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py train --resume
```

pilot 中断同样使用对应阶段加 `--resume`。运行锁阻止并发副本；半次优化器
更新不会替换上一次完整检查点。不能修改代码或配置后冒充原实验续跑。
正常慢运行可以继续等待，不将它降级成 quick，也不将内部拟合成功写成安全通过。

本轮首次训练在第一个完整模型之后发生 JSON 报告导出错误，退出码为 1。
修复只将报告中的采样次数数组转为列表；原注册保留，另存
`serialization_amendment.json`。恢复 PID10991 确认第一个检查点逐字节不变，
成功导出报告并继续剩余模型。详见 `attempt1_failure.md`，不要将旧 PID9913
或旧 heartbeat 当作仍在运行。

## 核查与限制

本轮已完成 9 个模型、评估与复算，科学结论为失败，不需要重复训练。
降低误放行的同时误拦了更多有益预测。以下额外诊断仅重新读取 own-B
拟合分区，已完成并逐项复算，不进行训练或读取新的评估结果：

```sh
.venv-pytorch/bin/python scripts/diagnose_m3w_harm_floor_tradeoff.py
.venv-pytorch/bin/python scripts/diagnose_m3w_harm_floor_tradeoff.py --verify
```

该诊断在看到本轮结果后提出，明确标为事后描述分析，不是预注册主检验。
初次 PID14367、复算 PID14462 已完成，不要把旧 PID 当作仍在运行。

`training_freeze.json` 核对 9 个完整模型；`full_refit.json` 核对完整重训；
`weighting_verification.json` 独立重算训练分区的权重和采样次数；
`verification.json` 记录全量指标重放。README 与 failure_analysis 解释科学结果。
这些复现检查不等于独立泛化验证。

公开仓库只保存代码、配置、汇总数字和文档，不保存原始数据、缓存或权重。
命令依赖本机已有的获准数据与历史模型，并不是下载仓库后即可无数据运行。
未执行 Stage5C，未启用 SMC，也没有 metric、seconds、true 3D 或 foundation 声明。
