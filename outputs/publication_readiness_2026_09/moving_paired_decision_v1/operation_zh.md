# 成对决策实验：运行与恢复

本实验保持原 2% 风险、easy、参考与支持要求；不是部署命令。
不允许通过改阈值、读取保留测试数据或恒等回退认证来改变结论。

## 环境

在 World 根目录使用 `.venv-pytorch/bin/python`，原生 arm64。
不要使用 x86_64 Conda/Rosetta。入口在数值库导入前检查架构。
CPU4线程、inter-op1、workers0，不进行 torch 资源探测。
试跑实际训练三个模型并比较中断恢复与连续训练，不能用 import 成功替代。
本地数据已具备，历史和本轮试跑成本适合本机，不提交新的 CREATE 作业。
读取过的旧 CREATE 记录不代表当前队列状态。

## 固定阶段

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_paired_decision.py
```

已经完成的阶段保持不可覆盖；不要重复启动。拟合侧原检查未过时，不做
新的开发集读数，报告 not_run。检查通过也不等于科学安全门槛通过。
代码、配置、协议和注册在训练前固化；报告脚本只读取已验证的完整结果。

## 恢复

本地检查点、heartbeat 和事件记录在
`data/stage_cvpr2027_experiments/moving_paired_decision_v1/`。
每100个完整优化步骤保存模型、优化器、RNG、采样器和配对步数。
先检查原 PID/会话；连接观察超时不意味着进程结束，不可启动副本。
确认退出且训练阶段没有完成收据后，原配置恢复：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_paired_decision.py train --resume
```

pilot 恢复使用对应阶段加 `--resume`。进程锁阻止并发副本；半步更新不写入
检查点。磁盘保留5GiB，私有实验产物上限512MiB。正常慢运行不降级为 quick。

## 数据和科学限制

训练使用 own-B，开发评价使用已经曝光的 nested-C；不是独立确认。
未来标签仅参与监督和评估，不进入预测输入；unknown/零目标不作有益配对。
所有权重与预处理保留原划分。公开 Git 不保存原始数据、缓存、权重或图片。
本机获准历史资产是复现前提，克隆代码不等于自动取得数据。
原24组主评价仍不可估。obs8/pred12 使用图像坐标，不声称米、秒、true3D、
foundation 或人工 gold；Stage5C 和 SMC 保持关闭。
