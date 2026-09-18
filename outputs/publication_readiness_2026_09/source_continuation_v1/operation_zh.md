# 完整训练子集续训：本地操作说明

## 这个实验做什么

使用原 SDD train40 中、排除 bookstore 后的 15,430 行完整静止历史样本。
三个已有模型各分出恒定学习率、余弦衰减两条续训分支。每条从第 2,000
步续到第 10,000 步，共新增 48,000 次更新。它用于诊断训练是否充分和
步长是否妨碍优化，不是新的测试集成绩，也不是整个 SDD 全量训练。

不会读取新的主任务测试、校准或确认结果，不会执行 Stage5C 或 SMC。
未来轨迹只用于训练损失和训练集误差计算，不进入模型输入。

## 环境与启动

在项目根目录操作。必须使用原生 arm64 的 `.venv-pytorch`，不要使用
x86_64 Conda/Rosetta。入口会在导入 Torch 前阻止错误架构，计算线程为 4，
interop 为 1，数据加载 worker 为 0。成功导入 Torch 不等于训练已成功；
本实验另有真实训练、checkpoint 恢复和预测重放检查。

先检查输入和父模型身份，不做训练：

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json --audit-only
```

开始或恢复训练使用同一个命令：

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json
```

运行前先确认没有该实验的存活进程，不能启动第二个写入者。私有目录为
`data/stage_cvpr2027_experiments/source_continuation_v1/`。`heartbeat.json`
包含 PID、分支、步数和时间；`checkpoints/` 保存每 200 步原子更新的状态。
`milestones/` 保存四个固定训练检查点及逐行预测，不上传 GitHub。

## 如何判断正常、恢复还是完成

- PID 存活、步数增长且定期有 checkpoint：正常运行，慢不等于卡死。
- 只有旧 heartbeat，不能判断进程已结束；需核对当前进程状态和运行日志。
- 进程确实结束但实验未完成：保留现有文件，运行同一命令续训，不删除父模型。
- 完成后重跑训练命令：应报告零新增更新，而不是重新训练。
- 哈希或身份不符：停止对应运行并诊断，不强行覆盖注册文件绕过检查。

恢复内容包括参数、优化器、抽样随机状态、Torch 随机状态和累计抽样次数。
三个父 checkpoint 是共享起点，不是六个独立从零训练的模型。

## 完成后的核查

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json --replay
.venv-pytorch/bin/python scripts/verify_analyze_m3w_source_continuation.py --registration configs/m3w_source_continuation_v1.json
```

核查 24 组固定步数预测的精确重放、三个配对抽样流、父文件不变，以及
完成后恢复不新增训练。公开结果放在本说明所在目录。重点看全训练集
ADE、发生位移样本的收益、零位移样本的绝对伤害，不只看某个 batch 的 loss。
简单样本的基线误差为零时，百分比退化没有定义，不能人为填写成零或通过。

训练种子范围不是泛化置信区间；训练集改善也不能证明 held-out 场景有效。
原始标注可能包含插值/生成点，不能据此声称实时感知、米制或秒级预测。

## 本地与 CREATE 的边界

本轮短试跑支持本地完成，数据也已在本地，因此没有重复提交远程任务。
CREATE 保存的上次访问记录是认证失败，当前远程任务状态未知。恢复访问
后应先核对项目目录、已有任务和资源限制，再通过调度器提交；本说明不
编造用户名、远程路径、分区或 job ID，也不把本地命令当作已完成 HPC 运行。
