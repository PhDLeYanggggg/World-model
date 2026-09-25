# 本轮实验的运行与证据边界

这轮不重新训练轨迹预测器，而是重新训练“是否值得采用神经预测”的收益和
伤害控制头。三组输入分别是不提供预测器标识、提供真实标识、提供与标签无关
的随机式标识。每个主要比较使用相同的候选轨迹和回退轨迹。

## 本机运行

使用项目里的 arm64 `.venv-pytorch/bin/python`。本轮计算线程4、跨算子线程1、
DataLoader workers0。历史问题来自 x86_64 Conda/Intel OpenMP，不应重新使用
该环境。不能只凭 import 成功认定训练正常；本轮先完成真实100步反向传播，
再从检查点继续到注册的2000步。

完整顺序和命令见 `execution_notes.md`。训练中每200步保存检查点和心跳。
中断后使用 `--phase train --resume`；评估中断后使用
`--phase evaluate --resume`，已完成的分组会校验后复用。不要对完整目录重新
跑 pilot。两个阶段不得同时启动，进程锁会阻止并发覆盖。

本地私有产物在：
`data/stage_cvpr2027_experiments/european_producer_conditioned_v1/`。
其中 heartbeat.json 记录当前进程和步骤，events.jsonl 保留阶段时间，
heads 保存检查点与loss，decisions 保存读标签前冻结的决策，evaluation 保存
逐组评价。GitHub 只保存代码、配置、测试、聚合指标和文本报告。

## 怎样读结果

先看 producer 对 global/placebo 的同预测器比较，再看对旧四源 stop 策略的
比较。若前者改善而后者退步，不能说当前系统整体升级。若只有改错标识后变差，
只能说明模型使用了标识，不能说明真实标识稳定有效。

training_losses.svg 展示实际训练loss。utility 每次记录的是不同训练batch；
risk 另有固定训练batch曲线。它们不是验证集loss，更不是独立测试成功。
所有种子、事件目标和两个预测器分支必须一起看，不挑最好的一组。

2%的预测风险预算不是2%的真实安全保证。正伤害比例和净easy误差增加不是
同一个量。分母为零、缺失未来标签和完整轨迹支持必须分别保留，不能记成零误差。

## CREATE 与复现

本轮只执行已授权的只读队列检查，不提交作业、不修改远程环境。获准使用的
M3W远程结果目录仍未定位，不能据此声称服务器没有产物。当前计算在本机合理，
无需为等待远程目录而停工。以后需要CREATE时仍遵守 simulation model 给出的
访问范围，通过调度器运行，不在登录节点训练。

公开聚合结果可检查表格、哈希、loss和实验状态，但从零重建预测仍需要本地
授权原始数据及父实验检查点。公开Git仓库不是原始数据的替代品。

这些是已开放场景上的开发性实验。独立选模、风险校准和确认集继续封闭。
当前不是true 3D，也不是foundation；像素坐标和raw annotation stride不改写成
米或秒，自动检测轨迹不称human gold。Stage5C不执行，SMC不开启。
