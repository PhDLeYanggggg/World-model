# 预测器替换诊断操作说明

## 本轮做什么

本轮不训练新模型、不重调阈值。它使用上一轮已完成的18个两场景预测器、
9个四场景预测器及54个收益/风险头。每个对比在同样8个排除场景上，检查
改用两场景预测器后，轨迹误差和“何时介入”的判断分别如何变化。

三个种子、三个拟合fold、两个half全部保留。72个逐fold视图共享12个已打开
开发场景，不是72个独立实验；本轮也不会打开保留的独立选择、校准或确认组。

## 本地运行

使用已有原生arm64环境。CPU线程4、inter-op1、workers0。训练权重来自已经
核验的检查点，不能用不明权重替代。代码与模型、原始源数据及旧结果哈希绑定。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_producer_transport.py --prepare
.venv-pytorch/bin/python scripts/run_m3w_european_producer_transport.py --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_producer_transport.py --run
.venv-pytorch/bin/python scripts/run_m3w_european_producer_transport.py --verify
.venv-pytorch/bin/python scripts/report_m3w_european_producer_transport.py
.venv-pytorch/bin/python scripts/plot_m3w_european_producer_transport.py
.venv-pytorch/bin/python scripts/complete_m3w_european_producer_transport.py
```

4,096行试跑约0.755秒，完整推理计划3,827,628个“预测器-行”组合。这不是
新增这么多独立数据行，同一开发行被多个不同预测器评估。完整缓存按预测器
原子保存；重跑`--run`会校验并复用已经完成的缓存，不默默重做。中断时先
核对PID和心跳，不能看到旧状态文件就重启仍在运行的进程。

本地数据路径为`data/stage_cvpr2027_experiments/european_producer_transport_v1/`。
检查点仍在上一轮目录。私有缓存、权重和原始数据不提交Git；仅克隆代码并不
意味着已经具备这些本地依赖。磁盘低于10GiB时保留已有产物并停止新缓存写入。

## CREATE

本轮只进行了已授权的只读队列检查，未提交、修改或重启任务。检查当时有
2个运行任务、1个等待任务，这不证明M3W在远程已有结果。M3W专属远程目录
尚未核实，不能把simulation项目目录拿来作为替代。

## 如何读结果

小预测器相对四场景预测器的正差值表示该次替换有利，不证明训练样本越少越好。
替换同时改变训练场景组成、归一化和可能的训练选定基线。风险头和阈值则固定。

分开看原始轨迹误差、受保护策略误差、easy最差场景、零误差基线受损、风险头
误差，以及事后oracle机会。Oracle只参与诊断，不能作为输入或可部署结果。
每个条件区间用3,000次场景重采样；重叠窗口不能冒充独立样本。

数据是检测轨迹、图像像素、观察8步/预测12步、raw stride12。不等于历史t50，
也不是秒级、metric、人工gold、物理安全、true3D或foundation证据。
不执行Stage5C，不启用SMC，不因为完成诊断而升级部署。

## 已完成的结果

18份预测缓存已经完成，累计推理/缓存写入582.22秒；18个模型分别重放4,096行，
预测完全一致。72份原有决策和完整指标精确重算。30个针对性测试文件共201项
通过，不代表运行了整个历史测试集。原始ADE/FDE与受保护策略分别报告。

小预测器替换的36组all-ADE对比中，9组点估计改善，5组区间支持改善，15组
区间支持退化。把预测器换小并不是稳定修复。下一步重点检查并修复“候选轨迹
发生变化后，收益/伤害头不能准确反映风险”的问题，而不是从这次结果挑一个
有利half，也不会拿保留测试集反复调阈值。详细结论见conclusions.md。
