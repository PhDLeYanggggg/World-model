# 配对损失实验：运行、恢复与结果核验

这轮实验比较同一个因果代价预测头的两种损失，只替换简单样本正伤害的一个
损失项。它不是重新训练轨迹预测器，也不是已经证明有效的世界动力学模型。

## 当前证据

真实 TRAIN 试跑完成；完整实验共144个固定训练任务，每个2,000步，包含
72组配对身份和三个 head seeds。独立校准和确认集保持关闭。已经通过核验的
36个结果不重跑，其余任务在 CREATE 续跑。

历史精确复现检查有一个失败。原始训练器在当前节点重新执行后，与新实现、
从头训练及断点恢复的结果逐位一致；历史保存的权重和优化器状态略有不同。
本次明确登记了一个身份的参照修订，不能把它写成72个历史检查点全部复现。
最终应分别报告71个历史精确对照和1个原始训练器精确重放对照。

## 环境和任务

本地使用原生 arm64 `.venv-pytorch`，不使用 Rosetta/x86_64 Conda。CREATE
使用已有的独立 M3W CPU 环境和调度器分配的计算节点；不在登录节点训练。
计算线程4，interop线程1，DataLoader workers为0。检查点每100步保存。

原始数组37814169的任务1已经完成。续跑数组37815216只包含0、2、3号分片；
37815222负责联合验收。每个训练分片仍保留12小时上限，没有减少样本或更新数。

```sh
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance status --phase train
```

观察超时不等于任务失败，不要直接重新提交。有既存 submission intent 或
已排队任务时先查同一个作业。中断时保留最后一个原子检查点，并核对当前作业、
登记哈希和配置后再恢复。不能把历史 heartbeat 当作新作业的实时进度。

## 完整训练验收

只有所有剩余任务和联合验收完成0:0后才能收集完整结果：

```sh
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance collect --phase train
```

收集的是轻量元数据，不下载权重缓存。核对全部144个结果、更新数、检查点哈希、
三种子、精确对照类别、数据角色和联合任务记录；提交训练冻结文件后才能评价。
仅有36个结果、完成日志片段或合成测试通过，都不满足完整验收。

## 固定评价

评价代码已经登记，并通过针对性算术、缺失支持和准入检查；当前真实评价仍为
`not_run`。完整训练冻结并提交后使用：

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout preflight
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout run
```

中断后，先核对已写结果和锁状态，再用同一版本显式恢复：

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_easy_harm_readout run --resume
```

权重从自有 CREATE 路径读取到内存，先校验哈希再反序列化，不建立本地权重副本。
对照包括原始森林、additive、positive-harm、cost、quadratic和easy-deviance。
保留全部负结果，按录像与帧匹配介入数量。未知标签留在推理人口中，不能删除。

场景内先汇总不同 head seeds 和视图，再对12个 source localities做3,000次
配对 bootstrap。它们是已暴露开发数据上的名义区间，不是独立确认。Head seeds
也不能冒充新轨迹预测器的独立训练种子。

## 边界

2%是“被介入简单样本的正伤害/参考代价”预算，与全体简单样本的净误差退化
不是同一指标。正收益不能掩盖风险失败或无标签支持。即使开发筛查通过，也只
允许设计后续迁移实验，不直接部署或打开独立确认集。

数据仍是图像局部像素坐标、raw annotation frames和detector-silver标签。
不声称米、秒、true3D、foundation或物理安全。Stage5C和SMC不执行。

原始数据、特征、权重和大缓存不进Git；保留10GiB磁盘预留、父子实验合计
256MiB检查点上限和2MiB原子写入余量。不要修改其他项目或环境。
