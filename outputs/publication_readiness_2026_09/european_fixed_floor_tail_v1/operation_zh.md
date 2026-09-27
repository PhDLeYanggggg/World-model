# 本轮训练与复现说明

## 结果状态

本轮真正训练了216个神经风险头，共432,000次更新。没有重训轨迹预测器或
JEPA。既有预测器、保护性阻尼基线、utility头和预处理经过哈希校验后复用。
结果属于12个已用于开发的来源，不是最终独立测试。训练完成不等于风险门通过。

主对照失败：高伤害加权策略主要降低介入率；相同帧、相同介入人数下的ADE优势
区间跨零。3个留出视图完全回退，预注册的伤害比值主指标因此无定义，不能写成0。
95个相关留出视图仍超过2%预算。部署不变，Stage5C与SMC继续关闭。

## 本地环境

工作目录为 `/Users/yangyue/Downloads/World`。必须使用本机arm64环境
`.venv-pytorch/bin/python`，不能使用历史x86_64 Conda。入口沿用架构检查，
CPU计算线程4、interop线程1；不启用DataLoader多进程，不执行资源探测。
本轮真实训练411.32秒、峰值RSS9.51GB，无卡死。小型风险头适合本机，未提交HPC任务。

## 核验现有产物

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_tail.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_tail.py --phase causal_replay
.venv-pytorch/bin/python scripts/replay_m3w_fixed_floor_tail_training.py
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_tail.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_fixed_floor_tail.py
.venv-pytorch/bin/python scripts/diagnose_m3w_fixed_floor_tail.py
.venv-pytorch/bin/python scripts/verify_m3w_fixed_floor_tail.py
```

完整预测回放检查216个头及108组决策。因果回放移除未来标签字段再运行第一组真实
预测。首对模型从初始化训练到2,000步，对比参数、优化器、RNG、抽样次数和loss轨迹；
它是复现检查，不计入新候选。整个评分回放必须逐字一致。

验证器还独立重建同帧人数匹配，并独立计算固定12来源的汇总和bootstrap；
有无定义来源时不得删掉该来源重算成功结果。图表与报告也要求可重复。

## 中断恢复

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_tail.py --phase train --resume
```

每500步保存原子检查点，包含优化器、采样器和随机状态。已完成的头校验后跳过，
不重复训练；第一轮100步试跑包含在2,000步预算内。`heartbeat.json`和
`events.jsonl`记录私有目录内的进度，判断完成仍以进程退出和检查点验证为准。
本次训练PID28294已正常退出。若需新实验，使用新命名空间和注册文件，不能覆盖封存结果。

## 文件和安全边界

代码、配置、汇总报告和轻量指标提交GitHub；不提交私有检查点、特征缓存、原始轨迹、
图像视频或第三方数据。公共目录为本文件所在目录。私有产物在
`data/stage_cvpr2027_experiments/european_fixed_floor_tail_v1/`。
保留10GiB磁盘空间，不删除无关文件，也不改动其他暂存中的研究数据。

CREATE连接和资源限制沿用simulation model项目已核准的交接规范。本轮无远程训练
或作业修改；队列可见不等于模型已在HPC上跑过。遇到连接暂不可用先继续本地独立工作。

本轮是image-local坐标、观察8步/预测12步、raw-frame stride12、检测器silver标签。
不是旧Stage37的t50成绩，不是米、秒、人工gold、物理安全保证、true3D或foundation。
完整旧测试套件和从原始数据冷重建均未运行，不能以局部复现替代这些证据。
