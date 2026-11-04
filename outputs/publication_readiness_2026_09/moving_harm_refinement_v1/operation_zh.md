# 伤害代价修正实验操作说明

## 范围

本实验固定已有的18个神经代价头和预测轨迹，另外拟合54个小型树修正器。
不是54个新神经网络，不是轨迹残差训练，不是独立风险校准。
只使用原有64来源录像范围；23个有支持的来源组仍是已暴露开发评价。
原24组主要结果不可估，未恢复为独立测试。Stage5C、SMC保持关闭。

## 运行顺序

在项目根目录使用原生arm64环境。实验定义已登记并提交，不改动旧版本。

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_harm_refinement.py evaluate
.venv-pytorch/bin/python scripts/audit_m3w_harm_fixed_regions.py register
.venv-pytorch/bin/python scripts/audit_m3w_harm_fixed_regions.py audit
.venv-pytorch/bin/python scripts/verify_m3w_harm_cost_observer.py register
.venv-pytorch/bin/python scripts/verify_m3w_harm_cost_observer.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_harm_refinement.py
.venv-pytorch/bin/python scripts/report_m3w_harm_risk_taxonomy.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_harm_refinement.py tests/test_m3w_harm_refinement_reporting.py
```

这些是带不可变回执的研究命令，不是每次都覆盖重跑的演示。已完成阶段会拒绝
重新执行。训练中断时用同一阶段加`--resume`，不能更换配置、权重或数据哈希。
恢复会从保存的树序列重建拟合预测，并检查哈希；不是仅从最后一个计数器继续。
每一轮保存检查点并更新心跳。CPU计算4线程、interop1、DataLoader workers0。
树拟合为单进程；没有调用MPS/CUDA或声称这些后端已验证。
每个新登记都必须先提交其定义和登记文件，脚本会核对。当前完整实验的回执
已经存在，应读取它们；上述顺序是依赖说明，不是要求覆盖已完成的不可变结果。
核验阶段由 observer 入口执行，内部调用原始完整 verify，然后只增加决策后的
成本统计。不要先运行裸 verify 再补 observer，否则已有完整回执会阻止重复阶段。
原18个冻结神经头和获准数据需要先由父流程构建；私有权重、缓存和第三方数据
没有提交到Git。这是本机精确重算记录，不是他人独立完成的复现声明。

## 监控

```bash
tail -n 5 data/stage_cvpr2027_experiments/moving_harm_refinement_v1/training.log
cat data/stage_cvpr2027_experiments/moving_harm_refinement_v1/heartbeat.json
```

心跳包含实际PID、时间、阶段、轮数和产物大小。超过12小时的阶段通过原定义
恢复；健康慢任务不改成quick结果。初始磁盘保留9.5GiB，私有产物上限100MiB。
首次继承10GiB保护的启动在登记前退出，没有训练；资源设置的前瞻性修订已写入
protocol，不改变2%科学要求。磁盘继续不足时保留现有结果，不清理无关数据。
实际训练在9.5GiB保护处中断一次，保留16个完整修正器及第17个的一轮检查点；
通过`resume_m3w_harm_refinement_local.py train --resume`在9GiB操作保留线续跑。
推理/重算不写训练缓存，单独采用5GiB保留线，入口为
`resume_m3w_harm_inference_local.py`，observer 已调用该入口。训练线没有降到5GiB。
资源修订与旧中断记录分别保留；不是把2%的科学门槛放宽。

## 结果解释

主要对比是动态区域减固定区域的joint-free直接配对completion-lower差，
不是从三种方法中挑一个最好看的测试结果。全部六种策略、来源组、重复录像视图、
缺失标签和空同切换数对照都必须保留。空对照是不可估，不代表测得零风险。
记录级完整2%要求不变；汇总正收益或更少切换不能代替它。
保存的求解记录可以核验绑定和可行性，不声称限时MILP逐比特重新求解一致。
图中训练损失使用各方法自己的权重，不是相同评价分布上的性能对照。

坐标仍是数据集内pixel-space；native8步观察/12步预测不等于已验证视频帧或秒。
没有新的米、秒、true3D、foundation或可部署声明。

## 本次实际完成

54个修正器、1080棵保留主实验树均完成；中断前另有一棵未保存的树被重新计算。
9个恢复试验及3个完整重训另计132次树拟合，不混入主实验训练量。
原训练中断阶段至少477.38秒，续跑995.18秒；只报续跑时间会低估成本。
完整评估1543.02秒，带逐录像成本统计的重算370.62秒，均观察到退出码0。
另有310.13秒同选择集合拟合诊断。阶段有部分重叠，不能直接相加当端到端墙钟时间。

全部72个留出成本输出、原策略指标和缓存控制均精确一致。288个同区域拟合
统计单元保留，72个拟合预测哈希及18块旧指标一致；72个留出成本诊断单元
在决策完成后读取标签，确认成本预测和分配哈希没有变化。

结果仍不部署。已知标签主对比约+0.0705个百分点，但配对保守指标约+0.0027，
95%区间[-0.1412,+0.1087]跨零。最差easy净退化1.37%通过净退化条件；
仍有42个all正伤害、72个easy正伤害视图超2%，完整风险仅47/348视图有支持。
214个easy所选事件分母不可估，其中161个整段无路径变化，另53个实际有动作。
这些分开报告，不把所有不可估项都称为实际伤害，也不把它们算经验风险通过。
48/138个同切换数对照为空，仍不可估。保留原24组主要结果不可估和23组已暴露
开发结果的限制。完整旧pytest套件未运行，因为包含会改写旧研究报告的集成流程；
本轮单独运行与改动相关的定向测试，不声称整个项目全部测试通过。
