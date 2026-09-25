# 本轮结果与复现说明

## 做了什么

本轮没有重新训练轨迹 Transformer 或 JEPA，而是保持此前 18 个预测器和
预测增益头冻结，新训练了 18 个岭回归风险头、18 个真实 PyTorch 风险头。
36,000 次神经更新全部完成，三个种子、三个源场景折、两种风险目标以及
有无支持不足回退的 24 组策略全部评估。不是只做试跑，也没有挑最好种子。

任务保持观察 8 步、预测 12 步、原始帧间隔 12，数据来自 European Squares
已开放的 12 个训练地点。不能与历史 SDD/external raw t50 数字混为一谈。
独立选择、校准、确认数据及 DroneCrowd 确认部分没有打开。

## 得到了什么

新的 easy 专用神经风险头在完整逐 agent 评估中，相对恒速 CV 的 ADE 改善
为 0.2382%、0.1663%、0.4310%，对应三个条件性地点 bootstrap 区间均为正。
最差地点 easy 退化分别为 0.0225%、0%、0.8170%，四个已观察到的 CV 零误差
样本均未受损。这里是“观察到的保护改善”，不是已经获得安全保证。

代价很明显：上一轮相对 CV 还有约 4.18% 到 4.43% 的改善；现在介入率只有
0.29% 到 1.25%，大部分增益消失。与固定阻尼 0.97 比较仍落后约 3.92% 到
4.20%。固定阻尼自身也没有通过最差场景 easy 安全要求，因此不能直接替代。

对照分析显示，easy 专用目标比同结构的普通风险目标保留更多增益，但普通
岭回归头也能得到 0.48% 到 0.61% 的点式增益。这还不能证明神经风险头具有
不可替代的贡献。联合决策中两个种子的最差 easy 退化超过 2%；相同介入数
下的联合优势仍未建立。全部负结果见 [完整表格](results.md)。

四个 CV 零误差样本都在一个地点，每个仅有 12 步中的 2 步标签，缺少最终
端点。联合试验小样本中没有它们。样本少或样本缺失不能写成“安全已验证”。
历史 Stage37 分数在后续来源检查中发现重复和选择暴露，仍是探索性历史，
不能用其高分替代当前独立证据。

结论：风险目标有部分效果，但当前不升级部署，也未达到论文投稿候选证据
要求。不是 true 3D、foundation、米或秒级预测、人工 gold 或物理安全证明。
Stage5C 和 SMC 均未执行。

## 如何复核

工作目录：`/Users/yangyue/Downloads/World`。
使用 `.venv-pytorch/bin/python` 原生 arm64 环境，不能用 x86_64 Conda。
计算线程 4、inter-op 1、DataLoader workers 0。以下命令复核本轮完整结果：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_conditional_risk.py --verify --replay
.venv-pytorch/bin/python scripts/report_m3w_european_conditional_risk.py
.venv-pytorch/bin/python scripts/audit_m3w_event_risk_numerics.py
.venv-pytorch/bin/python scripts/plot_m3w_european_conditional_risk.py
```

第一步校验冻结身份并重新计算全部指标，再对每个 checkpoint 抽取 4,096 行
重新推理。当前 36 个头均完全一致；这不是全部重新训练，也不是新数据验证。
报告脚本进一步核对旧预测、训练抽样和联合控制的真实介入数量。

首次运行该版本所用训练命令为：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_conditional_risk.py --prepare --pilot
.venv-pytorch/bin/python scripts/run_m3w_european_conditional_risk.py --train --resume --evaluate
```

100 次更新的试跑包含在正式预算中。中断时用第二条从模型、优化器及随机状态
恢复；完整运行过的头只核验，不冒充再次新训练。代码或数据身份不一致必须
停下检查，不能覆盖旧 checkpoint。修改目标、阈值或科学设计需新建版本。

## 日志与文件

- 公共报告：`outputs/publication_readiness_2026_09/european_conditional_risk_v1/`。
- 私有训练文件：`data/stage_cvpr2027_experiments/european_conditional_risk_v1/`。
- 私有 `events.jsonl` 和 `heartbeat.json` 保存进程、阶段和时间。
- `analysis.json` 是所有策略、地点、尾部与支持情况的汇总，不是原始轨迹。
- `training_losses.md` 保存真实 minibatch loss，不能用低 loss 代替预测提升。
- `verification.json`、`checkpoint_replay.json`、`accounting_audit.json` 分别
  对应完整指标复算、checkpoint 推理、抽样与介入数量核对。
- `numerical_audit.json` 记录极小分母造成的十个求解失败视图及定点修复。

数值修复不放松风险预算、不读取 future，也没有修改原版科学结果。修复后
十个实例仍选择相同 CV 回退，因此不能把它说成新增预测增益。

训练和以上核验已结束。本轮适合本地，未提交 CREATE 任务，也未改变 simulation
项目的环境或作业。后续大规模任务仍按其已给定的目录和调度限制执行。

下一步优先做同等保护下简单运动模型与神经预测器的严格对照，再诊断风险分母
可靠性和 teacher 训练场景数量不一致。常规审计与运行不再交回用户逐项审批；
不因此放宽数据、未来泄露、部署或正式投稿边界。
