# 幅度风险头实验操作说明

本实验学习介入幅度对应的收益与伤害，不重新训练轨迹骨干，不训练普通
residual，不执行 Stage5C，也不启用 SMC。它仍是开发来源上的对照实验，
不是独立最终测试。旧的完整安全门槛和全部失败录像都保留。

## 环境与规模

在仓库根目录使用原生 arm64 `.venv-pytorch/bin/python`。入口在导入 Torch
前拒绝 Darwin x86_64，使用 CPU4、interop1、workers0，不做资源探测。
已有真实六格试跑通过，峰值内存约 9.16 GB，不是仅验证 import。

两组均为 28,229 参数、428 个输入字段。36 个正式风险头各训练 2,000
更新，共 72,000 新增正式更新；两个首任务的完整重跑和试跑另计。
相同网络、候选输入、旧物理归一化、初始化、采样和预算下，仅改变
可监督、可部署的幅度集合。另有同权重的 binary deployment 对照。

完整试跑用时约 80.33 秒，六格断点恢复与直接训练完全一致。纯拟合
投影约 2,335.68 秒，不是完整训练已完成；准备数据、完整重跑、推理与
求解器另计。正常慢跑不会自动缩小样本或更新预算。

## 依次运行

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py train
.venv-pytorch/bin/python scripts/audit_m3w_moving_amplitude_fit.py evaluate
.venv-pytorch/bin/python scripts/audit_m3w_moving_amplitude_fit.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_amplitude_cost.py
```

已完成阶段的回执是不可覆盖的；不要删除旧结果来伪造 fresh run。
`register` 验证原数据和定义。新增定义必须先提交，再运行训练。
正式训练全部完成后才评价留出策略。拟合侧检查先于新留出评价，比较
固定动作区域上的成本可靠性，不能用于挑 checkpoint 或调阈值。

## 日志与恢复

私人运行目录为
`data/stage_cvpr2027_experiments/moving_amplitude_cost_v1/`。
检查 `heartbeat.json`、`events.jsonl` 和 `training.log`；同时核对进程
PID 是否存在、日志步数是否推进，不能只凭锁文件判断仍在运行。
每 100 更新保存原子 checkpoint 和 heartbeat。进程结束后先读退出码和
错误日志；不要因一次观察超时而重复启动任务。

中断后，确认旧进程终止、环境和注册哈希未变，再恢复同一任务：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_amplitude_cost.py train --resume
```

未完成的试跑或策略评价同样支持 `--resume`。完整的已完成任务会核验并
复用，不因恢复请求重新计数。10 GiB 磁盘保留量、100 MiB 私人产物上限
或 12 小时阶段边界触发时保留已有 checkpoint，诊断后恢复原预算。

## 怎样读结果

`training_freeze.json` 证明 36 个任务完成及配对一致，不证明模型有效。
`verification.json` 证明新成本输出和策略指标重算一致，不证明独立泛化。
旧 binary 结果是 cached_verified，不是本轮重新推理。

两组 active-action loss 的监督集合不同，不能只比较数值大小。
主对照是同一来源、种子和物理基线下的幅度头减 binary 头配对结果。
必须同时看完整录像风险、easy、未知标签、同数量空对照、介入率及
幅度分布。相同介入数量不代表相同改动幅度；空对照不可估，不是安全零。

原来的 15 个 empty-event 视图不会因新网络自动可估。不得删除它们，
也不得把剩余可估子集改称原完整主要结果。未来标签只用于监督和评价；
easy 始终由原始 CV 定义，R/ER 分母不随幅度缩小。

当前单位是 pixel / native annotation step；不是米或秒，也不是 SDD
raw-frame t50。即使平均收益为正，完整门槛失败也不升级部署。权重、
特征、缓存和原始数据留在本地，不提交 GitHub。未使用的新来源角色
仍待既有科学划分问题确认，DUT 和 DroneCrowd 保留用途不变。

## 本次实际运行

36 个正式风险头和 72,000 新增正式更新全部完成，两个首任务完整重训
一致；训练约 3,025.52 秒，峰值内存约 8.71 GB。拟合侧 144 个检查及
独立重算用时约 209.62 / 208.74 秒。完整留出评价约 910.43 秒，全部
138 个来源/模型/种子视图保留；独立输出重算约 199.65 秒并观察到退出
码 0。没有因为慢而缩小预算。评价阶段的完成回执、结束日志与进程
终止已核验，但原会话句柄在工具输出截断中丢失，不补造该阶段的退出码。

最终主对照的已知标签效应 +0.46 个百分点，区间跨零；配对缺失结果
下界 -1.16 个百分点，区间也跨零。幅度头只在 10/348 个重复录像视图
满足原完整风险要求，故不升级部署。同权重动作集合对照虽有小幅已知
标签正信号，也不能替代主对照与安全要求。完整数字、loss 曲线、
easy/hard/尾部/终点诊断见 `readout.md`，原因分析见 `failure_analysis.md`。
这不是独立最终测试，也没有新增轨迹骨干、JEPA 或生成式训练。
