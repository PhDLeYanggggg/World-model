# 净风险目标实验：复现与恢复

## 实验边界

本实验重新训练风险预测头，不重新训练Transformer、EqMotion或运动预测器。
协议是观察8步、预测12个原生标注步，SDD标注像素坐标；不能和历史raw-frame t+50结果混用。
四个源站点已经参与开发，本轮站点排除结果和3000次站点Bootstrap不是独立确认。

原始训练程序保存于`run_m3w_net_easy_moment.py`，仅使用其preflight/train功能。
新策略决策和评估必须使用`run_m3w_net_easy_moment_guarded.py`。
原始程序的决策/评估分支存在已知保护缺口，保留仅用于不可变来源记录，不应调用。

## 训练与恢复

在项目根目录使用原生arm64环境，CPU计算线程4、interop1、DataLoader worker0。

```bash
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment.py --phase preflight
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment.py --phase train --resume
```

每16棵树保存一次检查点。第一次全新训练也可使用`--resume`；已有检查点不会被重置。
完整预算是3个预测器、4个排除站点、3个种子，共36个128棵树的模型。
`--stop-at 16 --view coupa_seed17 --action damped_velocity_005`仅用于真实试跑，不能当完整结果。

训练心跳、PID和事件在：
`data/stage_cvpr2027_experiments/net_easy_moment_v1/heartbeat.json`及`events.jsonl`。
`trials/<站点_seed种子>/<预测器>/complete.json`存在且校验通过才算该模型完成。

## 决策、评估与验证

所有模型完成后，先冻结全部past-only决策，再读取已有监督标签计算表现：

```bash
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment_guarded.py --phase preflight
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment_guarded.py --phase decide
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment_guarded.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment_guarded.py --phase decide --verify
.venv-pytorch/bin/python scripts/run_m3w_net_easy_moment_guarded.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/verify_m3w_net_easy_moment.py
```

决策每256个场景查询保存；中断后重新运行会验证并复用已完成分块。
重放模式要求原始产物已存在，不允许首次运行自称重放。
脚本会校验原始来源、训练身份、对应action/seed、输出哈希；不一致时停止，不静默覆盖。

正向风险约束不抵扣改善；净风险允许同一录像/帧的预测改善抵扣伤害，约束含义较弱。
两者都固定2%预算比例，不能把求解器满足预测预算解释成真实安全。
个体伤害、零基准误差样本伤害、最差站点/种子easy退化、缺失标签都要单独阅读。
风险监督使用完整未来标签；主要ADE允许已有有效未来点，这一支持差异没有被校准解决。

## 数据与发布

Git仅保存代码、配置、聚合指标、报告。训练缓存、模型、逐行决策和第三方数据保留本地。
DroneCrowd确认数据保持关闭，DUT结果仍是已暴露诊断；HT21/CroHD尚未准入预测评估。
任何本轮有利结果都不自动改变部署模型。Stage5C执行、SMC和独立安全声明仍关闭。
