# 固定回退基线的直接风险目标：运行与复现

## 证据范围

本轮新增108个神经风险头、216,000次更新。轨迹预测器没有重训，108个普通MSE
对照及保护性阻尼基线来自已验证的上一轮。固定12个开发来源，未打开独立确认数据。
观察8步、预测12步、raw-frame stride12；不是旧Stage37的t50成绩。

## 环境与路径

工作目录 `/Users/yangyue/Downloads/World`，使用原生arm64的
`.venv-pytorch/bin/python`。不要用旧x86_64 Conda。CPU4、interop1、workers0，
不做资源探测或DataLoader多进程。每500步保存检查点及随机状态，慢不等于卡死。
本轮全部108个头已在240.02秒内完成，峰值RSS9.71GB，PID30724正常退出。
本地适合该规模；CREATE只做了已批准的只读队列检查，没有提交或修改远程任务。

私有数据、模型和日志：
`data/stage_cvpr2027_experiments/european_fixed_floor_excess_v1/`。
公开的配置、报告和轻量指标在本文件同目录；不上传检查点、原始轨迹或图像。

## 恢复已有训练

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_excess.py --phase train --resume
```

完成的头校验后跳过。恢复必须保持配置、来源、监督目标和随机状态不变。
不能覆盖已封存实验来做新模型选择；新假设应使用新命名空间和新注册。
第一轮100步试跑已经计入2,000步，不额外增加训练预算。

## 复现核验

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_excess.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_excess.py --phase causal_replay
.venv-pytorch/bin/python scripts/replay_m3w_fixed_floor_excess_training.py
.venv-pytorch/bin/python scripts/run_m3w_european_fixed_floor_excess.py --phase replay_evaluate
.venv-pytorch/bin/python scripts/report_m3w_fixed_floor_excess.py
.venv-pytorch/bin/python scripts/verify_m3w_fixed_floor_excess.py
```

预测回放检查全部108个新头和对应的对照、决策、训练误差诊断。因果回放删除未来
字段，运行第一组真实模型。首头从初始化完整重训，核对参数、优化器、随机数、
抽样次数与loss轨迹；该重训只是验证，不是新增候选。评分与报告也要求精确回放。

验证器独立重建“同来源、同录像、同帧”的人数匹配，不允许借用未来帧分配名额。
所有汇总保留固定来源集合，零介入导致的无定义风险不能偷偷删掉。局部自动测试
和精确回放证明实现范围内的可重复性，不证明独立泛化或物理安全。

## 解读结果

直接监督的是“正伤害减去2%参考误差”的组合；两个输出分量不再分别具有可校准
误差的含义。loss下降不能证明风险预算满足。需要一起看all/hard/easy、正伤害、
同介入率对照、空覆盖、来源/种子、完整/部分标签和未知标签介入数量。

全部结果仍是image-local、检测器silver、raw-frame，不能写成米、秒、人工gold、
true3D、foundation或安全认证。Stage5C和SMC继续关闭。独立测试和正式投稿未执行。
