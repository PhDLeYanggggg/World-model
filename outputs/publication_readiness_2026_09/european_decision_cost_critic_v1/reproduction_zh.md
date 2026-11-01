# 本轮复现与结果核查

本轮确实训练了18个神经成本模型，不是重新训练轨迹动力学模型。
使用原生arm64 `.venv-pytorch/bin/python`，CPU4线程、interop1、workers0。
实际loss、模型、优化器和随机数状态保存在忽略目录；公开的是标量训练记录和汇总指标。

## 公开汇总复核

```bash
.venv-pytorch/bin/python scripts/report_m3w_decision_cost_critic.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_decision_cost_evidence.py
```

报告读取校验过的公开汇总包，不需要私有轨迹或权重。绘图仅在本轮忽略目录写一个PNG预览。
这不是从原始数据重新训练。测试中专门禁止读取项目data和external_data后复核报告。

## 有授权素材时复核固定策略

```bash
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py verify
```

需现有原始授权素材、冻结缓存和成本模型权重。此命令重新推理并重算已保存策略的完整分数，
不重新求解有墙钟限制的MILP。检查本次退出码和日志，不能只拿旧verification.json判断本次成功。
首次完整评价PID79489，独立复核PID82655。全部72组原始对照与最终分数一致。

## 本轮原始执行顺序

下面是历史执行记录，不是覆盖已经冻结结果的按钮。首次运行应在隔离工作区准备相同上游素材。
通用匿名从零训练接口尚未完成，不声称完整公开复现已经交付。

```bash
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py register
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py pilot
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py train
mkdir -p data/stage_cvpr2027_experiments/european_decision_cost_critic_v1/allocation_progress
mkdir -p data/stage_cvpr2027_experiments/european_decision_cost_critic_v1/progress
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py eval
.venv-pytorch/bin/python scripts/run_m3w_decision_cost_critic.py verify
```

原始首次评价因目录缺失退出，日志保留；只创建本轮两个输出目录后用eval --resume恢复。
训练无须重跑，注册脚本、loss、阈值、模型和评价规则均未修改。训练断点每100步保存，心跳每50步。
确认进程已终止后才能用train --resume或eval --resume，避免重复任务；不要根据文件存在猜测任务状态。
12小时是单次运行上限，不因正常训练慢提前降低目标。求解器2秒预算不等于删除困难样本，失败保留可行父策略。

## 软回退能力诊断

```bash
.venv-pytorch/bin/python scripts/run_m3w_decision_support_isolated.py
```

它使用单独心跳目录，可与完整评价并行，不改训练或策略。重复执行会核对已有冻结结果，不覆盖不同结果。
444视图的软约束下界为正，说明这版平滑训练形式表达不了足够保守的回退；不证明硬策略无解。

## HPC与证据边界

本轮本机峰值约11GB，运行成本合理，因此没有提交CREATE重复作业。
最新只读队列连接被远端关闭，当前队列未知，不把连接失败写成历史作业失败。
未来确需HPC时，先按已有授权连接复核队列和指定作业，再提交调度任务，保存job ID、环境、日志和断点。
迁移后Torch/GPU环境尚未在本轮验证，不凭旧退出码认定新环境可用。

全部结果仍是已暴露SOURCE拟合，不是独立测试、米/秒、人工gold、true3D或foundation。
本轮5/7拟合检查不等于投稿准备完成。部署不变，Stage5C与SMC保持关闭。
