# 固定候选轨迹的容量诊断：复现与恢复

本实验是对已暴露 Moving A 开发集的事后诊断，不训练或部署模型。
使用原生图像坐标，观察 8 个标注步、预测 12 个标注步。真实未来只进入离线
求解与评分。保留原 2% 规则、参考样本支持和不可估判定。

## 本机环境

在项目根目录使用已有的原生 arm64 `.venv-pytorch`，不要用 Rosetta 下的
x86_64 Conda。先核对所需资产的登记及哈希；没有本地预测库或源数据时，
报告缺失，不替换为随机预测。该程序用 NumPy 和 SciPy，不使用数据加载
子进程或 GPU；对这个规模，已验证的本机路径就足够。

```sh
.venv-pytorch/bin/python -c 'import platform, scipy; print(platform.machine(), scipy.__version__)'
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_prefix_capacity.py
.venv-pytorch/bin/python scripts/run_m3w_prefix_capacity.py score
.venv-pytorch/bin/python scripts/run_m3w_prefix_capacity.py replay
.venv-pytorch/bin/python scripts/plot_m3w_prefix_capacity.py
```

`score` 检查登记的代码与输入，逐录像写小型检查点。已经完成的录像无需
重算。中断后先确认对应 PID 已结束，再执行同一 `score` 命令恢复。
不要删除锁、启动重复进程、或把静止的日志文件当作进程已死。
心跳在 `data/stage_cvpr2027_experiments/moving_record_joint_actions_v1/causal_ledger/prefix_capacity/heartbeat.json`；
其中记录 PID、阶段、种子、已完成视图数和正在求解的约束模式。
早期长尾经系统采样定位为求解器预处理，续跑保留已有检查点，未改科学目标。

`replay` 只重新汇总已登记的检查点，并比对固定结果，不是第二次求解。
检查结果有 102 个视图、14 个来源，固定切换策略合计 6,952 次；
求解超时或未通过原评分器复核的项保留为未证实。不要把只含可解视图
的平均值写成全数据成绩。图在本机论文附录目录，不含样本或预测轨迹。

## 何时使用 CREATE

这项诊断没有提交 CREATE 作业。若后续独立数据推理需要远程计算，
先用已授权连接在登录节点只读查询任务与资源，确认没有同一运行；
仅在自己的 M3W 路径提交经验证的计算节点脚本。连接失败时远端状态未知，
不据此重提、取消或抢用 simulation 项目的资源。已有连接与安全边界见
[CREATE 交接记录](../../../create_readonly_handoff_20260923.md)。
外部角色、候选与策略来源仍须单独冻结；本教程不会开启外部结果。
