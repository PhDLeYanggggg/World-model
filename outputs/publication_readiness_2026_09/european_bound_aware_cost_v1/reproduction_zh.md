# 本轮实验复现与结果核查

## 这轮实际做了什么

比较同一个成本预测网络的两种监督方式：只监督已有未来标签，或另外
监督完全缺失标签行的保守误差贡献。不是重新训练 EqMotion，也不是新训练
JEPA、Transformer 或轨迹生成器。36 个真实 Torch 网络已完成；本说明及论文
更新读取并核对这些已完成实验，不重复计为一次新训练。

使用 149,510 个 SOURCE 窗口、74 段录像、12 个地点，三个真实训练种子。
对每个模型，预测器拟合地点、成本头拟合地点和评价地点均不重叠。SOURCE
已用于研究开发，所以这种排除仍不是最终独立测试。

## 环境与执行

在项目根目录使用原生 arm64 `.venv-pytorch/bin/python`，不要使用 x86 Conda。
现有运行配置为计算线程 4、inter-op 线程 1、数据加载 worker 0。真实完整
拟合约 379 秒、峰值约 9.24 GB，本轮无需新增 CREATE 作业。不能把导入 Torch
成功当训练验证；`pilot` 中有实际更新、保存、恢复和直接重跑的状态比较。

首次完整执行顺序如下；它们要求本地已有经过校验的上游数据和预测器产物：

```sh
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py train
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py infer
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py eval
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py verify
```

若训练被中断，使用原配置恢复，不另起一个相同训练：

```sh
.venv-pytorch/bin/python scripts/run_m3w_bound_aware_cost.py train --resume
```

私有目录 `data/stage_cvpr2027_experiments/european_bound_aware_cost_v1/` 中保存
日志、heartbeat、检查点与数组；其中数据不进入 Git。先核对日志/PID/实际
进程状态，不把旧 heartbeat 或锁文件当作进程仍在运行。完成阶段的收据是
不可覆盖的；不要删除既有结果来制造一次“fresh_run”。本轮所有阶段已结束，
不能再次运行完整命令并把拒绝覆盖误判为训练卡死。

已有公开标量结果可独立重建文字报告和图表，不需要重新读取原始标签：

```sh
.venv-pytorch/bin/python scripts/report_m3w_bound_aware_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_bound_aware_cost.py tests/test_m3w_bound_aware_design.py tests/test_m3w_supported_bound_targets.py
```

图表重新生成后应重新核对 SVG 哈希和实际渲染，不沿用旧版视觉检查。
CREATE 环境和调度器检查见项目的 `local_create_runbook_zh.md`。若后续扩大
规模再使用 CREATE，应先核查现有作业，不因连接失败推断任务已结束或重复提交。
本轮没有远程任务状态的新证据，不声称完成 HPC 训练。

## 应得到的结果

- 36 个完整拟合，116,160 次主更新；4,000 次完整重拟合与 1,200 次恢复试跑
  单独计数。两种监督的初始权重、标准化和查询抽样相同。
- 36 组完整成本/幅度数组及 72 组指标精确复现。
- 主要配对保守差异为 -1.7031 个百分点，95% CI [-5.7081, +1.1166]，未通过。
- 已知标签的配对差异为 +0.3774 个百分点，CI [-1.4814, +1.9475]，同样不确定。
- 完整风险通过视图由 28/444 增至 108/444，但原定要求仍失败，不可部署。

这里的 444 是相同录像经不同模型/种子形成的重复视图，不是 444 个独立地点。
Bootstrap 重采样单位是 12 个地点，先平均排除视图及种子，再进行 3,000 次抽样。
不能把两种策略各自的保守下界相减来替换上述直接配对主要指标。

## 还没有完成什么

公开轻量产物能重建统计报告，但完整匿名冷启动复现还依赖尚未打包的合法数据
准备与模型产物。最终独立确认、可靠的场景联合介入方法与完整风险通过仍未完成。
这不是“程序已跑完，所以论文目标完成”。标签是 detector/tracker silver，
坐标是图像像素，8 步观察/12 步预测的步长为 raw stride12；不可称米、秒、
人工金标准或物理安全。Stage5C 与 SMC 均未执行。
