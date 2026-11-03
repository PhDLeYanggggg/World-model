# 标签可用性分解实验：复现说明

## 完成情况

这轮确实训练了36个 Torch 成本预测头，不是重新读旧指标，也不是新训练
轨迹预测器。使用当前原生 arm64`.venv-pytorch`、计算线程4、inter-op1、
DataLoader worker0。真实恢复试跑1200次更新，正式116160次更新，另外4000次
完整重拟合。两个新模型的结构、初始化、标准化和抽样相同。

新概率表示“释放数据中是否有至少一个可用的未来标签”，不是物理可见性。
未来标签只监督loss和评价，不能输入模型。未观测行的已知分母贡献为零，
不表示其真实未来误差为零。完整原理、既有负结果和条件加权限制见
`method_scope.md`；并没有证明新的风险控制定理。

## 执行顺序

从项目根目录执行；需要已有经过核验的SOURCE数据、EqMotion和直接成本头产物。

```sh
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py register
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py pilot
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py train
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py infer
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py eval
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py verify
```

中断后先查日志、PID和实际进程，不重复启动同一任务。使用原定义恢复，例如：

```sh
.venv-pytorch/bin/python scripts/run_m3w_observation_cost.py train --resume
```

完整阶段禁止覆盖。不要删除已经完成的检查点或收据来制造一次“新运行”。
私有日志、heartbeat、检查点与数组在
`data/stage_cvpr2027_experiments/european_observation_cost_v1/`，不进入Git。
保留10GiB磁盘余量、1GiB本实验私有空间上限及12小时阶段边界；如遇真实
资源错误，保留产物后按原预算恢复，不把慢改称quick完成。

本轮本机内存峰值约9.23GB、训练404.41秒，无需新建CREATE任务。远程作业
状态未重新核验，不据此声称“CREATE没有任务”。后续需用HPC时按项目
`local_create_runbook_zh.md`先核查现有作业、环境和资源，再通过调度器提交。

公开标量报告和图表可以直接重建，不重新打开原始标签：

```sh
.venv-pytorch/bin/python scripts/report_m3w_observation_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_observation_cost.py tests/test_m3w_cost_mass_decomposition.py tests/test_m3w_bound_aware_cost.py tests/test_m3w_bound_aware_design.py tests/test_m3w_supported_bound_targets.py
```

## 结果核查

- 14项针对性测试通过。六组真实恢复、两个首次完整重拟合精确一致。
- 36组成本/概率/幅度预测及72组指标完整复算一致，旧直接成本策略也精确复现。
- 新模型相对直接成本头的已知标签收益下降2.564786个百分点，95%CI
  [-3.740926,-1.480043]；主要保守差异为-3.133282个百分点，CI
  [-4.410254,-1.987128]。这是失败，不是因保守下界不确定而误报的失败。
- 风险通过视图增至160/444，但最差easy净退化仍为8.319583%，不可部署。
- 标签可用性AUROC改善是另一个目标的次要诊断，不能替换轨迹与完整风险结果。

CI以12个地点重采样3000次，先平均排除视图和真实训练种子，不把重叠窗口
当独立样本。SOURCE已经参与开发，不是最终独立测试。公开材料能复算统计
报告，但完整匿名冷启动复现仍需补齐合法数据与模型准备说明。

仍为像素/raw-frame/silver结果，不是米、秒、human gold、物理安全、true3D
或foundation。没有部署升级，没有执行Stage5C，没有启用SMC。
