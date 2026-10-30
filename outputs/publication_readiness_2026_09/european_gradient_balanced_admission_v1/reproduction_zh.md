# 梯度修复实验：本地复核与恢复

## 先看结论

这不是新的轨迹预测器训练，而是在固定候选预测上训练成本/增益/伤害头。
本轮真正新训练18个头，36个旧对照经过hash及共同设计校验后复用，不能称为54个新模型。
72组完整评价和独立进程重放已完成；主对比为负，完整风险不通过，不部署。
这些是已暴露SOURCE拟合证据，不是独立测试或整条流水线的OOF结果。

## 环境与证据位置

工作目录：`/Users/yangyue/Downloads/World`。
使用本目录的arm64 `.venv-pytorch/bin/python`，不要用x86_64 Conda/Rosetta训练。
入口在导入Torch前检查架构；CPU线程4、interop1、DataLoader workers0。
不需要MPS资源探测；本轮实测训练约3.5分钟、峰值内存10.92GB，本地足够。
因此没有为本轮提交、取消或复制CREATE任务。旧远程状态不代替当前调度器核验。

公开结果在 `outputs/publication_readiness_2026_09/european_gradient_balanced_admission_v1/`。
私有权重、事件日志、heartbeat、恢复锁和训练状态在
`data/stage_cvpr2027_experiments/european_gradient_balanced_admission_v1/`。
原始数据、特征和父模型也需在本机合法存在；公开仓库不包含这些资产。
只有GitHub代码和聚合指标，不能凭空重建私有真实数据训练。

## 已完成运行记录

|阶段|PID|时间|核查|
|---|---:|---:|---|
|真实100+100/direct200恢复试跑|8653|15.66秒|400次真实更新，恢复完全相同|
|完整18头训练|9772|209.66秒|36,000主更新，额外2,000更新完整重训相同|
|完整72组推理评价|12163|125.41秒|两类旧对照共144项完整指标校验相同|
|单独进程完整重放|13361|122.28秒|全部预测、动作、指标和汇总相同|

这些PID是历史运行记录，不代表当前仍有进程。检查heartbeat及进程实际状态，
不能仅凭旧PID判断运行中。真实训练loss在 `training_traces.json`；实际梯度统计在
`optimization_pressure.json`。训练曲线/梯度统计不是正式泛化结论。

## 不重训的公开结果核验

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_gradient_balanced_admission_v1 --verify
```

该命令验证72个聚合JSON的完整字节与清单，不读取原始数据，不重新评价模型。
若从GitHub检出时没有展开的groups目录，用以下命令恢复聚合指标，不恢复原始数据：

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_gradient_balanced_admission_v1 --restore
```

## 有私有资产时的完整推理重放

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_gradient_balanced_admission.py verify
.venv-pytorch/bin/python scripts/report_m3w_gradient_balanced_admission.py
```

前者真正重新推理，核查注册代码、权重hash、行对齐、全部72组动作和分数。
后者从已验证结果和训练状态生成报告，不是又一次科学重算。
正式训练/评价完成凭据不可覆盖；若hash不同，先定位差异，不删除证据重新命名为成功。

## 只针对未完成运行的恢复

```bash
.venv-pytorch/bin/python scripts/run_m3w_gradient_balanced_admission.py train --resume
.venv-pytorch/bin/python scripts/evaluate_m3w_gradient_balanced_admission.py run --resume
```

只对尚无完整freeze/complete凭据的中断运行使用。不要对本轮已完成实验重跑这两条命令；
程序会拒绝覆盖。checkpoint每100步，heartbeat每50步，单进程锁防重复，12小时与
10GiB剩余磁盘保护。检查点含模型、优化器、RNG、抽样及梯度统计，可继续同一注册任务。
新假设须用新实验目录和先冻结协议，不能删除旧结果后复用名称。

## 针对性检查

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_gradient_balance_artifacts.py tests/test_m3w_gradient_balanced_admission.py tests/test_m3w_gradient_balance_readout.py tests/test_m3w_admission_monitor.py tests/test_m3w_admission_monitor_artifacts.py tests/test_m3w_admission_cost.py tests/test_m3w_admission_readout.py tests/test_m3w_active_bank_cost.py tests/test_m3w_active_bank_inference.py tests/test_m3w_metric_bundle.py
```

本轮34项通过，用时6.59秒；不是整个仓库测试重跑。精确重放证明工程可复现，
不恢复历史独立性。当前任务为观察8步、请求未来12步、raw stride12，不是补充raw t+50，
不能写成秒、米、人工gold、true3D或foundation。Stage5C未执行，SMC未启用。
