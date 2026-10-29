# 成本库拆分诊断复现

## 这轮做了什么

使用已经冻结的 54 个树模型和 54 个神经成本头，重新执行原始拟合数据
上的 144 组推理。没有训练新模型，没有打开 inner-held 或独立评价角色。
三个成本头种子不等于三个重新训练的轨迹预测器。

固定对照包括原始三种成本库、混合估计器只允许神经动作、物理与神经专用
估计器组合、混合估计器与自身重组，以及两个仅依赖预测值的拒绝原因子集。
自身重组必须逐值、动作和排序一致。专用估计器组合使用双倍估计器容量，
不能当成等算力或仅改变模型结构的实验。

## 本地命令

在仓库根目录使用原生 arm64 环境。线程为 4，数据加载 worker 为 0。

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_cost_bank_factorization.py register
.venv-pytorch/bin/python scripts/diagnose_m3w_cost_bank_factorization.py run
.venv-pytorch/bin/python scripts/diagnose_m3w_cost_bank_factorization.py run --resume
.venv-pytorch/bin/python scripts/diagnose_m3w_cost_bank_factorization.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_cost_bank_factorization_v1
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_cost_bank_factorization_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_cost_bank_factorization.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_cost_bank_factorization.py tests/test_m3w_multiaction_cost_readout.py
```

注册文件和绑定文件需已经提交。新执行使用 `run`，中断后才使用 `--resume`。
已经完成则使用 `verify`，不能覆盖旧结果。代码保留不可变分组、进程锁、
私有 heartbeat 和 10 GiB 空间下限；12 小时边界后应按原清单恢复，而不是
缩小实验。公开克隆不含原始数据、预处理包或模型权重，单靠克隆不能复现。

本轮运行 PID 17568，退出 0，用时 262.978860 秒；独立重放 PID 18293，
退出 0，用时 262.283743 秒。144 组及汇总完全一致。聚合压缩包
1,983,544 字节，逐文件校验通过，不含轨迹或权重。

## 结果边界

两类估计器拆分后拟合效用上升，但完整逐录像 2% 风险支持都下降，不部署。
空 easy 风险不可估，不能当安全零值；同切换数量对照只检验效用。拟合数据
上的区间不是独立泛化证据。原始帧、像素坐标与 detector-track 标签的限制
不变。Stage5C、SMC 和独立确认集仍未启用。
