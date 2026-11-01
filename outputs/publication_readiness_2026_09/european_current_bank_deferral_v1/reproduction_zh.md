# 当前预测库的公开 Deferral 对照复现

本实验比较连续 ADE 回归与 ICML 2024 多专家 deferral 损失，不是新轨迹模型。
旧版 deferral 已经跑过，但用的是另一预测库，不能混用成绩。
本轮只使用已暴露的 EuropeanSquares SOURCE：8 步观察、12 步预测、raw stride 12。
不是独立测试、SDD t+50、米、秒或真实 3D。Stage5C/SMC 均关闭。

## 本地训练

使用原生 arm64 `.venv-pytorch/bin/python`，不要使用 x86_64 Conda/Rosetta。
CPU 计算线程 4，interop 1，DataLoader worker 0；入口先检查架构，不做资源探测。
完整源数据、因果特征、原预测器和历史冻结权重需在本地存在，不能仅靠轻量报告重训。
`registration.json` 绑定代码、协议、依赖及上一轮完整产物；不得覆盖它来修改已运行定义。

```sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_current_bank_deferral.py pilot
.venv-pytorch/bin/python scripts/run_m3w_current_bank_deferral.py train
.venv-pytorch/bin/python scripts/run_m3w_current_bank_deferral.py eval
.venv-pytorch/bin/python scripts/run_m3w_current_bank_deferral.py verify
```

以上用于尚未运行的新阶段；已完成阶段拒绝覆盖。中断后给对应命令增加 `--resume`，
例如 `train --resume`。不要启动重复训练。私有目录下有 `heartbeat.json`、阶段日志、
进程锁和每 50 更新的检查点，包含模型、优化器、采样器、随机状态与损失轨迹。
观察超时不代表进程结束，应检查同一 PID/会话和输出增长。

必须完成 36 个模型、208320 次主更新、14336 次完整复训检查、72 组评价和单独数值复核。
速度慢不能自动减少样本或预算。单阶段上限 12 小时，保留 10 GiB 磁盘和 1 GiB 私有检查点上限。
400 更新真实试跑用于估算资源，不是模型效果试验。未知标签保留为 NaN，只参与因果上下文，
不能成为零误差监督或推断时的筛选条件。新分数和原控制必须逐一精确复核。

## 只复现公开报告

完整报告产生后，不加载数据、特征或权重即可重新计算全部标量汇总与图表：

```sh
.venv-pytorch/bin/python scripts/verify_m3w_current_bank_public_report.py
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_current_bank_deferral.py \
  tests/test_m3w_current_bank_deferral_evidence.py \
  tests/test_m3w_current_bank_public.py \
  tests/test_m3w_regression_deferral.py \
  tests/test_m3w_current_guard_support.py \
  tests/test_m3w_current_guard_public.py
```

公开重建会在导入报告前禁止 Python 的项目私有输入读取，允许写出本地结果图。
这不是操作系统隔离，也不是重新训练或独立科学复现。完整结果未产生时，相应产物检查
明确 skip，不视为通过。`verification.json` 记录的是完整分数推断和动作计分复核，
不是新运行整数求解；本实验没有用 deferral logit 冒充原生 ADE 效用。

## 固定筛选规则的失败取证

完整评价后的诊断单独注册，并保留原 72 组和 444 个重复录像视图。
本轮已经完成，拒绝覆盖；不应直接重启它。诊断使用真实标签解释错放和错拒，
这些标签从未进入推断筛选。只重建公开标量报告可运行：

```sh
.venv-pytorch/bin/python scripts/report_m3w_current_guard_support.py
```

`guard_support_diagnostic.json` 是新计算的诊断产物，`guard_support_statistics.json`
和 `guard_support_readout.md` 是从它重建的轻量汇总。12 个重复视图没有筛选后
已知 easy 参考支持，其中 6 个存在原始单目标查询结构下的支持阻塞。这不等于
原候选库没有好预测，也不等于整个多智能体世界没有邻居。错放和错拒均需修复。
后续修复必须与已有风险残差、低估惩罚、未知端点监督及 own-choice 对照区分，
不能重命名重复实验，不能使用 oracle 筛选部署。完整实验仍是 5/9 gates，未部署。

## 评价与计算位置

主要对照固定为 guarded TDEF 减 guarded ADE 的完整结果保守下界。
还需优于冻结 own-joint、相对 floor 下界为正，且全部原录像通过原 2% 风险要求。
Raw 版本和同数量裁剪仅作补充；空对照不可估，不是安全零值。
3000 次 locality bootstrap 先平均重复 context/head 视图，不能把重叠窗口当独立样本。

试跑后根据实际速度、内存和总成本决定本地或 CREATE，不能仅凭 import 成功判断。
CREATE 只在规模或数据位置确有需要时使用，先核对现有任务，通过调度器提交，记录 job ID，
保持相同注册定义和检查点；不因访问失败而推断远端任务终止或重复提交。
原数据、特征、缓存、权重及第三方 PDF 不上传 Git；只同步代码、配置、报告和轻量标量包。
