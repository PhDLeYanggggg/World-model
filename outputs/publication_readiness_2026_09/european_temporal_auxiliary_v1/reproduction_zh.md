# 时序辅助监督训练与恢复说明

工作目录 `/Users/yangyue/Downloads/World`。本轮是训练器开发与工程验证，
正式真实数据拟合尚未启动，不能把下面的运行命令当作已经完成的实验。

## 本轮已验证

- 训练器及旧 target 测试 28 项通过，新增 readout 测试 28 项通过，合计
  56 项通过；未运行全部历史测试。
- 无辅助监督时，训练权重与原主损失实现逐项相同。
- 三组模型初始化、采样次数与顺序相同，只改变辅助监督。
- 中断后保留最后一个完整检查点，恢复结果与连续运行精确一致。
- 一个真实 TRAIN 来源完成输入校验与反向传播：109,454 行中 106,960 行
  有监督，实际反传批次 141 行。三组初始主损失一致，梯度有限。
- 真实数据优化器更新数为 0，新增真实模型检查点为 0，部署未改变。

## 环境与资源

使用原生 arm64 `.venv-pytorch`，四个计算线程，DataLoader worker 为 0。
不要使用 x86 Conda/Rosetta，不探测 GPU 或修改共享环境。
先检查存储，保留原来的 10 GiB 余量与检查点额度。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary.py preflight
```

最初试跑 `allowed=false`：还差约 1.67 GiB。10 月 5 日 12:07 UTC 复查
差额变为约 1.91 GiB。不要跳过检查或删除无关资产。
空出约 3 GiB 后仍需重新检查，或等 CREATE 恢复后按已授权的独立 M3W
环境、存储和调度器边界迁移；本入口不会自动提交 HPC 作业。

## 正式试跑与继续

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary.py pilot
```

试跑对首个来源、首个 head seed 的三组模型各更新 100 次，另做一次
不中断重放核对。代码按真实运行时间和内存估算完整 216 次拟合的成本。
本轮执行这条命令返回存储门槛错误，尚未进入训练，也没有产生 `pilot.json`。

试跑成功且资源合适后，复用试跑检查点继续固定 2,000 次更新：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary.py train --resume
```

检查点每 100 次更新原子保存，包含优化器、随机状态、采样哈希和损失历史。
若中断，先核对实际进程是否终止，不能只凭旧心跳判定；再用相同命令恢复。
不要修改原协议、输入或超参数来续跑。心跳、日志、锁及检查点在
`data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1/`，不提交 Git。

正式训练完成后，必须先核验并提交 `training_freeze.json` 和轻量训练记录。
readout 已实现，并在提交 `4b842ae2` 冻结，不需要另行设计阈值或挑检查点。
当前没有已完成的真实验证比较、独立测试或可部署新模型。

## 已冻结的评估流程

先检查是否具有完整训练结果；这条命令不读取新的验证集预测：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary_readout.py preflight
```

只有 216 个固定最终检查点、训练记录和四个强对照均可核验时才能继续。
实际新验证预测仅由下面命令读取，当前尚未运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary_readout.py run
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_temporal_auxiliary_readout.py verify
```

旧强对照通过已授权的 M3W 只读路径核验；远程不可达时停止比较，不跳过对照。
每个来源的输出不可覆盖；中断后确认进程终止，再以 `run --resume` 重放并核验
已有输出。`verify` 重新预测并核验所有数值和独立标量算法，不重新训练。
缺标签样本保留在策略评估中。未知标签在两种策略共享的选择中相互抵消，
只在不同选择之间保留不确定性；不能把两个收益下界相减当成提升已获证明。

## 仅工程检查

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python -m pytest \
  tests/test_m3w_temporal_auxiliary.py tests/test_m3w_temporal_target_audit.py \
  tests/test_m3w_temporal_target_reporting.py \
  tests/test_m3w_temporal_auxiliary_readout.py \
  tests/test_m3w_temporal_auxiliary_readout_runner.py -q -p no:cacheprovider
```

`inspect` 阶段只做真实 TRAIN 前向/反向传播，不更新权重，不评 validation。
它已完成，报告为 `input_check.json`；原来源检查点与划分哈希仍需一致。
复核旧检查不等于重新训练，更不等于泛化提升。

继续保留 2% selected positive easy-harm 风险定义。逐步正伤害均值不能替换
整段轨迹正伤害。Future labels/masks 仅用于监督和离线评估，不能进入推理。
不声明米、秒、human gold、true 3D 或 foundation；Stage5C/SMC 继续关闭。
