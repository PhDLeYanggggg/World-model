# 欧洲开发域成本头：运行、恢复和核查

更新：2026-10-02。本文对应 `european_cost_harm_newton_v1`，不是旧 SDD
selector 部署教程。完整 72 头训练正在运行；下列最终核查步骤须在完成后执行。

## 本轮到底训练什么

本轮重用冻结的轨迹预测器和森林分区，只训练条件伤害成本头。比较原始森林、
加性伤害头、正值 Poisson 头和正值平方成本头。前三个是逐项核验的旧模型，最后
一个是本轮新拟合；没有新增神经动力学训练，也没有新树分裂。

数据为 12 个已暴露的欧洲开发 locality，各 locality 内按完整录像分开拟合与
验证。三种成本头种子不是三次端到端预测器训练。独立选择、风险校准、最终确认
仍关闭。观察 8 步、预测 12 步，raw stride 12，坐标为 image-local，标签为
detector-silver；不是米、秒或 human gold。

2% 的保护预算指选中 easy 样本的正伤害总和除以其 reference error 总和，
不是全 easy 净误差变化。未知标签不能补零或删除；缺支持和未定义分母不通过。
所有 future target 仅用于训练损失或离线评估，不进入推理。

## 环境和资产

在项目根目录使用 `.venv-pytorch/bin/python`。macOS 必须是 arm64，不使用
x86_64 Conda/Rosetta。计算线程 4，interop 1，DataLoader workers 0。
本轮是明确的 NumPy 数值成本优化器，不冒充 Torch 神经网络训练或 Torch fallback。

必要资产包括冻结的本地预测缓存、源森林 checkpoint、旧实验登记与报告，以及
CREATE 上的加性/Poisson checkpoint。仅克隆 Git 仓库不能完成新训练。脚本会检查
输入、划分、目标、历史特征、配置和 checkpoint 的 hash，不允许通过刷新 hash
绕过差异。

CREATE 只使用已授权的 `/users/k24101830/m3w/`。本轮计算在本地，远程仅存储和
核验 M3W 权重、读取本人队列；没有新 Slurm 科学计算。不要触碰 simulation 的
任务和目录，不向 Git 提交认证信息、数据或权重。

本机磁盘低于固定 10 GiB 数值缓存储备，因此不生成本地数值缓存；聚合报告预算
20 MiB，远程新增 checkpoint 上限 512 MiB。不得通过删除无关数据腾空间。

## 已登记实验与恢复

登记提交为 `6160fcea`，真实单头 pilot 记录在 `pilot.json`。不能原地修改登记
源码、配置或协议后继续训练。修复须保留旧版本并使用新身份。

完整训练的实际入口：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_cost_harm_newton.py run --resume
```

不要在同一实验运行中再启动这一命令。首先检查
`data/stage_cvpr2027_experiments/european_cost_harm_newton_v1/heartbeat.json`
中的 PID，查看进程是否存在、CPU 时间是否增长，以及 `events.jsonl` 的阶段变化。
读取超时不代表训练卡死。初始 full-run PID 为 49738，恢复后以新心跳为准。

checkpoint 按完整头保存，不是按每棵树或优化器步保存。中断后 `--resume` 会
核验并跳过已完成的头；正在拟合、尚未保存的头需要重算。先确认旧进程终止，
再恢复，不启动重复任务。文件锁会阻止并发写入。

本轮每头均做两次确定性拟合、序列化和推理核对；权重只写入
`/users/k24101830/m3w/european_cost_harm_newton_v1/inputs/`。单头完成日志不是
72 头实验完成。`complete.json` 只在所有头、远程权重核验和聚合成功后写入。

## 最终核查

确认训练进程正常退出后运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/verify_m3w_cost_harm_newton.py
```

该核查独立重算聚合量和 locality bootstrap，检查 72 头报告、登记、旧模型对照、
训练收敛、完整权重 manifest 与完成记录。它不是新的训练，也不是新的独立测试。
运行入口的 `verify` phase 会重新拟合全部头，成本较高，不等同于上述聚合核查。

定向工程测试：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_cost_aligned_positive_harm.py \
  tests/test_m3w_cost_harm_newton.py \
  tests/test_m3w_cost_harm_newton_boundaries.py
```

前两组 8 项在真实 pilot 前已通过；最后 5 项于完整训练期间独立补查并通过，
未修改注册模型。它们涵盖多叶不等权有限差分、独立 SciPy 对照、叶权重尺度、
零实际伤害与负等价训练目标、迭代耗尽报错。测试通过不等于研究 gate 通过。

## 如何解释结果

旧 Gauss-Newton 实现在 TRAIN 上不收敛；保留在
`../european_cost_aligned_positive_harm_v1/`，不能把它写成完成训练后的负结果。
本轮保留同一损失、初始化、正则化和 1e-7 收敛要求，改用经过独立有限差分
验证的完整残差曲率。非凸目标的驻点不等于全局最优。

最终同时读取 `summary.json`、`complete.json` 和 `verification.json`：

- 成本 MSE 是否改善，不等同于 ADE/FDE 改善。
- 全量与相同介入数下的 utility 都要比较，不能靠多切换制造优势。
- 已知标签风险、未知标签上界、支持率和最差场景必须分别报告。
- 3,000 次 locality bootstrap 是已暴露开发比较的名义区间，不是自适应搜索
  校正后的确认性置信区间，也不是有限样本安全保证。
- 通过内部 advance screen 也不会自动部署或打开独立数据角色。

Stage5C 和 SMC 保持关闭。正式投稿仍由作者最终确认。
