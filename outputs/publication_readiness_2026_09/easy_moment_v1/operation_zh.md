# 条件 Easy 风险实验操作说明

## 这次训练什么

固定 damping005、Transformer、EqMotion 预测器及旧的神经增益/伤害头，
新增 36 个小型四输出风险森林。它们不是 36 个新世界模型，也不是重新训练
Transformer。直接估计 easy 加权伤害与分母，与“easy 概率乘总体伤害”的
近似共用同一模型，从而检查风险目标是否与 easy 保护要求对齐。

数据仍是四个已暴露 SDD 场景，8 步观察、12 步预测，标注像素。每个模型排除
自己的外场景，其训练成本的预测器也排除相应监督行的场景。这个排除关系
支持开发实验，不能把反复用于设计的场景变回独立确认数据。

## 本机运行

使用原生 arm64 环境，CPU4、interop1、workers0；不使用默认 x86 Conda。
不调用 Torch 设备资源探测，不启动 DataLoader 子进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_easy_moment.py --phase preflight
.venv-pytorch/bin/python scripts/run_m3w_easy_moment.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_easy_moment.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_easy_moment.py --phase verify
.venv-pytorch/bin/python scripts/verify_m3w_easy_moment.py
.venv-pytorch/bin/python scripts/report_m3w_easy_moment.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_easy_moment.py tests/test_m3w_protected_motion_controls.py tests/test_m3w_protected_eqmotion_controls.py tests/test_m3w_scene_risk_bounds.py tests/test_m3w_native_metrics.py -q
```

没有已有模型时 `--resume` 也会开始新拟合；有模型时必须显式使用它，不能覆盖
已有断点。每 16 棵树保存原子 checkpoint 和四个训练目标的 MSE，完成128棵后
写完成回执。恢复会校验代码、配置、输入、目标、抽样次数、源站排除和软件版本。
36 个模型未全部完成时不允许进入结果评价。文件锁阻止同目录重复任务。

首个16树试跑用于验证实际计算与断点，随后恢复到相同128树预算；不是把小试跑
冒充完整实验。初始读取字段和 easy 分位点的两次 pre-fit 错误见 execution_notes。

## 核查结果

- `data/stage_cvpr2027_experiments/easy_moment_v1/heartbeat.json` 记录最新PID和进度，
  但文件存在本身不证明进程存活；应同时检查该PID或原运行会话。
- `trials/<view>/<action>/complete.json` 记录每个实际完成的拟合和 checkpoint 哈希。
- `decisions_complete.json` 必须先于新结果读取完成，所有选择不接收未来标签。
- `analysis.json` 保留每个固定策略、场景、种子、尾部误差、缺标签数量和完整网格
  增益界限；`verification.json` 记录精确推理和汇总回放。
- `independent_verification.json` 使用独立运算核查真实误差、选择规则、抽样预算、
  分场景指标及配对区间。它不是另一个研究团队的独立研究复现。
- `results.csv`、`results.md`、`training_losses.md` 是轻量导出，不替代源结果。

Easy 上限仍为每个场景/种子不超过2%的相对退化；零CV误差样本另报受损数量。
模型预测的风险不等于校准过的风险保证。若 direct moment 失败，不在当前版本
上反复调阈值，也不把 bootstrap 的四个开发场景说成独立确认。

## 资源和限制

试跑中16树拟合循环约2.77秒，不含加载和检查；不能用它替代总运行时间。
完整模型的全部 trace 保留。原始数据、缓存、checkpoint 不进入Git。
CREATE 最新读取存在认证问题，队列状态未知；本次本地计算无需绕过远程限制。
HT21仍隔离，DUT不重开预测，DroneCrowd确认角色保持关闭。Stage5C和SMC未运行。
