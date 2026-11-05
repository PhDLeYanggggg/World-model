# 固定预算续训：结果与复现说明

## 这次做了什么

从上一轮 8 个风险头的检查点继续，不重新初始化。网络仍为
120→64→32→2，9,890 个参数；优化器、学习率、输入、风险标签和阈值不变。
每折至少增加 4,096 次更新并完成整轮，总计增加 32,871 次更新。
小数据折最终训练到 544–2,080 轮，大数据折为 102–138 轮。

所有训练和完整重复训练完成后，才统一读取最终 checkpoint 的源外表现。
没有根据源外结果挑轮数，也没有调整原 2% 要求或引入恒等回退认证。
这是神经风险回归头的续训，不是新训练 Transformer 轨迹预测器。

## 结果

训练 loss 在 8/8 折下降，源外 MSE 却在 6/8 折上升。
相对森林对照，三组决策成本改善率为 -4.72%、-68.65%、-6.69%，筛选 0/3。
结果支持进一步调查过拟合和来源差异，不支持把“再多训练”当作已验证修复。
完整 loss 记录见 `loss_trace.csv`，不是只保留最好的 checkpoint。

本地对照图：
`data/stage_cvpr2027_experiments/moving_risk_convergence_v1/figures/fitting_and_transfer.png`。
它区分拟合误差与源外误差；图像、数据和模型文件不上传 GitHub。

## 执行顺序

工作目录 `/Users/yangyue/Downloads/World`，使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py fit
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py eval
.venv-pytorch/bin/python scripts/report_m3w_moving_risk_convergence.py
.venv-pytorch/bin/python scripts/run_m3w_moving_risk_convergence.py replay
```

训练前须提交注册文件；父检查点、输入缓存、固定参考缓存必须存在并通过 hash。
上面是本轮实际使用的完整顺序。完成阶段不可覆盖，不要删除回执来伪装新实验。
真实中断时只恢复同一未完成阶段，例如 `fit --resume`。模型、优化器、随机状态、
当前打乱顺序、批次位置和访问计数一起恢复。每 128 次新增更新保存并写心跳。
只读重放期间不要运行报告生成器。

CPU 4 线程，inter-op 1，workers 0，不做设备资源探测。训练约 25.403 秒，
完整重复训练约 20.501 秒，峰值内存均不足 0.8 GB。本轮风险头很小且输入已缓存，
快速完成不是省略更新；没有新建 CREATE 作业。单阶段上限 12 小时，预留 5 GiB 磁盘。

针对性检查：

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_risk_convergence.py tests/test_m3w_risk_convergence_reporting.py tests/test_m3w_neural_risk_transport.py tests/test_m3w_neural_transport_reporting.py tests/test_m3w_match_attribution.py tests/test_m3w_track_analogue.py tests/test_m3w_track_analogue_reporting.py tests/test_m3w_boundary_audit.py -q
```

本轮 24 项通过；未重跑历史全测试集。完整重复续训和恢复一致是工程复现证据，
不等于多种子、不等于新的独立测试、更不等于模型有效。
当前仍是原生观察 8 步/预测 12 步的图像坐标开发实验，标签不是 human gold。
原 24 组主评价仍不可估，Stage5C/SMC 不执行，论文尚未达到投稿准备完成状态。
