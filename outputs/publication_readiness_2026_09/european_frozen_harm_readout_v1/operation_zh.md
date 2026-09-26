# 本轮运行与恢复

在项目根目录使用原生arm64环境。CPU计算线程4、interop1、workers0；
不使用默认x86_64 Conda，不使用DataLoader多进程。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_frozen_harm_readout.py --phase train --resume
```

首个模型的100步试跑属于固定2000步预算。恢复保留优化器、随机数和采样
状态，不重新计预算。私有目录data/stage_cvpr2027_experiments/
european_frozen_harm_readout_v1/保存checkpoint、heartbeat和运行日志。
观察超时不等于进程失败，核查原PID后继续，不重复启动。保持10GiB磁盘余量。

全部288个读出头训练结束后，先推送prediction_freeze.json，再运行：

```bash
.venv-pytorch/bin/python scripts/evaluate_m3w_european_frozen_harm_readout.py
.venv-pytorch/bin/python scripts/report_m3w_european_frozen_harm_readout.py
.venv-pytorch/bin/python scripts/plot_m3w_european_frozen_harm_readout.py
.venv-pytorch/bin/python scripts/verify_m3w_european_frozen_harm_readout.py --replay-only
.venv-pytorch/bin/python scripts/verify_m3w_european_frozen_harm_readout.py
```

本轮同时比较冻结mean表征与fractional表征，不能只报后者比旧头好而忽略
同样加训的对照。先验证模型及逐行预测，再写科学结论；测试通过不等于
主门槛通过。完整旧项目测试没有重跑时须明确标not_run。

只提交代码、报告和轻量聚合指标。不要提交原始数据、表征、权重、图像
或.venv-pytorch，不碰已有无关暂存项。不运行新策略，不开确认集，
Stage5C与SMC保持关闭。长期目标不会因这一轮程序完成而标记完成。
