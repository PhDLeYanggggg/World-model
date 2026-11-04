# 本轮复现说明

本轮是冻结神经预测器上的树代价估计实验，不是新训神经世界动力学。
协议、代码、配置和registration先提交，再拟合。使用原生arm64环境。

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_shared_harm_refinement.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_shared_harm_refinement.py
```

上述命令是阶段入口。已经完整结束的阶段不会覆盖旧结果；未完成阶段加
`--resume`，继续原定义、原预算。不要重命名结果或通过重跑优化器挑测试赢家。
每棵树保存checkpoint，私有events.jsonl和heartbeat.json记录PID、耗时、进度。
num_workers=0，CPU4/interop1，不用资源探测或多进程数据加载。

两组均435维；差异仅为末48维保留还是置零。旧387维对照只读历史决策缓存，
校验SHA和逐行输入绑定，再重新计算所有指标；不是新推理旧模型代价。
完整verify重新推理54组代价并核对108组决策的指标，不声称MILP位级重求解。

结果看readout.md和gate_readout.md。负结果、不可估和空同切换数对照都保留。
23组是已经暴露的开发证据，原24组primary仍不可估，未启用独立校准或确认。
没有metric/seconds/true3D/foundation声明；Stage5C和SMC始终关闭。
