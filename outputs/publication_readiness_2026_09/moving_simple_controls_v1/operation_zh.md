# 固定幅度强对照复现

本轮没有训练神经网络。先冻结13种静态选择，再对同一批62,466行重新评分。
旧学习策略只作为已校验对照，不是本轮重新训练。

在项目根目录使用原生arm64环境，单进程CPU4线程：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_simple_controls.py replay
.venv-pytorch/bin/python scripts/report_m3w_moving_simple_controls.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_fixed_amplitude_controls.py
```

replay重新计算所有静态选择与指标，并要求旧指标完全相同、公开文件不变。
不要与报告写入进程同时执行。它不重训预测器，也不证明独立泛化。
原prepare/evaluate支持`--resume`，只恢复完全相同定义，不能覆盖旧结果。
输入路径、缓存哈希、代码哈希和选择冻结记录见registration及choices_freeze。
原始数据、选择数组和图像留在data目录，不提交Git。

应核查全部静态行和录像风险，而非挑最高平均提升。当前没有安全赢家：
最小幅度带静止保护也只有5/24录像检查通过。旧失败、2%门槛、NA支持规则
不变；没有启用恒等认证、Stage5C或SMC。
