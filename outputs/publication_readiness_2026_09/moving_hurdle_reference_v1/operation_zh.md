# 本轮参考头复现

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py train
.venv-pytorch/bin/python scripts/verify_m3w_hurdle_reference_oof.py verify
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_hurdle_reference.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_hurdle_reference.py
```

已完成阶段不覆盖。未完成训练加`--resume`，从已保存的8树块继续，不宣称保留
未完成块。私有目录含checkpoint、OOF数组和PID/heartbeat，不能提交Git。
CPU4/interop1/workers0；森林单进程1线程。不能用x86Conda/Rosetta启动训练。

先用原C内部完整来源留出预测挑模型，再全C拟合；所有18任务结束才首次对H预测。
OOF复核要核对216个模型的来源、样本、输入、标签、权重及108组数组，而不只看
脚本成功。H复核核对18组完整输出、共108数组及全部指标。

这里的loss是参考误差MSE和事件Brier，不是轨迹ADE/FDE。预测变好不等于旧策略
安全，不得把概率当逐录像支持保证。旧策略、原2%要求、独立保留数据用途不变。
未启用Stage5C/SMC，没有metric/seconds/true3D/foundation声明。
