# 诊断与复现

本轮不重新训练，只读取已核验的因果特征和冻结模型，重新提取专家输出。
真实未来误差和简单样本标签只进入离线误差分解，不能用于实际推理。

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_cost_attribution.py --phase run
.venv-pytorch/bin/python scripts/report_m3w_european_cost_attribution.py
.venv-pytorch/bin/python scripts/plot_m3w_european_cost_attribution.py
.venv-pytorch/bin/python scripts/run_m3w_european_cost_attribution.py --phase verify
.venv-pytorch/bin/python scripts/verify_m3w_european_cost_attribution.py
```

run 会校验并复用已完成的组；verify 则从冻结权重和原始特征重新计算全部组。
运行使用 arm64 环境、CPU 4线程、interop 1、workers 0。每组写心跳，可恢复。
CREATE 仅只读查看队列，没有提交作业或修改其他项目，M3W 远程目录仍未核实。

本轮标签替换结果不能作为模型效果；不能把重复窗口当独立样本，不能把负交叉
项删除后归因。重点区别“相对旧模型增加的误差”与“当前全部误差”。
下一步是单独注册成本预测加辅助分类的训练对照，不修改当前推理策略。
结果为开发性证据，不是独立测试或安全认证。Stage5C/SMC 不执行。
