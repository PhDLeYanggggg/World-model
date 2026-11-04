# 固定决策风险分解复现

```bash
.venv-pytorch/bin/python scripts/audit_m3w_moving_record_risk_factors.py run
.venv-pytorch/bin/python scripts/audit_m3w_moving_record_risk_factors.py verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_fixed_risk_factor_audit.py
```

已完成阶段不会覆盖；不要重新生成旧结果来挑有利值。`--resume`仅用于未完成
阶段的原定义重算。输入是上轮已封存的逐录像代价与策略汇总，每份文件校验SHA。
本轮没有重新读取轨迹、推理模型、训练模型、调整阈值或修改介入决策。

四项分解是恒等式，不是四个可部署的新策略。修正项用了真实标签或保守标签界，
只能作诊断，绝不能接入推理。未知参考误差置零是原评价界的构造，不是观测事实。
47个“已选行都有某个未来标签”的视图也不等于12步全部标签完整。

全部4176录像代价视图和8352风险分解已经重新计算并精确复核。原2%要求、
原失败结果、不可估视图和封闭数据用途不变。Stage5C和SMC未启用。
