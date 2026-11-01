# 冻结成本与训练覆盖诊断复现

本轮没有训练新模型。它重新计算完整72组，确认原始候选轨迹、冻结成本门槛与分配器
各自丢失了什么，并从36个已冻结检查点重建真实查询采样情况。

```bash
.venv-pytorch/bin/python scripts/run_m3w_frozen_cost_support.py pilot
.venv-pytorch/bin/python scripts/run_m3w_frozen_cost_support.py run
.venv-pytorch/bin/python scripts/run_m3w_frozen_cost_support.py verify
.venv-pytorch/bin/python scripts/diagnose_m3w_actor_query_coverage.py
.venv-pytorch/bin/python scripts/report_m3w_frozen_cost_support.py
```

使用原生arm64环境，CPU4、interop1、workers0。当前已完成目录不能覆盖；中断的未完成
正式诊断可用`run --resume`，不会重新训练。真实4来源试跑17.011790秒、峰值9.72GB；
完整92.763621秒、峰值10.97GB。独立数值复算93.513954秒、峰值10.98GB。
上述阶段计时从runtime_ready开始，不含入口资产核验时间。

`registration.json`冻结诊断代码、测试、协议和原实验哈希。`complete.json`绑定全部72组；
`metric_groups.zip`只保存标量汇总。`verification.json`证明完整数值重算一致。
未来标签只用于这些oracle和评价，不进入推理。Oracle成绩绝对不能写成真实模型成绩。

`actor_query_coverage.json`来自冻结检查点中的当前查询键和采样随机状态，不读取未来
轨迹标签。它证明梯度batch的覆盖情况，不等于全流程未见样本，也不等于重训或全部
行抽样哈希的复现。大来源约32.5%至38.0%的查询没进入过该actor的梯度batch；
风险监控的全量计算仍然覆盖它们。

公开汇总存在后，可只重建报告和图片，不需要读取轨迹或权重；这不同于从原始数据开始
完整复现。图表的标签清楚注明“标签知情诊断，不是模型分数”。相关定向测试：

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_frozen_cost_support.py tests/test_m3w_multiaction_support.py tests/test_m3w_frozen_support_evidence.py tests/test_m3w_actor_query_coverage.py
```

完整仓库测试未由这些定向检查替代。当前没有新增CREATE任务；本轮本地资源足够，
此前SSH连接重置不代表历史任务失败。公开仓库不包含轨迹、特征、权重、私有查询键或缓存。

所有结论仅限暴露SOURCE开发数据：EuropeanSquares斜视检测器silver、本地图像坐标，
观察8步/预测12步/间隔12个原始帧。不是SDD t50、米、秒、人工gold、true3D或foundation。
未开启独立确认、部署、Stage5C或SMC。
