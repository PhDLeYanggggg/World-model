# 复现与数据边界

本次不是新训练。36个已训练的真实Torch模型来自3个实际训练种子、3个SOURCE折和4种输入。
本轮重新构建38,102行输入，重新推理并完整重放36组预测，随后才读取6个模型选择场景的标签。
54组指标和28个录像的标签已单独从原始数据复核。校准集、最终确认集和DroneCrowd未读取。

公共标量报告可运行：

```bash
.venv-pytorch/bin/python scripts/report_m3w_selection_forecast.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_selection_forecast_readout.py tests/test_m3w_selection_forecast_report.py
```

报告重建只读取仓库中的标量JSON，不读取私有轨迹、特征、预测、标签或checkpoint。
它核对54组文件哈希、完成回执和复核回执，重算6个场景为单位的3000次bootstrap。
完整原始数据复现需要依法获取原始数据及私有SOURCE checkpoint，不能把公共标量重建说成重训。

完整流程入口为run_m3w_selection_forecast.py和run_m3w_selection_forecast_readout.py。
已完成阶段拒绝覆盖原结果，不应删除回执后重跑。中断阶段使用原绑定和--resume。
CPU4线程、单进程、arm64 .venv-pytorch；checkpoint/heartbeat和原始运行PID均有记录。

本次主要结果：已知标签增益为正，但预先登记的保守主指标为负，easy和正伤害风险不通过。
因此不部署、不宣称投稿就绪；它是模型选择证据，不是最终独立确认。
obs8/pred12、raw stride12、local-image pixel、detector-silver；不是米、秒、true3D或foundation。
Stage5C不执行，SMC不启用。
