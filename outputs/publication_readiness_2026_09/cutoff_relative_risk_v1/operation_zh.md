# Cutoff-relative 对照运行与恢复

沿用已批准的 SDD 观察8／预测12、stride12、annotation pixel 协议和2%风险容忍度。
这轮是36个新风险头，不是新Transformer／EqMotion训练。既有72个控制头只验证后
复用，不把缓存记作新训练。四个场景已有开发暴露，不能当独立确认。

```bash
.venv-pytorch/bin/python scripts/run_m3w_cutoff_relative_risk.py --phase train --view coupa_seed17 --action damped_velocity_005 --stop-at 16
.venv-pytorch/bin/python scripts/run_m3w_cutoff_relative_risk.py --phase all --resume
.venv-pytorch/bin/python scripts/run_m3w_cutoff_relative_risk.py --phase decide --verify
.venv-pytorch/bin/python scripts/run_m3w_cutoff_relative_risk.py --phase evaluate --verify
.venv-pytorch/bin/python scripts/check_m3w_cutoff_relative_risk.py
.venv-pytorch/bin/python scripts/report_m3w_cutoff_relative_risk.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_cutoff_relative_risk.py tests/test_m3w_dimensionless_risk.py tests/test_m3w_net_easy_guard.py tests/test_m3w_net_easy_risk.py tests/test_m3w_native_metrics.py -q
```

运行环境为本地arm64 `.venv-pytorch`，CPU4，interop1，workers0。不要用x86_64
Conda；不是通过缩减样本、模型预算或改阈值来处理训练慢。每16棵树保存检查点，
恢复会检查数据、目标、训练抽样、预处理、代码和配置身份。并发写入会被锁拒绝。

私有缓存：`data/stage_cvpr2027_experiments/cutoff_relative_risk_v1/`。
heartbeat.json和events.jsonl记录PID、训练组、树数和六目标MSE。判断任务是否
活着必须同时核对实际PID或会话，旧heartbeat文件不能证明仍在运行。

本轮复用本地与GitHub上已核验的上一轮结果。CREATE最近保存的连接记录是认证
失败，当前远程任务和资产未知；本地资源足够，未为本轮重复提交或干预远程作业。

两个新输入是log(history_scale/train_cutoff)和log1p(forecast_disagreement/train_cutoff)。
cutoff必须来自训练源，不能用排除场景或新外部测试误差估计。改变坐标单位时，
cutoff也必须同单位换算；这不能把SDD像素cutoff直接赋给另一个数据集。相对原生
特征，这主要是保留信息的重新参数化，不是增加神经模型容量或证明新方法创新。

核验时使用原六目标所继承的标签／决策cutoff，不误用旧预处理对象中用于抽样分层
的同名cutoff；二者不一定相等。本轮目标和抽样各自保持原值，没有重新定义easy。

本轮没有混入新的prefix精度修复，也不打开IMPTC／DroneCrowd预测误差。无法满足
easy保护或CI不能支持优势时继续保留负结果，不升级部署。Stage5C、SMC关闭。
源数据、特征、权重和决策缓存不提交Git；只提交代码、配置、报告和轻量统计。
