# 神经风险修正：训练、结果与复现

## 这轮确实训练了什么

8 个真实 PyTorch MLP，每个 9,890 参数，结构是120→64→32→2。
输入为原因果特征；输出是全体和 easy 的 signed risk，不是未来轨迹。
另外拟合8个固定岭回归控制。主要训练共5,792次优化更新，每个拟合样本32遍。
验证阶段把全部8个神经模型和8个岭模型重新训练，结果精确一致。
这不是 NumPy fallback，也不是新的 Transformer/JEPA 动力学成果。

新方法用“查询风险预测 + 参考样本的真实风险减其预测风险的加权平均”。
参考只来自其他来源，使用原先冻结的32个身份桶；查询的未来标签只用于评价。
没有改2%规则，没有恒等回退认证，没有新开保留测试集。

## 结果怎么看

相对同一神经模型直接预测，风险修正使决策成本改善约1.77%、6.43%、0.89%。
但相对原森林控制仍为-6.91%、-54.52%、-5.52%，内部筛选0/3，不部署。
这些不是轨迹ADE/FDE改善，也不是全策略安全证书。

完整loss已保存为 `loss_trace.csv`，README列出每个来源的起末损失和源外误差。
例如来源076，首次记录loss约0.980，32遍后约0.526，源外MSE约7.738。
说明确实发生了优化，但没有可靠泛化。32遍完成不等于已经收敛。

## 本地复现

工作目录 `/Users/yangyue/Downloads/World`，使用原生arm64环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_moving_neural_transport.py replay
.venv-pytorch/bin/python -m pytest tests/test_m3w_neural_risk_transport.py tests/test_m3w_neural_transport_reporting.py tests/test_m3w_match_attribution.py tests/test_m3w_track_analogue.py tests/test_m3w_track_analogue_reporting.py tests/test_m3w_boundary_audit.py -q
```

首次流程为 `register`、`pilot`、`run`、`verify`、`replay`。已经完成的阶段不可覆盖。
仅在某阶段真实中断时，对同一阶段加 `--resume`。恢复包括参数、优化器、随机状态、
当前排列、批次位置、每行访问计数；不是重新从头跑。pilot在真实29,807条拟合行上
检查“一遍后恢复到两遍”与连续两遍完全相同。

本地检查点位于 `data/stage_cvpr2027_experiments/moving_neural_transport_v1/`。
原始特征和检索参考缓存仍留在本地；GitHub只存代码、配置、报告和轻量统计。
缺少这些本地资产不能只凭GitHub报告复现真实数据结果，不能以随机数据代替。

CPU4/inter-op1/workers0；真实训练约10.272秒，峰值RSS约1.286GB。
完整重训验证约10.442秒，峰值RSS约1.250GB。模型小、特征已缓存，因此快；
不是减少了32遍或约定折数。每遍及每128步保存，逐遍心跳。日志中的旧forest_threads
字段不代表本次训练森林。本机资源充足，没有新增CREATE作业。

19项针对性测试通过，历史全测试集未重跑。只读replay禁止重新拟合并核对结果。
完成回执写在私有目录；重复运行已完成阶段会拒绝，不删旧回执伪装fresh_run。
只读replay期间不要同时运行报告生成器。

当前是已暴露开发数据上的风险模型实验。独立确认、完整策略安全、三种子正式
主结果都没有由这次实验完成。原24组主评价仍NA，Stage5C/SMC不执行。
