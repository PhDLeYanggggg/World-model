# 同样本神经参考头对照

这不是新训练，也不改变已冻结的策略。它在主实验完成后，重新生成旧神经头的
18组成本输出及新头的108个参考数组；哈希必须与原评估一致。

```bash
.venv-pytorch/bin/python scripts/audit_m3w_hurdle_strong_reference.py run
.venv-pytorch/bin/python scripts/audit_m3w_hurdle_strong_reference.py verify
.venv-pytorch/bin/python scripts/report_m3w_hurdle_strong_reference.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_hurdle_reference_strong_control.py tests/test_m3w_hurdle_reference_reporting.py
```

定义与注册需先提交 Git；已完成阶段不能覆盖。若中断，用原命令续跑，已有逐任务
收据必须完全一致。私有 heartbeat 记录 PID、任务进度、耗时和缓存体积。使用本机
arm64 环境、CPU4线程、单进程，不新开 CREATE 作业，不上传权重或逐行缓存。

必须保留两个不同的评价目标：

- `conditional_known`：只在有监督标签的合法样本上评价参考误差预测。
- `original_bounded_label`：保留全部合法样本，对完全未知标签使用旧规则规定的
  零参考贡献。这个零不是“未来不会发生事件”的真实标签。

新头与旧头的训练目标、容量不同，因此只能说明实际输出的预测表现，不能归因于
单一架构变化。新头未改轨迹或介入决策；即使预测 loss 更低，也不能宣称原2%风险
合同、逐录像支持、独立确认或部署通过。23个支持来源均是已暴露开发资料；原24组
主要结果仍不可估。保留组、DUT 和 DroneCrowd 确认结果不开放。Stage5C/SMC 关闭。
