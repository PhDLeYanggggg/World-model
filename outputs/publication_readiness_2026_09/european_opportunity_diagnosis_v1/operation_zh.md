# 本轮归因结果与复核

这轮是实际计算的失败归因，不是新模型训练，也不是新测试集成绩。
使用已校验的预测、评分和决策，重算了 48 个策略、144 个 all/easy/hard
分解表，核对 1,728 个地点统计块，并完整重跑一次得到相同结果。

## 发现了什么

神经候选不是完全没有有用预测：事后在 CV 和神经预测之间选更好者，上限约
17.2%；简单阻尼对应约 8.7%。但真实策略不能看到未来，这不等于可训练得到的
成绩。当前 easy 神经风险策略只抓到神经候选潜在收益的 1.04%--2.78%，阻尼
策略抓到 23.31%--25.34%。约六成神经机会先被收益评分拒绝，其余大多被风险
规则拒绝。不能直接取消保护，因为被拒绝样本同时包含会造成伤害的情况。

另一个发现是预测器训练场景变化不能忽略：八场景模型没有稳定超过两个固定
四场景对照。但它们同时存在场景构成、归一化和训练预算差异，不能据此断言
“数据越多越差”，也不能认定这就是所有选择失败的根因。

已找到下一项具体实验：收益头目前对“低估伤害”施加四倍惩罚，却把其输出当
预期伤害相减；之后又有一个保守风险头把关。解析小例子验证了这种目标可以
把正的预期收益变成负评分。下一轮只改收益头损失，风险头和上限保持不动。
这项训练在本诊断包中尚未运行，不冒充已修复模型。

## 如何复核

目录：`/Users/yangyue/Downloads/World`。使用原生 arm64 环境：

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_opportunity_diagnosis.py --verify
.venv-pytorch/bin/python scripts/report_m3w_european_opportunity_diagnosis.py
.venv-pytorch/bin/python scripts/plot_m3w_european_opportunity_diagnosis.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opportunity_diagnosis.py tests/test_m3w_utility_objective_semantics.py
```

首轮使用 `--prepare` 然后 `--run`。中断后同一版本使用 `--run`，已完成种子按
身份和校验值恢复；`--verify` 强制重新计算全部诊断。不得覆盖身份不一致的
旧产物。没有本地私有输入时，只能查看公共结果，不能声称完成训练或计算复现。

首轮进程 PID 30958，UTC 2026-09-25 01:37:15 验证输入、01:37:33 完成。
复核进程 PID 31038，01:38:10 到 01:38:28，已成功退出。该任务读取现成数组，
不进行优化器更新，短运行时间不是训练模型很快的证据。没有新提交 CREATE
作业，也没有修改 simulation 项目。未来仍按其目录、资源与调度限制使用 HPC。

私有目录：`data/stage_cvpr2027_experiments/european_opportunity_diagnosis_v1/`，
保留逐种子结果、校验记录、进程事件和 heartbeat。公共精简指标约 1.65 MB，
完整展开的 `analysis.json` 和 PNG 本次不提交。SVG 和报告随代码公开。

保留集没有打开；部署没有变化。仍是像素、观察 8 步/预测 12 步、原始帧间隔
12 的源场景研究，不是历史 t50 成绩，不是米、秒、true 3D、foundation 或
人工 gold。Stage5C、SMC 未执行。完整结论见 [conclusions.md](conclusions.md)。
