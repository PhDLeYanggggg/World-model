# 本轮成本学习与录像约束实验操作说明

## 1. 先区分三种工作

1. **重新训练/推理**：实际运行 Torch 优化器或完整前向推理，标记 fresh_run。
2. **核验已有产物**：验证数据、模型、配置及结果哈希，标记 cached_verified。
3. **汇总文稿**：从已核验指标生成表格和英文稿，不是重新训练、重新 bootstrap 或独立测试。

旧结果仍是已暴露的 SOURCE 开发实验。不能重新命名为独立 test。
本轮不是历史 raw-frame t+50；任务为观察 8 步、请求预测 12 步，步间隔为
12 个原始帧。数据时间和尺度未另行验证，不能写成米或秒级预测。

## 2. 本地运行环境

仓库位于 `/Users/yangyue/Downloads/World`。使用本机 arm64 的
`.venv-pytorch/bin/python`，不要换成历史 x86_64 Conda/Rosetta 环境。
当前正式脚本使用 CPU 4 线程、interop 1 线程、单进程加载，不需要探测 GPU
资源。已有的真实训练和恢复试验比单独 `import torch` 更有证明力。

各实验私有目录中的 `heartbeat.json` 和 `events.jsonl` 保存进度、PID、
优化指标及拟合完成事件。检查点包含权重、优化器、随机状态和输入身份。
健康但慢的训练可以继续运行；脚本保留 10 GiB 磁盘空间，单次健康运行边界为
12 小时。不要仅因速度慢而改成较小样本、少种子或假称完成。

## 3. 找到本轮四组实验

| 实验目录名 | 变化 | 正式训练量 |
|---|---|---:|
| european_multiaction_cost_v1 | 连续收益/伤害等五个成本矩预测 | 54 拟合、108,000 更新 |
| european_multiaction_record_policy_v1 | 冻结成本头、整录像约束决策 | 54 拟合、27,000 全梯度更新 |
| european_regression_deferral_v1 | 公开多专家 regression-deferral 损失对照 | 54 拟合、108,000 更新 |
| european_multiaction_support_scale_v1 | 仅修改第三个正支持量约束的数值单位 | 54 拟合、27,000 全梯度更新 |

公开配置和轻量结果在 `configs/` 和 `outputs/publication_readiness_2026_09/`。
数据、逐行缓存及权重位于被忽略的 `data/stage_cvpr2027_experiments/`。
禁止把这些大文件、视频、图像或第三方原始数据加入 Git。

## 4. 从轻量归档恢复指标

```bash
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_multiaction_record_policy_v1 --restore
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_regression_deferral_v1 --restore
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_multiaction_support_scale_v1 --restore
```

恢复命令只写回归档中已有、且本地缺失的 owned 指标 JSON；已有文件若哈希
不同会报错，不覆盖。归档不含训练数据或模型。恢复与字节核验不能称为重新推理。

## 5. 有完整私有资产时重新推理

```bash
.venv-pytorch/bin/python scripts/eval_m3w_multiaction_record_policy.py --verify
.venv-pytorch/bin/python scripts/run_m3w_regression_deferral.py verify
.venv-pytorch/bin/python scripts/run_m3w_multiaction_support_scale.py verify
```

旧录像约束实验使用独立 inference adapter：它修复单一动作的 NumPy 索引
维度，不修改冻结的训练代码、权重或决策规则。不要改用存在原始形状问题的
旧读出入口。每次完整 verify 会逐组复现前向结果、比较器及汇总；其运行记录
与最初读出分开，不覆盖初次运行时间。

## 6. 中断恢复，而不是另开重复训练

只对**尚未完成**的相应阶段使用以下命令；已完成阶段会主动拒绝覆盖。

```bash
.venv-pytorch/bin/python scripts/run_m3w_regression_deferral.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_multiaction_support_scale.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_multiaction_support_scale.py evaluate --resume
```

运行前先核对 PID、心跳和锁，确认没有同一实验仍在运行。不因一次连接失败
就认定任务死亡。恢复保留原配置、抽样、随机状态和拟合名单，不是新实验。
若要从零重新训练全部正式模型，需要单独的干净复现实例及同身份数据资产，
不能在当前冻结目录直接覆盖终点，也不能把完成的训练命令当作可重复覆盖接口。

## 7. 生成比较表和稿件

```bash
.venv-pytorch/bin/python scripts/report_m3w_cost_decision_evidence.py --check
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript_v6.py --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_multiaction_cost.py tests/test_m3w_multiaction_cost_readout.py tests/test_m3w_multiaction_record_policy.py tests/test_m3w_multiaction_record_inference.py tests/test_m3w_record_support_units.py tests/test_m3w_regression_deferral.py tests/test_m3w_metric_bundle.py tests/test_m3w_cost_decision_evidence.py tests/test_m3w_evidence_manuscript_v6.py
```

核验导出不等于科研结论成立。主表的已知标签收益不能替代缺失标签完成界；
重复的 context/head 视图不能当成独立录像；只通过部分录像不能写成全策略通过。
训练日志最后一条约束监控位于最后一次参数更新之前，和终点前向读出分开报告。

## 8. CREATE 与投稿边界

本轮正式优化和推理都在本地完成，没有提交新的 CREATE 作业。本说明不确认
CREATE 当前队列、迁移后环境或旧任务状态。需要转到 CREATE 时，先只读核对
调度器、已有 PID/日志/产物、环境与数据哈希，再决定是否提交，避免重复作业。

作者管理信息另存于被忽略的私有目录，不进入匿名稿件。个人 OpenReview
主页已核实不等于会议登记或投稿已完成。正式投稿仍需用户最终确认。
英文稿件仍是研究草稿，不是已达 A 会投稿候选质量的证明。Stage5C 未执行，
SMC 未启用，不声称 true 3D、foundation、metric 或秒级预测。
