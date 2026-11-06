# 完整策略复现

本实验不是重新训练：它把原来只评估已知候选的范围补齐为 62,466 个
past-eligible 样本，保留未知标签和不切换的样本，执行原始逐录像风险合同。
均值与一致同意规则都未通过，不升级部署。

在项目根目录使用本机 arm64 环境：

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_donor_policy.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_donor_policy.py prepare
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_donor_policy.py infer --resume
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_donor_policy.py evaluate
.venv-pytorch/bin/python scripts/report_m3w_moving_complete_donor_policy.py
.venv-pytorch/bin/python scripts/run_m3w_moving_complete_donor_policy.py replay
```

入口固定 CPU 4 线程、interop 1、workers 0；文件锁防止重复任务。每个阶段
记录 PID、内存、进度和完成状态。推理按三个 context 保存，`--resume` 校验
已有结果后跳过完成项。私有路径为 `data/stage_cvpr2027_experiments/` 下的
同名目录；原始数据、预测、权重和图不上传 Git。

原 v1 推理因一个批尾分数差异约 1.06e-9 触发严格浮点校验而退出；此失败
保留。v2 只修工程校验：训练 RMS 归一化的 float32 容差，所有已知特征、
逐通道阈值决策和均值/一致同意选择仍须完全相同。不改变模型、风险阈值、
数据或科学评价规则。见 `numeric_verification_amendment.md`。

`replay` 重新推理并重算指标，逐数组和指标比较，不重新训练。复现时不要
同时运行报告生成器，否则公开文件不变性检查应当失败。中断后恢复同一
冻结版本，不重选模型；超过 12 小时或磁盘保护触发时保留完成缓存。

公开轻量证据是 `analysis.json`、`record_risk.csv`、`failure_taxonomy.json`、
注册和阶段记录。它们属于 exposed-development，不是新独立测试。
