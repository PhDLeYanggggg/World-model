# 三种子预测器复核与接续

在项目根目录用已有的原生 arm64 环境运行：

```sh
.venv-pytorch/bin/python scripts/check_m3w_external_three_seed_bank.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_external_three_seed_bank.py tests/test_m3w_dronecrowd_frozen_inputs.py tests/test_m3w_dronecrowd_role_proposal.py
```

这不是训练命令。它核验本地已保存的 18 份权重、原训练记录、原预测数组和
影像筛查记录，只对已暴露 A 的因果输入做等值推理。缺失本地资产时应先按收据
找到对应归档并验证 SHA256，不能用新随机权重代替。已完成的核验可直接复用。
训练仍用原生 arm64、4 个 Torch 计算线程、1 个 inter-op 线程和零加载器进程。

若后续启动外部实验，先冻结科学角色和所用模型与代价头的训练录像清单。本次
筛查遗漏的 fold 2 的 10 个贡献录像已由[合并筛查](../dronecrowd_combined_forecaster_coverage_v1/README.md)
补齐。另一次[训练录像核对](../dronecrowd_controller_coverage_v1/README.md)覆盖直接
代价头和方向收益头的 42 个来源与种子组合。两份收据都不是外部策略冻结，不能
替代实际所用候选、阈值和策略的来源核对。按录像流式预测和保存进度，不物化全量输入表。
本次 A 上的粗略测算，按约 452 万目标行计算，单份 EqMotion 权重一次约 4.7
小时，PCT 约 0.2 小时，不是外部实测速度。正式预算先测代表性录像；如果整批
规模更适合 CREATE，先只读核实队列、数据和可恢复状态，避免重复作业。最新
CREATE 查询超时，远端状态未知，本轮没有提交任务。

本轮只准备对照，不读取 DroneCrowd 未来标签或报告外部效果。两种角色方案仍
未分配。原 2% 规则、不支持场景的不可估判定和历史失败结论均保留。
