# 本轮实验操作说明

## 已完成的实际工作

首次冻结特征提取约 2.01 小时，独立重新提取校验约 2.03 小时。
全部 1,913,814 行原预测逐元素一致，897,060 行 h64 特征也全部一致。
随后训练了 36 个真实 PyTorch 代价头，不是 NumPy 回退或小样本探针：
主更新 116,160 次，完整重训复现 4,000 次，恢复试跑 1,200 次。
全部新预测和 72 个指标分组再次完整重放一致。

主要指标未通过，完整风险要求也未通过。本轮不部署新模型。
最差录像 easy 净退化 39.37%，不能用整体 easy 平均改善代替。

## 环境与恢复

在项目目录使用原生 arm64 环境，不使用默认 x86 Conda：

```sh
.venv-pytorch/bin/python -c "import platform; print(platform.machine())"
```

正式运行采用四个计算线程、一个 inter-op 线程、零 DataLoader workers，
不进行 Torch 资源探测。训练入口先检查架构，使用原子 checkpoint。
日志、heartbeat 和私有权重在：

```text
data/stage_cvpr2027_experiments/european_eqmotion_context_cache_v2/
data/stage_cvpr2027_experiments/european_interaction_feature_cost_v1/
```

下列是中断任务的同定义恢复方式，不是要求重新启动已完成的任务：

```sh
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_context_cache_v2.py build --resume
.venv-pytorch/bin/python scripts/run_m3w_interaction_feature_cost.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_interaction_feature_cost.py infer --resume
```

完成阶段是不可覆盖的。不要删除完成标记、改变种子或缩小预算来强行重跑。
本轮本机峰值内存约 12.55GB，未提交新 CREATE 作业。

## 结果核查

```sh
.venv-pytorch/bin/python scripts/report_m3w_interaction_feature_cost.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_interaction_feature_cost.py tests/test_m3w_interaction_feature_summary.py tests/test_m3w_interaction_feature_roles.py tests/test_m3w_frozen_context_source.py tests/test_m3w_frozen_interaction_capture.py tests/test_m3w_bound_aware_cost.py
```

报告程序从全部公开轻量分组重建规范汇总，并核对原指标、训练记录和 SHA。
它不是重新训练，也不是从原始视频重做实验。完整原始复现还需要合法的私有
数据、冻结预测器权重和缓存；公开标量重建不能代替匿名冷启动复现包。

`registration.json` 保留首次定义，`registration_v2.json` 记录训练角色修正。
首次失败试跑在任何参数更新前停止，但确实计算过临时 H 组标签预处理；
该统计已丢弃，未用于模型或阈值选择。实际 36 个 checkpoint 的训练行和
归一化都另外核查为 G 组。不要把失败记录删掉或称为从未访问过 H 标签。

当前只支持像素、raw-frame 和 detector/tracker silver 声明。
没有执行 Stage5C，没有启用 SMC，也没有进行论文投稿。
