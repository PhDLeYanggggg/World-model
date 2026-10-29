# 录像约束均衡实验：复现与恢复

## 本轮是什么

这是一次新的神经选择策略训练，不是新轨迹预测器训练。只把约束损失的
录像权重改为相等，保留原收益目标、2% 风险要求、输入和训练步数。
18 个模型、36,000 次更新均已完成；18 个最终权重在读取结果前提交哈希。
结果是训练层面的部分改善，完整策略仍失败，不部署。

运行目录：`/Users/yangyue/Downloads/World`。
本机环境：原生 arm64 `.venv-pytorch`，计算线程4、interop1、数据加载进程0。
不要用历史 x86_64 Conda/MKL 环境替代。试跑已经验证真实更新与精确断点恢复，
不是仅验证能否 import torch。

## 已有产物核验

以下命令读取冻结权重与训练数据，重算同一批训练结果，不重新训练，不开新测试：

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy check
.venv-pytorch/bin/python -m scripts.report_m3w_record_balanced_policy --check
.venv-pytorch/bin/python -m scripts.analyze_m3w_record_balanced_failures --check
```

第一条需要已授权的 CREATE 只读连接。源 TRAIN 数据包通过 SSH 传到本机内存，
读取前校验大小、SHA256、所属实验和 TRAIN 清单。不是把全部数据保存在本地。
训练和推理计算都在本机，没有新提交 CREATE 训练作业。

私有权重位于：
`data/stage_cvpr2027_experiments/european_record_balanced_policy_v1/heads/`。
公开权重清单位于同名输出目录的 `fits/` 和 `training_freeze.json`。
只有公开仓库、没有授权数据和私有权重，不能声称完成完整复现。

## 原始运行顺序

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy train
# 冻结并提交全部最终权重的轻量清单后，才读取结果。
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy evaluate
```

本次上述阶段都已完成，程序会拒绝覆盖完成阶段。不要删除完成标记或权重来强行重跑。
若未来需要独立重复训练，应使用单独注册的实验目录，不能覆盖本轮证据。

## 中断恢复

每100次更新保存原子检查点，每50次更新保存心跳，进程锁防止同一实验重复启动。
先核对当前 PID、心跳与最终完成清单；进程健康时继续等待。
只有明确中断且未完成的阶段使用相应恢复命令：

```sh
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy train --resume
.venv-pytorch/bin/python -m scripts.run_m3w_record_balanced_policy evaluate --resume
```

只运行实际中断的那一条。保留10GiB磁盘余量、128MiB检查点上限和12小时单次运行边界。
不要把 SSH 暂时不可用当成远程训练失败或已完成，更不要因此重复提交作业。
本轮已捕获训练 PID28251、读取结果 PID30534、重放 PID31124 的退出码0。

## 怎样读结果

录像约束通过数从84/444升至131/444，但仍有289个视图的已知总正伤害超限。
444个视图对应74段唯一录像在不同预测上下文和种子下的重复观察，不是444个独立样本。
简单样本的净误差退化都在2%内，并不代表正伤害比例也合格。

完整策略效用相对上一版略降；同切换数量的效用略升。后者只回答效用问题，
不能代替完整策略的逐场景安全检查。这里的区间是未知标签的结果补全界，
不是置信区间，也不是 FDE 改善百分比。旧失败结论不改写。

全部72个读取结果精确重放，1,836项策略摘要算术一致；68项回归测试及3项新诊断测试通过。
没有运行完整历史测试集，没有打开新的独立选择、风险校准或最终确认数据。
因此下一步应修复候选预测支持与约束拟合，不应根据测试结果再调阈值。
Stage5C、SMC、metric/seconds 声明和部署升级均未启用。
