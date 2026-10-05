# CREATE 训练与恢复

这里延续已注册的时序辅助 cost-head 实验，不重做预测器，不改变数据划分、
损失、2% 风险预算或最终检查点评价规则。独立测试角色仍关闭。
实时状态看 `create_execution_status.json`；成功提交或排队不等于训练完成。

## 已核实的边界

本地使用原生 arm64 `.venv-pytorch`，四线程、零 DataLoader worker。
CREATE 使用已独立建立的 M3W CPU 环境，环境作业 37560918 的成功状态、
真实合成优化和精确恢复记录均已核验。此环境验证本身不是科研训练结果。
不安装到 simulation 环境，不修改其他项目作业，不在登录节点训练。

本机保留原 10 GiB 存储保护，因此只在内存中准备训练包。
CREATE 同样保留 10 GiB 个人配额余量，另限制总输入 4 GiB、检查点 256 MiB。
每个源分区的不同随机种子共享数值相同的输入包。未来标签只存监督字段，
不拼接到推理特征。训练包不包含验证行或新的独立测试角色。

## 入口

在项目目录使用以下入口。SSH 参数来自已经审计的本地私有 handoff，
不更换凭据或修改系统 SSH 设置。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create inspect
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create export-pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create submit-pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create status
```

`submit-pilot` 已提交作业 **37790290**，不能重复提交。即使 SSH 返回超时，
也先核对远程 submission intent、receipt 和调度记录，不能把未知状态当作失败。
传输包是不可变、按哈希幂等核验的；传输中断后可重试，但不会重复提交训练。
首次文件名校验失败及修复见 `create_port_filename_fix.md`。

## 完整训练准入

试跑在首个 TRAIN 源上为三组各更新 100 次，包含中断恢复与连续运行逐项核对。
只有作业成功、恢复一致、内存满足要求且完整训练估计在 12 小时内，才运行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create export-train
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create submit-train
```

完整预算为 72 个来源/种子组合乘三组，共 216 个 cost-head 拟合，每个固定
2,000 次更新。试跑检查点直接继续，不重新选择种子或超参数。
每 100 次更新原子保存权重、优化器和随机状态；心跳同时记录损失与来源。
中断时保留最后一个完整检查点。恢复前必须核对终止状态、身份和输入哈希；
不能有两个作业同时写同一实验。重新提交需新恢复记录，不能删除旧提交意图。

## 结果准入

必须先核验 216 个最终检查点并冻结训练记录，再执行已经注册的七组 readout。
四个强对照不能省略。没有完成 readout、标量重放和场景级不确定性核验前，
不得声称时序辅助提高策略效果。部分试跑不是完整实验，checkpoint 不是过 gate。
同一运行环境内要求精确恢复；不预先声称不同 CPU 架构的训练逐位一致。

继续使用 image-pixel / raw-frame / detector-silver 的证据范围，
不声明米、秒、人工金标准、true 3D 或 foundation。Stage5C 和 SMC 不执行。
