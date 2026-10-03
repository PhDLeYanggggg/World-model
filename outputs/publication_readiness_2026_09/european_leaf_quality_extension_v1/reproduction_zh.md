# 训练预测不变的叶内质量外推对照

## 范围

本实验不重新训练模型。使用训练数据确定每个树叶子中七个历史质量特征的
最小值和最大值，只在查询超出范围时截到边界，再使用原来的正值成本函数。
所有已知标签训练样本的预测必须逐位相同。

四个对照为原始森林、加性伤害头、Poisson 正值头和平方成本正值头。
验证四个对照的权重、输入、预测、动作及分数后，重新计算新的外推策略。
未来标签只能用于训练边界所属样本的既定身份核对和离线评价，不进入预测接口。
未知验证标签保留为未知，不剔除、不记为零风险。

这是已经暴露的开发数据上的机制实验。独立选择、校准、确认角色保持关闭，
不据此宣称独立泛化、物理安全或部署成功。

## 环境与保存

目录：`/Users/yangyue/Downloads/World`。
解释器：原生 arm64 `.venv-pytorch/bin/python`，4 个计算线程，0 个 worker。
注册提交：`c03c8de0`；规则在 `protocol.md`、参数在
`configs/m3w_european_leaf_quality_extension_v1.json`。

CREATE 只读取 M3W 自有权重，保持严格主机密钥校验、只读范围和超时保护。
没有新调度作业，没有新增权重或数值缓存。本地轻量结果在本目录 `groups/`；
心跳、事件及恢复锁在 `data/stage_cvpr2027_experiments/european_leaf_quality_extension_v1/`。

## 运行与中断恢复

首次运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_leaf_quality_extension.py pilot
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_leaf_quality_extension.py run
```

如运行中断，先核实旧进程已经结束且没有 `complete.json`，再在第二条命令末尾
加 `--resume`。已有结果会重算后逐项核对，不允许用不同配置覆盖。

已完成实验需要重新验证真实推理时，使用 `verify`，不重新训练：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pytorch/bin/python scripts/run_m3w_leaf_quality_extension.py verify
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/verify_m3w_leaf_quality_extension.py
env PYTHONDONTWRITEBYTECODE=1 \
  .venv-pytorch/bin/python scripts/summarize_m3w_leaf_quality_extension.py
```

第一条读取权重并重放推理；后两条核对已有报告的独立统计运算并整理表格。
不能把仅重读报告说成重新训练。

## 结果判定

真实试跑已完成：50.37 秒，峰值内存 7.11 GB，299 项独立策略标量检查通过。
全量已完成：72 个头，426.82 秒，峰值内存 10.297 GB；读取并核验旧权重
586.975 MB，没有新训练、新数值缓存或新调度任务。运行内 21,528 项标量检查、
独立复核 12,660 项标量/bootstrap 检查和 10 项针对性测试通过。
没有重跑整个旧测试库，不能把既有检查全部记成本轮新检查。
结果由 `complete.json`、`summary.json`、`verification.json` 共同证明。
训练不变、程序退出、测试通过都不等于科学门槛通过。

结果：相对平方成本头，标准化成本分数 MSE 降低 0.126005，但改善几乎全在
未选样本上；仅 10/72 个头改变动作。4 个已知标签风险失败和 7 个上界失败仍在，
最差上界 5.7195%，高于 2% 预算。相对提升和绝对风险门槛都未通过，不升级部署。
这不是 ADE/FDE 改善值，也不是新的神经网络世界动力学训练结果。
详见 `conclusions.md` 和 `findings.md`。当前长期目标为暂停，本轮只完成已有实验归档。

保持原有严格比较：对四个对照的成本误差和完整/同数量介入收益均须有支持，
且支持度、风险失败数和最差上界不得恶化。相对对照门槛与绝对风险门槛分别报告。
风险仍是选中 easy 样本的正伤害总量除以相应参考误差总量，预算 2%。
不能以全体 easy 的净退化替代，也不能放宽预算来通过。

bootstrap 为 3,000 次 locality 重采样；72 个头和重叠窗口不是独立样本。
保留 detector-silver、image-local、obs8/pred12、rawstride12 表述。
Stage5C/SMC 关闭，不声明 metric、seconds、human gold、true 3D 或 foundation。
